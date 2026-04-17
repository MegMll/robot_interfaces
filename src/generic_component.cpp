#include "robot_interfaces/generic_component.hpp"

#include "controller_interface/helpers.hpp"
#include <sstream>
#include <unordered_map>

namespace robot_interfaces
{
  GenericComponent::GenericComponent(const std::string &name, size_t state_interface_size,
                                     size_t command_interface_size)
      : component_name(name)
  {
    state_names.reserve(state_interface_size);
    command_names.reserve(command_interface_size);

    state_interfaces.reserve(state_interface_size);
    command_interfaces.reserve(command_interface_size);
  }

  bool GenericComponent::assign_loaned_state(
      std::vector<hardware_interface::LoanedStateInterface> &state_int)
  {
    return controller_interface::get_ordered_interfaces(state_int, state_names, "",
                                                        state_interfaces);
  }

  bool GenericComponent::assign_loaned_command(
      std::vector<hardware_interface::LoanedCommandInterface> &command_int)
  {
    return controller_interface::get_ordered_interfaces(command_int, command_names, "",
                                                        command_interfaces);
  }

  void GenericComponent::release_all_interfaces()
  {
    state_interfaces.clear();
    command_interfaces.clear();
  }

  void GenericComponent::set_states_names()
  {
    if (state_names.empty())
    {
      for (size_t i = 0; i < state_names.capacity(); ++i)
      {
        state_names.emplace_back(component_name + "/" + std::to_string(i + 1));
      }
    }
  }

  void GenericComponent::set_commands_names(std::vector<std::string> custom_names)
  {
    for (size_t i = 0; i < custom_names.size(); ++i)
    {
      command_names.emplace_back(custom_names[i]);
    }
  }

  std::vector<double> GenericComponent::get_states_values() const
  {
    std::vector<double> state_values;
    // Collect current values from all state interfaces
    for (const auto &state_interface : state_interfaces)
    {
      state_values.emplace_back(state_interface.get().get_value());
    }
    return state_values;
  }

  std::vector<double> GenericComponent::get_command_values() const
  {
    std::vector<double> command_values;
    // Collect current values from all command interfaces
    for (const auto &command_interface : command_interfaces)
    {
      command_values.emplace_back(command_interface.get().get_value());
    }

    return command_values;
  }

  bool GenericComponent::set_values(const std::vector<double> &values)
  {
    // Verify size matches before setting values
    if (values.size() != command_interfaces.size())
    {
      return false;
    }
    // Set each command interface with provided value
    for (size_t i = 0; i < command_interfaces.size(); ++i)
    {
      command_interfaces[i].get().set_value(values[i]);
    }

    return true;
  }

  bool GenericComponent::initKinematics(const std::string &urdf_xml, const std::string &tool_frame)
  {
    try
    {
      pinocchio::urdf::buildModelFromXML(urdf_xml, model);
      data = pinocchio::Data(model);
      tool_frame_id = model.getFrameId(tool_frame);

      q.setZero(model.nq);
      v.setZero(model.nv);
      tau.setZero(model.nv);

      joint_map.clear();
      // Pinocchio joints: 0 is 'universe', 1..n are actual joints
      for (size_t i = 1; i < (size_t)model.njoints; ++i)
      {
        std::string name = model.names[i];
        JointInterfaceIndices indices;

        for (size_t j = 0; j < state_names.size(); ++j)
        {
          if (state_names[j].find(name) != std::string::npos)
          {
            if (state_names[j].find("/position") != std::string::npos)
              indices.position = j;
            else if (state_names[j].find("/velocity") != std::string::npos)
              indices.velocity = j;
            else if (state_names[j].find("/effort") != std::string::npos)
              indices.effort = j;
          }
        }
        joint_map.push_back(indices);
      }
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR(rclcpp::get_logger(component_name), "Pinocchio Init: %s", e.what());
      return false;
    }
    return true;
  }

  void GenericComponent::syncState() const
  {
    for (size_t i = 0; i < joint_map.size(); ++i)
    {
      if (joint_map[i].position != -1)
        q(i) = state_interfaces[joint_map[i].position].get().get_value();
      if (joint_map[i].velocity != -1)
        v(i) = state_interfaces[joint_map[i].velocity].get().get_value();
      if (joint_map[i].effort != -1)
        tau(i) = state_interfaces[joint_map[i].effort].get().get_value();
    }

    pinocchio::forwardKinematics(model, data, q, v);
    pinocchio::computeJointJacobians(model, data, q);
    pinocchio::updateFramePlacements(model, data);
  }

  CartesianPosition GenericComponent::getCurrentEndEffectorPose() const
  {
    const auto &pose_m = data.oMf[tool_frame_id];
    CartesianPosition pose;
    pose.translation = pose_m.translation();
    pose.quaternion = Eigen::Quaterniond(pose_m.rotation());

    // Continuity check
    if (pose.quaternion.dot(last_valid_pose_.quaternion) < 0.0)
      pose.quaternion.coeffs() *= -1.0;

    last_valid_pose_ = pose;
    return pose;
  }

  Eigen::MatrixXd GenericComponent::getEndEffectorJacobian() const
  {
    Eigen::MatrixXd J(6, model.nv);
    pinocchio::getFrameJacobian(model, data, tool_frame_id,
                                pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J);
    return J;
  }

  Eigen::MatrixXd GenericComponent::getMassMatrix() const
  {
    pinocchio::crba(model, data, q);
    data.M.triangularView<Eigen::Lower>() = data.M.transpose().triangularView<Eigen::Lower>();
    return data.M;
  }

  Eigen::VectorXd GenericComponent::getNonLinearEffects() const
  {
    // Returns Coriolis + Gravity: b(q, v)
    return pinocchio::nonLinearEffects(model, data, q, v);
  }

  Eigen::MatrixXd GenericComponent::getCoriolisMatrix() const
  {
    // Returns Coriolis Matrix C
    return pinocchio::computeCoriolisMatrix(model, data, q, v);
  }

  Eigen::VectorXd GenericComponent::getGravityEffort() const
  {
    // Returns Gravity
    return pinocchio::computeGeneralizedGravity(model, data, q);
  }
} // namespace robot_interfaces