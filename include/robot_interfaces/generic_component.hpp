#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

// Hardware Interface
#include "hardware_interface/loaned_command_interface.hpp"
#include "hardware_interface/loaned_state_interface.hpp"

// RCLCPP (Logging only)
#include "rclcpp/rclcpp.hpp"

// Pinocchio
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "command_types.hpp"

namespace robot_interfaces
{
  /**
   * @brief Generic robot interfaces for ros2_control logic.
   * Handles interface management and pinocchio-based kinematics and dynamics.
   */
  class GenericComponent
  {
  public:
    explicit GenericComponent(const std::string &name, size_t state_interface_size = 0,
                              size_t command_interface_size = 0);

    virtual ~GenericComponent() = default;

    // ==================== Interface Management ====================

    bool assign_loaned_state(
        std::vector<hardware_interface::LoanedStateInterface> &state_interfaces);
    bool assign_loaned_command(
        std::vector<hardware_interface::LoanedCommandInterface> &command_interfaces);
    void release_all_interfaces();

    // ==================== Getters / Setters ====================

    virtual void set_states_names();
    virtual void set_commands_names(std::vector<std::string> custom_names = {});

    std::vector<std::string> get_states_names() const
    {
      return state_names;
    }
    std::vector<std::string> get_commands_names() const
    {
      return command_names;
    }

    std::vector<double> get_states_values() const;
    std::vector<double> get_command_values() const;
    bool set_values(const std::vector<double> &commanded_values);

    // ==================== Kinematics (Initialization) ====================

    /**
     * @brief High-level initialization for kinematics.
     * Called during controller on_configure.
     */
    virtual bool initKinematics(const std::string &urdf_xml, const std::string &tool_frame);
    // ==================== Abstract Methods ====================

    virtual bool setCommand(const CommandVariant &command) = 0;

    // ==================== Helpers Methods ====================
    CartesianPosition getCurrentEndEffectorPose() const;
    JointCommand getCurrentJointPose() const;
    Eigen::MatrixXd getEndEffectorJacobian() const;
    Eigen::MatrixXd getMassMatrix() const;
    Eigen::VectorXd getNonLinearEffects() const;
    Eigen::MatrixXd getCoriolisMatrix() const;
    Eigen::VectorXd getGravityEffort() const;

    const Eigen::VectorXd &getJointPositions() const
    {
      return q;
    }
    const Eigen::VectorXd &getJointVelocities() const
    {
      return v;
    }
    const Eigen::VectorXd &getJointTorques() const
    {
      return tau;
    }

    /// @brief function to call at the start of update, in order to update pinocchio data with new
    /// inputs from sensor (q, v, tau)
    void syncState() const;

  protected:
    // ==================== Data Members ====================
    std::string component_name;

    // Names used to "find" interfaces in the HardwareManager
    std::vector<std::string> state_names;
    std::vector<std::string> command_names;

    // Actual references to the hardware data
    std::vector<std::reference_wrapper<hardware_interface::LoanedStateInterface>> state_interfaces;
    std::vector<std::reference_wrapper<hardware_interface::LoanedCommandInterface>>
        command_interfaces;

    pinocchio::Model model;
    mutable pinocchio::Data data;
    pinocchio::FrameIndex tool_frame_id;

    mutable Eigen::VectorXd q;
    mutable Eigen::VectorXd v;
    mutable Eigen::VectorXd tau;

    struct JointInterfaceIndices
    {
      int position = -1;
      int velocity = -1;
      int effort = -1;
    };
    std::vector<JointInterfaceIndices> joint_map;
    mutable CartesianPosition last_valid_pose_;
  };
} // namespace robot_interfaces