// Copyright 2025 ros2_control development team
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "swerve_drive_controller/swerve_drive_controller.hpp"
#include <angles/angles.h>

#include <cmath>
#include <memory>
#include <rclcpp/time.hpp>
#include <string>
#include <utility>
#include <vector>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "lifecycle_msgs/msg/state.hpp"
#include "rclcpp/logging.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace
{

constexpr auto DEFAULT_COMMAND_TOPIC = "~/cmd_vel";
constexpr auto DEFAULT_COMMAND_UNSTAMPED_TOPIC = "~/cmd_vel_unstamped";
constexpr auto DEFAULT_ODOMETRY_TOPIC = "~/odom";
constexpr auto DEFAULT_TRANSFORM_TOPIC = "/tf";

}  // namespace

namespace swerve_drive_controller
{

using namespace std::chrono_literals;
using controller_interface::interface_configuration_type;
using controller_interface::InterfaceConfiguration;
using hardware_interface::HW_IF_POSITION;
using hardware_interface::HW_IF_VELOCITY;
using lifecycle_msgs::msg::State;

constexpr std::size_t NUM_WHEELS = 4;
constexpr std::size_t NUM_DIMENSIONS = 6;
constexpr double DEFAULT_COVARIANCE = 0.01;
constexpr double SECONDS_TO_MILLISECONDS = 1000.0;

Wheel::Wheel(
  std::reference_wrapper<hardware_interface::LoanedCommandInterface> velocity,
  std::reference_wrapper<const hardware_interface::LoanedStateInterface> feedback, std::string name)
: velocity_(velocity), feedback_(feedback), name_(std::move(name))
{
}

void Wheel::set_velocity(double velocity) { velocity_.get().set_value(velocity); }

double Wheel::get_feedback() { return Wheel::feedback_.get().get_value(); }

Axle::Axle(
  std::reference_wrapper<hardware_interface::LoanedCommandInterface> position,
  std::reference_wrapper<const hardware_interface::LoanedStateInterface> feedback, std::string name)
: position_(position), feedback_(feedback), name_(std::move(name))
{
}

void Axle::set_position(double position) { position_.get().set_value(position); }

double Axle::get_feedback() { return Axle::feedback_.get().get_value(); }

SwerveController::SwerveController()
: controller_interface::ControllerInterface(), swerveDriveKinematics_()
{
}

CallbackReturn SwerveController::on_init()
{
  node_ = get_node();
  if (!node_)
  {
    RCLCPP_ERROR(logger_, "Node is null in on_init");
    return controller_interface::CallbackReturn::ERROR;
  }
  logger_ = node_->get_logger();
  try
  {
    // Create the parameter listener and get the parameters
    param_listener_ = std::make_shared<ParamListener>(node_);
    params_ = param_listener_->get_params();

    swerveDriveKinematics_.configure(
      params_.wheelbase, params_.trackwidth, params_.wheel_radius,
      params_.offset[0], params_.offset[1]);
  }
  catch (const std::exception & e)
  {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

InterfaceConfiguration SwerveController::command_interface_configuration() const
{
  std::vector<std::string> conf_names;
  conf_names.push_back(params_.front_left_wheel_joint + "/" + HW_IF_VELOCITY);
  conf_names.push_back(params_.front_right_wheel_joint + "/" + HW_IF_VELOCITY);
  conf_names.push_back(params_.rear_left_wheel_joint + "/" + HW_IF_VELOCITY);
  conf_names.push_back(params_.rear_right_wheel_joint + "/" + HW_IF_VELOCITY);
  conf_names.push_back(params_.front_left_axle_joint + "/" + HW_IF_POSITION);
  conf_names.push_back(params_.front_right_axle_joint + "/" + HW_IF_POSITION);
  conf_names.push_back(params_.rear_left_axle_joint + "/" + HW_IF_POSITION);
  conf_names.push_back(params_.rear_right_axle_joint + "/" + HW_IF_POSITION);
  return {interface_configuration_type::INDIVIDUAL, conf_names};
}

InterfaceConfiguration SwerveController::state_interface_configuration() const
{
  std::vector<std::string> conf_names;
  conf_names.push_back(params_.front_left_wheel_joint + "/" + HW_IF_VELOCITY);
  conf_names.push_back(params_.front_right_wheel_joint + "/" + HW_IF_VELOCITY);
  conf_names.push_back(params_.rear_left_wheel_joint + "/" + HW_IF_VELOCITY);
  conf_names.push_back(params_.rear_right_wheel_joint + "/" + HW_IF_VELOCITY);
  conf_names.push_back(params_.front_left_axle_joint + "/" + HW_IF_POSITION);
  conf_names.push_back(params_.front_right_axle_joint + "/" + HW_IF_POSITION);
  conf_names.push_back(params_.rear_left_axle_joint + "/" + HW_IF_POSITION);
  conf_names.push_back(params_.rear_right_axle_joint + "/" + HW_IF_POSITION);
  return {interface_configuration_type::INDIVIDUAL, conf_names};
}

CallbackReturn SwerveController::on_configure(const rclcpp_lifecycle::State & /*previous_state*/)
{
  try
  {
    // Initialize covariance diagonals
    for (std::size_t i = 0; i < NUM_DIMENSIONS; ++i)
    {
      params_.pose_covariance_diagonal[i] = DEFAULT_COVARIANCE;
      params_.twist_covariance_diagonal[i] = DEFAULT_COVARIANCE;
    }

    // Validate joint names
    const std::array<std::pair<const std::string&, const char*>, 8> joint_checks {{
      {params_.front_left_wheel_joint, "front_left_wheel_joint"},
      {params_.front_right_wheel_joint, "front_right_wheel_joint"},
      {params_.rear_left_wheel_joint, "rear_left_wheel_joint"},
      {params_.rear_right_wheel_joint, "rear_right_wheel_joint"},
      {params_.front_left_axle_joint, "front_left_axle_joint"},
      {params_.front_right_axle_joint, "front_right_axle_joint"},
      {params_.rear_left_axle_joint, "rear_left_axle_joint"},
      {params_.rear_right_axle_joint, "rear_right_axle_joint"}
    }};

    for (const auto & [joint_name, param_name] : joint_checks)
    {
      if (joint_name.empty())
      {
        RCLCPP_ERROR(logger_, "%s is not set", param_name);
        return CallbackReturn::ERROR;
      }
    }

    wheel_joint_names[0] = params_.front_left_wheel_joint;
    wheel_joint_names[1] = params_.front_right_wheel_joint;
    wheel_joint_names[2] = params_.rear_left_wheel_joint;
    wheel_joint_names[3] = params_.rear_right_wheel_joint;
    axle_joint_names[0] = params_.front_left_axle_joint;
    axle_joint_names[1] = params_.front_right_axle_joint;
    axle_joint_names[2] = params_.rear_left_axle_joint;
    axle_joint_names[3] = params_.rear_right_axle_joint;
    cmd_vel_timeout_ =
      std::chrono::milliseconds(static_cast<int>(params_.cmd_vel_timeout * SECONDS_TO_MILLISECONDS));

    // Initialize with zero velocity command
    auto zero_cmd = createZeroVelocityCommand(node_->get_clock()->now());
    received_velocity_msg_ptr_.writeFromNonRT(zero_cmd);

    if (params_.use_stamped_vel)
    {
      velocity_command_subscriber_ = node_->create_subscription<TwistStamped>(
        DEFAULT_COMMAND_TOPIC, rclcpp::SystemDefaultsQoS(),
        [this](const std::shared_ptr<TwistStamped> msg) -> void
        {
          if ((msg->header.stamp.sec == 0) && (msg->header.stamp.nanosec == 0))
          {
            RCLCPP_WARN_ONCE(
              logger_,
              "Received TwistStamped with zero timestamp, setting it to current "
              "time, this message will only be shown once");
            msg->header.stamp = node_->get_clock()->now();
          }
          received_velocity_msg_ptr_.writeFromNonRT(std::move(msg));
        });
    }
    else
    {
      velocity_command_unstamped_subscriber_ = node_->create_subscription<Twist>(
        DEFAULT_COMMAND_UNSTAMPED_TOPIC, rclcpp::SystemDefaultsQoS(),
        [this](const std::shared_ptr<Twist> msg) -> void
        {
          // Write fake header in the stored stamped command
          const std::shared_ptr<TwistStamped> twist_stamped =
            *(received_velocity_msg_ptr_.readFromRT());
          twist_stamped->twist = *msg;
          twist_stamped->header.stamp = node_->get_clock()->now();
          twist_stamped->header.frame_id = params_.base_footprint;
          received_velocity_msg_ptr_.writeFromNonRT(twist_stamped);
        });
    }

    auto odometry_publisher = node_->create_publisher<nav_msgs::msg::Odometry>(
      DEFAULT_ODOMETRY_TOPIC, rclcpp::SystemDefaultsQoS());

    realtime_odometry_publisher_ =
      std::make_shared<realtime_tools::RealtimePublisher<nav_msgs::msg::Odometry>>(
        odometry_publisher);

    std::string tf_prefix = "";
    tf_prefix = std::string(node_->get_namespace());
    if (tf_prefix == "/")
    {
      tf_prefix = "";
    }
    else
    {
      tf_prefix = tf_prefix + "/";
    }

    const auto odom_frame_id = tf_prefix + params_.odom;
    const auto base_frame_id = tf_prefix + params_.base_footprint;
    auto & odometry_message = realtime_odometry_publisher_->msg_;
    odometry_message.header.frame_id = odom_frame_id;
    odometry_message.child_frame_id = base_frame_id;

    odometry_message.twist =
      geometry_msgs::msg::TwistWithCovariance(rosidl_runtime_cpp::MessageInitialization::ALL);

    for (std::size_t index = 0; index < NUM_DIMENSIONS; ++index)
    {
      const std::size_t diagonal_index = NUM_DIMENSIONS * index + index;
      odometry_message.pose.covariance[diagonal_index] = params_.pose_covariance_diagonal[index];
      odometry_message.twist.covariance[diagonal_index] = params_.twist_covariance_diagonal[index];
    }
    auto odometry_transform_publisher = node_->create_publisher<tf2_msgs::msg::TFMessage>(
      DEFAULT_TRANSFORM_TOPIC, rclcpp::SystemDefaultsQoS());

    realtime_odometry_transform_publisher_ =
      std::make_shared<realtime_tools::RealtimePublisher<tf2_msgs::msg::TFMessage>>(
        odometry_transform_publisher);

    auto & odometry_transform_message = realtime_odometry_transform_publisher_->msg_;
    odometry_transform_message.transforms.resize(1);
    odometry_transform_message.transforms.front().header.frame_id = odom_frame_id;
    odometry_transform_message.transforms.front().child_frame_id = base_frame_id;
  }
  catch (const std::exception & e)
  {
    RCLCPP_ERROR(logger_, "EXCEPTION DURING on_configure: %s", e.what());
    return CallbackReturn::ERROR;
  }

  return CallbackReturn::SUCCESS;
}

CallbackReturn SwerveController::on_activate(const rclcpp_lifecycle::State &)
{
  wheel_handles_.resize(NUM_WHEELS);
  axle_handles_.resize(NUM_WHEELS);

  for (std::size_t i = 0; i < NUM_WHEELS; i++)
  {
    wheel_handles_[i] = get_wheel(wheel_joint_names[i]);
    if (!wheel_handles_[i])
    {
      RCLCPP_ERROR(logger_, "ERROR IN FETCHING wheel handle for: %s", wheel_joint_names[i].c_str());
      return CallbackReturn::ERROR;
    }
    wheel_handles_[i]->set_velocity(0.0);

    axle_handles_[i] = get_axle(axle_joint_names[i]);
    if (!axle_handles_[i])
    {
      RCLCPP_ERROR(logger_, "ERROR IN FETCHING axle handle for: %s", axle_joint_names[i].c_str());
      return CallbackReturn::ERROR;
    }
    axle_handles_[i]->set_position(0.0);
    previous_steering_angles_[i] = axle_handles_[i]->get_feedback();
  }

  RCLCPP_INFO(logger_, "Subscriber and publisher are now active.");
  return CallbackReturn::SUCCESS;
}

controller_interface::return_type SwerveController::update(
  const rclcpp::Time & time, const rclcpp::Duration & period)
{
  if (this->get_state().id() == State::PRIMARY_STATE_INACTIVE)
  {
    halt();
    return controller_interface::return_type::OK;
  }

  const auto current_time = time;

  auto last_command_msg = *(received_velocity_msg_ptr_.readFromRT());

  if (last_command_msg == nullptr)
  {
    RCLCPP_WARN(logger_, "last_command_msg is nullptr, which is not expected. Creating zero velocity command.");
    last_command_msg = createZeroVelocityCommand(current_time);
    received_velocity_msg_ptr_.writeFromNonRT(last_command_msg);
  }

  const auto age_of_last_command = current_time - last_command_msg->header.stamp;

  if (age_of_last_command > cmd_vel_timeout_)
  {
    last_command_msg->twist.linear.x = 0.0;
    last_command_msg->twist.linear.y = 0.0;
    last_command_msg->twist.angular.z = 0.0;
  }

  auto wheel_command = swerveDriveKinematics_.compute_wheel_commands(
    last_command_msg->twist.linear.x, last_command_msg->twist.linear.y,
    last_command_msg->twist.angular.z);

  std::array<double, NUM_WHEELS> current_steering_angles{};
  for (std::size_t i = 0; i < NUM_WHEELS; ++i)
  {
    current_steering_angles[i] = axle_handles_[i]->get_feedback();
  }

  double vx=std::abs(last_command_msg->twist.linear.x);
  double vy=std::abs(last_command_msg->twist.linear.y);
  double wz=std::abs(last_command_msg->twist.angular.z);
  if((vx < EPS && vy > EPS && wz < EPS) || (wz > EPS && vx < EPS && vy < EPS)) {
    // 这里为了解决将反转时机设置为 120 度后在平移时或者原地旋转时轮子会有不必要转向运动的问题
    // TODO: 是否有更好的方法？此外是否应该将相关代码整合到 optimize_wheel_commands 函数？
    for(size_t i = 0; i < NUM_WHEELS; ++i)
    {
      double angle_diff = angles::shortest_angular_distance(current_steering_angles[i], wheel_command[i].steering_angle);
      if (std::abs(angle_diff) > M_PI_2)
      {
        wheel_command[i].drive_angular_velocity = -wheel_command[i].drive_angular_velocity;

        wheel_command[i].steering_angle = angles::normalize_angle(wheel_command[i].steering_angle + M_PI);
      }
    }
  }
  else
  {
    // Optimize wheel commands by potentially reversing direction and adjusting steering angle
  wheel_command =
    swerveDriveKinematics_.optimize_wheel_commands(wheel_command, current_steering_angles);
  }

  // Apply velocity scaling based on steering error to prevent motion when wheels are misaligned
  constexpr double min_steering_error = 0.001;

  for (std::size_t i = 0; i < NUM_WHEELS; i++)
  {
    const double steering_error = std::abs(
      angles::shortest_angular_distance(
        current_steering_angles[i], wheel_command[i].steering_angle));

    double velocity_scale = 1.0;
#if 1
    if (steering_error > min_steering_error) {
      if (steering_error >= M_PI_2) {
        velocity_scale = 0.0;
      } else {
        velocity_scale = std::cos(steering_error);
      }
    }
#else
    if (steering_error > min_steering_error) {
      velocity_scale = 0.0;
    }
#endif

    wheel_command[i].drive_angular_velocity *= velocity_scale;
  }

  // Apply wheel commands to hardware interfaces
  const bool is_stop =
    (std::fabs(last_command_msg->twist.linear.x) < EPS) &&
    (std::fabs(last_command_msg->twist.linear.y) < EPS) &&
    (std::fabs(last_command_msg->twist.angular.z) < EPS);

  for (std::size_t i = 0; i < NUM_WHEELS; i++)
  {
    if (is_stop)
    {
      // When stopped, maintain current steering angle to avoid unnecessary motion
      axle_handles_[i]->set_position(previous_steering_angles_[i]);
    }
    else
    {
      // When moving, apply computed steering angle and update tracking
      axle_handles_[i]->set_position(wheel_command[i].steering_angle);
      previous_steering_angles_[i] = wheel_command[i].steering_angle;
    }
    wheel_handles_[i]->set_velocity(wheel_command[i].drive_angular_velocity);
  }

  // update odometry using period from controller manager
  swerve_drive_controller::OdometryState odometry_;
  std::array<double, NUM_WHEELS> velocity_array{};
  std::array<double, NUM_WHEELS> steering_angle_array{};
  for (std::size_t i = 0; i < NUM_WHEELS; ++i)
  {
    if (params_.open_loop)
    {
      velocity_array[i] = wheel_command[i].drive_angular_velocity * params_.wheel_radius;
      steering_angle_array[i] = wheel_command[i].steering_angle;
    }
    else
    {
      velocity_array[i] = wheel_handles_[i]->get_feedback() * params_.wheel_radius;
      steering_angle_array[i] = axle_handles_[i]->get_feedback();
    }
  }
  odometry_ = swerveDriveKinematics_.update_odometry(
    velocity_array, steering_angle_array, period.seconds());
  tf2::Quaternion orientation;
  orientation.setRPY(0.0, 0.0, odometry_.theta);

  // Publish odometry (frequency controlled by controller manager)
  if (realtime_odometry_publisher_->trylock()) {
    auto & odometry_message = realtime_odometry_publisher_->msg_;
    odometry_message.header.stamp = time;
    odometry_message.pose.pose.position.x = odometry_.x;
    odometry_message.pose.pose.position.y = odometry_.y;
    odometry_message.pose.pose.orientation.x = orientation.x();
    odometry_message.pose.pose.orientation.y = orientation.y();
    odometry_message.pose.pose.orientation.z = orientation.z();
    odometry_message.pose.pose.orientation.w = orientation.w();
    odometry_message.twist.twist.linear.x = odometry_.vx;
    odometry_message.twist.twist.linear.y = odometry_.vy;
    odometry_message.twist.twist.angular.z = odometry_.wz;
    realtime_odometry_publisher_->unlockAndPublish();
  }

  if (params_.enable_odom_tf && realtime_odometry_transform_publisher_->trylock()) {
    auto & transform = realtime_odometry_transform_publisher_->msg_.transforms.front();
    transform.header.stamp = time;
    transform.transform.translation.x = odometry_.x;
    transform.transform.translation.y = odometry_.y;
    transform.transform.translation.z = 0.0;
    transform.transform.rotation.x = orientation.x();
    transform.transform.rotation.y = orientation.y();
    transform.transform.rotation.z = orientation.z();
    transform.transform.rotation.w = orientation.w();
    realtime_odometry_transform_publisher_->unlockAndPublish();
  }

  return controller_interface::return_type::OK;
}

CallbackReturn SwerveController::on_deactivate(const rclcpp_lifecycle::State &)
{
  halt();
  wheel_handles_.clear();
  axle_handles_.clear();
  return CallbackReturn::SUCCESS;
}

CallbackReturn SwerveController::on_cleanup(const rclcpp_lifecycle::State &)
{
  // Resources are automatically released by destructors
  return CallbackReturn::SUCCESS;
}

CallbackReturn SwerveController::on_error(const rclcpp_lifecycle::State &)
{
  // Don't clear handles in error state - update() might still be running
  // Just halt motion safely
  halt();
  return CallbackReturn::SUCCESS;
}

CallbackReturn SwerveController::on_shutdown(const rclcpp_lifecycle::State &)
{
  return CallbackReturn::SUCCESS;
}

std::shared_ptr<SwerveController::TwistStamped> SwerveController::createZeroVelocityCommand(
  const rclcpp::Time & stamp) const
{
  auto zero_cmd = std::make_shared<TwistStamped>();
  zero_cmd->header.stamp = stamp;
  zero_cmd->twist.linear.x = 0.0;
  zero_cmd->twist.linear.y = 0.0;
  zero_cmd->twist.angular.z = 0.0;
  return zero_cmd;
}

void SwerveController::halt()
{
  for (std::size_t i = 0; i < wheel_handles_.size(); ++i)
  {
    wheel_handles_[i]->set_velocity(0.0);
  }
}

template <typename T>
std::unique_ptr<T> get_interface_object(
  std::vector<hardware_interface::LoanedCommandInterface> & command_interfaces,
  const std::vector<hardware_interface::LoanedStateInterface> & state_interfaces,
  const std::string & name, const std::string & interface_suffix, const std::string & hw_if_type,
  const rclcpp::Logger & logger)
{
  if (name.empty())
  {
    RCLCPP_ERROR(logger, "Joint name not given. Make sure all joints are specified.");
    return nullptr;
  }

  const std::string expected_interface_name = name + interface_suffix;

  auto command_handle = std::find_if(
    command_interfaces.begin(), command_interfaces.end(),
    [&expected_interface_name, &hw_if_type](const auto & interface)
    {
      return interface.get_name() == expected_interface_name &&
             interface.get_interface_name() == hw_if_type;
    });

  if (command_handle == command_interfaces.end())
  {
    RCLCPP_ERROR(logger, "Unable to find command interface for: %s", name.c_str());
    RCLCPP_ERROR(logger, "Expected interface name: %s", expected_interface_name.c_str());
    return nullptr;
  }

  const auto state_handle = std::find_if(
    state_interfaces.begin(), state_interfaces.end(),
    [&expected_interface_name, &hw_if_type](const auto & interface)
    {
      return interface.get_name() == expected_interface_name &&
             interface.get_interface_name() == hw_if_type;
    });

  if (state_handle == state_interfaces.end())
  {
    RCLCPP_ERROR(logger, "Unable to find the state interface for: %s", name.c_str());
    return nullptr;
  }
  static_assert(
    !std::is_const_v<std::remove_reference_t<decltype(*command_handle)>>,
    "Command handle is const!");
  return std::make_unique<T>(std::ref(*command_handle), std::ref(*state_handle), name);
}

std::unique_ptr<Wheel> SwerveController::get_wheel(const std::string & wheel_name)
{
  return get_interface_object<Wheel>(
    command_interfaces_, state_interfaces_, wheel_name, "/velocity", HW_IF_VELOCITY, logger_);
}

std::unique_ptr<Axle> SwerveController::get_axle(const std::string & axle_name)
{
  return get_interface_object<Axle>(
    command_interfaces_, state_interfaces_, axle_name, "/position", HW_IF_POSITION, logger_);
}

}  // namespace swerve_drive_controller

#include "class_loader/register_macro.hpp"

CLASS_LOADER_REGISTER_CLASS(
  swerve_drive_controller::SwerveController, controller_interface::ControllerInterface)
