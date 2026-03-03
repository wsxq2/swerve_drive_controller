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

#include "swerve_drive_controller/swerve_drive_kinematics.hpp"

namespace swerve_drive_controller
{

SwerveDriveKinematics::SwerveDriveKinematics()
: odometry_{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  prev_wheel_positions_{{0.0, 0.0, 0.0, 0.0}},
  first_update_(true)
{
}

void SwerveDriveKinematics::configure(
  double wheel_base, double track_width, double wheel_radius,
  double x_offset, double y_offset)
{
  double half_length = wheel_base / 2.0;
  double half_width = track_width / 2.0;

  wheel_positions_[0] = {half_length - x_offset, half_width - y_offset};    // Front Left  (+x, +y)
  wheel_positions_[1] = {half_length - x_offset, -half_width - y_offset};   // Front Right (+x, -y)
  wheel_positions_[2] = {-half_length - x_offset, half_width - y_offset};   // Rear Left   (-x, +y)
  wheel_positions_[3] = {-half_length - x_offset, -half_width - y_offset};  // Rear Right  (-x, -y)

  wheel_radius_ = wheel_radius;
}

std::array<WheelCommand, 4> SwerveDriveKinematics::compute_wheel_commands(
  double linear_velocity_x, double linear_velocity_y, double angular_velocity_z)
{
  std::array<WheelCommand, 4> wheel_commands;

  if (wheel_radius_ <= 0.0)
  {
    std::cerr << "invalid wheel_radius_ <= 0.0\n";
    // fallthrough: compute but set angular velocities to 0 to avoid div-by-zero
  }

  for (std::size_t i = 0; i < 4; i++)
  {
    const auto & [wx, wy] = wheel_positions_[i];

    double vx = linear_velocity_x - angular_velocity_z * wy;
    double vy = linear_velocity_y + angular_velocity_z * wx;

    double linear_speed = std::hypot(vx, vy);
    double steering = std::atan2(vy, vx);

    if (wheel_radius_ > 0.0)
    {
      wheel_commands[i].drive_angular_velocity = linear_speed / wheel_radius_;  // rad/s
    }
    else
    {
      wheel_commands[i].drive_angular_velocity = 0.0;  // safe fallback
    }

    wheel_commands[i].steering_angle = steering;
  }

  return wheel_commands;
}

std::array<WheelCommand, 4> SwerveDriveKinematics::optimize_wheel_commands(
  const std::array<WheelCommand, 4> & wheel_commands,
  const std::array<double, 4> & current_steering_angles)
{
  std::array<WheelCommand, 4> optimized_commands = wheel_commands;
  
  auto reverse_angle_and_velocity = [](double &angle, double &velocity) {
    velocity = -velocity;
    angle = angles::normalize_angle(angle + M_PI);
  };

  for (std::size_t i = 0; i < 4; i++)
  {
    double target_angle = wheel_commands[i].steering_angle;
    double current_angle = current_steering_angles[i];

    double angle_diff = angles::shortest_angular_distance(current_angle, target_angle);
    
    if(std::abs(angle_diff) > M_PI_2)
    {
      reverse_angle_and_velocity(optimized_commands[i].steering_angle, optimized_commands[i].drive_angular_velocity);
    }
    
    target_angle = optimized_commands[i].steering_angle;  // update target angle after potential reversal

    // ensure target angle is within +/- 120 degrees (actual use 119)
    if (std::abs(target_angle) > 2.07694)
    {
      reverse_angle_and_velocity(optimized_commands[i].steering_angle, optimized_commands[i].drive_angular_velocity);
    }
  }

  return optimized_commands;
}

OdometryState SwerveDriveKinematics::update_odometry(
  const std::array<double, 4> & wheel_angular_velocities,
  const std::array<double, 4> & steering_angles,
  double dt)
{
  // Compute robot-centric velocity (assuming perfect wheel control)
  double vx_sum = 0.0, vy_sum = 0.0, wz_sum = 0.0;
  for (std::size_t i = 0; i < 4; i++)
  {
    // Convert angular velocity to linear velocity
    double wheel_linear_velocity = wheel_angular_velocities[i] * wheel_radius_;
    
    double vx = wheel_linear_velocity * std::cos(steering_angles[i]);
    double vy = wheel_linear_velocity * std::sin(steering_angles[i]);

    // Accumulate contributions to linear and angular velocities
    vx_sum += vx;
    vy_sum += vy;

    wz_sum += (vy * wheel_positions_[i].first - vx * wheel_positions_[i].second);
  }

  double vx_robot = vx_sum / 4.0;
  double vy_robot = vy_sum / 4.0;

  double wz_denominator = 0.0;
  for (std::size_t i = 0; i < 4; i++)
  {
    wz_denominator +=
      (wheel_positions_[i].first * wheel_positions_[i].first +
       wheel_positions_[i].second * wheel_positions_[i].second);
  }
  double wz_robot = wz_sum / wz_denominator;

  // Transform velocities to global frame
  double cos_theta = std::cos(odometry_.theta);
  double sin_theta = std::sin(odometry_.theta);

  double vx_global = vx_robot * cos_theta - vy_robot * sin_theta;
  double vy_global = vx_robot * sin_theta + vy_robot * cos_theta;

  // Integrate to compute new position and orientation
  odometry_.x += vx_global * dt;
  odometry_.y += vy_global * dt;
  odometry_.theta = angles::normalize_angle(odometry_.theta + wz_robot * dt);
  odometry_.vx = vx_robot;
  odometry_.vy = vy_robot;
  odometry_.wz = wz_robot;
  return odometry_;
}

OdometryState SwerveDriveKinematics::update_odometry(
  const std::array<double, 4> & wheel_positions,
  const std::array<double, 4> & wheel_angular_velocities,
  const std::array<double, 4> & steering_angles,
  double dt)
{
  // dt parameter kept for API consistency, but not needed for position-based calculation
  (void)dt;
  
  // First call: only save current positions, don't update odometry
  if (first_update_) {
    prev_wheel_positions_ = wheel_positions;
    first_update_ = false;
    
    return odometry_;
  }
  
  // ============ 1. Use position deltas to calculate position odometry (x, y, theta) ============
  
  // Calculate wheel position deltas (radians)
  std::array<double, 4> wheel_deltas;
  for (std::size_t i = 0; i < 4; ++i) {
    wheel_deltas[i] = wheel_positions[i] - prev_wheel_positions_[i];
  }
  
  // Convert wheel position deltas to linear distance deltas (meters)
  std::array<double, 4> wheel_distances;
  for (std::size_t i = 0; i < 4; ++i) {
    wheel_distances[i] = wheel_deltas[i] * wheel_radius_;
  }
  
  // Calculate robot center position change (robot frame)
  double dx_sum = 0.0;
  double dy_sum = 0.0;
  double dtheta_sum = 0.0;
  
  for (std::size_t i = 0; i < 4; ++i) {
    // Decompose wheel movement into robot frame
    double dx = wheel_distances[i] * std::cos(steering_angles[i]);
    double dy = wheel_distances[i] * std::sin(steering_angles[i]);
    
    dx_sum += dx;
    dy_sum += dy;
    
    // Calculate contribution to angular change
    dtheta_sum += (dy * wheel_positions_[i].first - dx * wheel_positions_[i].second);
  }
  
  // Average position change (robot frame)
  double dx_robot = dx_sum / 4.0;
  double dy_robot = dy_sum / 4.0;
  
  // Calculate angular change
  double wz_denominator = 0.0;
  for (std::size_t i = 0; i < 4; ++i) {
    wz_denominator +=
      (wheel_positions_[i].first * wheel_positions_[i].first +
       wheel_positions_[i].second * wheel_positions_[i].second);
  }
  double dtheta = dtheta_sum / wz_denominator;
  
  // Transform to global frame (using current orientation)
  double cos_theta = std::cos(odometry_.theta);
  double sin_theta = std::sin(odometry_.theta);
  
  double dx_global = dx_robot * cos_theta - dy_robot * sin_theta;
  double dy_global = dx_robot * sin_theta + dy_robot * cos_theta;
  
  // Update position and orientation
  odometry_.x += dx_global;
  odometry_.y += dy_global;
  odometry_.theta = angles::normalize_angle(odometry_.theta + dtheta);
  
  // ============ 2. Use angular velocities to calculate velocity odometry (vx, vy, wz) ============
  
  double vx_sum = 0.0;
  double vy_sum = 0.0;
  double wz_sum = 0.0;
  
  for (std::size_t i = 0; i < 4; ++i) {
    // Convert angular velocity to linear velocity
    double wheel_linear_velocity = wheel_angular_velocities[i] * wheel_radius_;
    
    // Decompose wheel velocity into robot frame
    double vx = wheel_linear_velocity * std::cos(steering_angles[i]);
    double vy = wheel_linear_velocity * std::sin(steering_angles[i]);
    
    vx_sum += vx;
    vy_sum += vy;
    
    // Calculate contribution to angular velocity
    wz_sum += (vy * wheel_positions_[i].first - vx * wheel_positions_[i].second);
  }
  
  // Update velocity (robot frame)
  odometry_.vx = vx_sum / 4.0;
  odometry_.vy = vy_sum / 4.0;
  odometry_.wz = wz_sum / wz_denominator;
  
  // Save current positions for next update
  prev_wheel_positions_ = wheel_positions;
  
  return odometry_;
}

}  // namespace swerve_drive_controller
