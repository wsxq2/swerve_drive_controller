def normalize_angle(angle):
    """归一化到 [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def shortest_angular_distance(from_angle, to_angle):
    """计算最短角度差"""
    return normalize_angle(to_angle - from_angle)

def optimize_wheel_commands(wheel_commands, current_steering_angles):
    optimized_commands = []
    for i in range(4):
        target_angle = wheel_commands[i]["steering_angle"]
        current_angle = current_steering_angles[i]
        angle_diff = shortest_angular_distance(current_angle, target_angle)
        target_linear_velocity = wheel_commands[i].get("linear_speed")
        # 优化：目标角度超出 ±120°，反转驱动速度并加 π
        if abs(target_angle) > 2.0*np.pi / 3.0:
            target_linear_velocity = -target_linear_velocity
            target_angle = normalize_angle(target_angle + np.pi)
        optimized_commands.append({
            "steering_angle": target_angle,
            "linear_speed": target_linear_velocity,
            "position": wheel_commands[i]["position"]
        })
    return optimized_commands
import numpy as np
import matplotlib.pyplot as plt

def swerve_inverse_kinematics(vx, vy, wz, trackwidth, wheelbase):
    # 计算四个轮子的坐标（以底盘中心为原点）
    half_tw = trackwidth / 2
    half_wb = wheelbase / 2
    wheel_positions = [
        ( half_wb,  half_tw),  # 左前轮 (FL)
        ( half_wb, -half_tw),  # 右前轮 (FR)
        (-half_wb,  half_tw),  # 左后轮 (RL)
        (-half_wb, -half_tw),  # 右后轮 (RR)
    ]
    wheel_results = []
    for wx, wy in wheel_positions:
        vx_i = vx - wz * wy
        vy_i = vy + wz * wx
        linear_speed = np.hypot(vx_i, vy_i)
        steering_angle = np.arctan2(vy_i, vx_i)
        wheel_results.append({
            "linear_speed": linear_speed,
            "steering_angle": steering_angle,
            "position": (wx, wy)
        })
    return wheel_results



def plot_vs_wz(trackwidth, wheelbase, vx_fixed, v_y, w_z_range, current_steering_angles, if_optimize=True):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    vx = vx_fixed
    steer_angles = []
    steer_angles_RL = []
    steer_angles_FR = []
    steer_angles_RR = []
    wheel_speeds_FL = []
    wheel_speeds_RL = []
    wheel_speeds_FR = []
    wheel_speeds_RR = []
    for wz in w_z_range:
        wheels = swerve_inverse_kinematics(vx, v_y, wz, trackwidth, wheelbase)
        if if_optimize:
            optimized = optimize_wheel_commands(wheels, current_steering_angles)
        else:
            optimized = wheels
        steering = np.degrees(optimized[0]["steering_angle"])  # FL
        steer_angles.append(steering)
        steer_angles_FR.append(np.degrees(optimized[1]["steering_angle"]))  # FR
        steer_angles_RL.append(np.degrees(optimized[2]["steering_angle"]))  # RL
        steer_angles_RR.append(np.degrees(optimized[3]["steering_angle"]))  # RR
        wheel_speeds_FL.append(optimized[0]["linear_speed"])
        wheel_speeds_FR.append(optimized[1]["linear_speed"])
        wheel_speeds_RL.append(optimized[2]["linear_speed"])
        wheel_speeds_RR.append(optimized[3]["linear_speed"])
    wz_deg = np.degrees(w_z_range)

    # 检测突变点（以FL为例，其他轮可类似处理）
    steer_angles_arr = np.array(steer_angles)
    diff = np.abs(np.diff(steer_angles_arr))
    jump_threshold = 45
    jump_indices = np.where(diff > jump_threshold)[0] + 1

    ax1.plot(wz_deg, steer_angles, label='FL')
    ax1.plot(wz_deg, steer_angles_RL, label='RL')
    ax1.plot(wz_deg, steer_angles_FR, label='FR')
    ax1.plot(wz_deg, steer_angles_RR, label='RR')
    for idx in jump_indices:
        ax1.scatter(wz_deg[idx], steer_angles[idx], color='red', zorder=5)
        ax1.annotate(f'({wz_deg[idx]:.4f}, {steer_angles[idx]:.1f})',
                     (wz_deg[idx], steer_angles[idx]),
                     textcoords="offset points", xytext=(0,10), ha='center', color='red', fontsize=8)
    ax1.set_title(f"Effect of Angular Velocity (wz) on Steering Angle (vx={vx_fixed} m/s, vy={v_y} m/s, optimized={if_optimize})")
    ax1.set_ylabel("Steering Angle (degrees)")
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    ax1.axhline(0, color='black', linewidth=1)
    ax1.axvline(0, color='black', linewidth=1)

    ax2.plot(wz_deg, wheel_speeds_FL, label='FL linear speed')
    ax2.plot(wz_deg, wheel_speeds_RL, label='RL linear speed')
    ax2.plot(wz_deg, wheel_speeds_FR, label='FR linear speed')
    ax2.plot(wz_deg, wheel_speeds_RR, label='RR linear speed')
    ax2.set_title("Wheel Linear Speed vs. Chassis Angular Velocity")
    ax2.set_xlabel("Angular Velocity wz (deg/s)")
    ax2.set_ylabel("Wheel Linear Speed (m/s)")
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    ax2.axhline(0, color='black', linewidth=1)
    ax2.axvline(0, color='black', linewidth=1)

    plt.tight_layout()
    plt.show()

def plot_vs_vx(trackwidth, wheelbase, wz_fixed, v_y, vx_range, current_steering_angles, if_optimize=True):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    steer_angles_FL = []
    steer_angles_FR = []
    steer_angles_RL = []
    steer_angles_RR = []
    wheel_speeds_FL = []
    wheel_speeds_FR = []
    wheel_speeds_RL = []
    wheel_speeds_RR = []
    for vx in vx_range:
        wheels = swerve_inverse_kinematics(vx, v_y, wz_fixed, trackwidth, wheelbase)
        if if_optimize:
            optimized = optimize_wheel_commands(wheels, current_steering_angles)
        else:
            optimized = wheels
        steer_angles_FL.append(np.degrees(optimized[0]["steering_angle"]))
        steer_angles_FR.append(np.degrees(optimized[1]["steering_angle"]))
        steer_angles_RL.append(np.degrees(optimized[2]["steering_angle"]))
        steer_angles_RR.append(np.degrees(optimized[3]["steering_angle"]))
        wheel_speeds_FL.append(optimized[0]["linear_speed"])
        wheel_speeds_FR.append(optimized[1]["linear_speed"])
        wheel_speeds_RL.append(optimized[2]["linear_speed"])
        wheel_speeds_RR.append(optimized[3]["linear_speed"])
    ax1.plot(vx_range, steer_angles_FL, label='FL')
    ax1.plot(vx_range, steer_angles_FR, label='FR')
    ax1.plot(vx_range, steer_angles_RL, label='RL')
    ax1.plot(vx_range, steer_angles_RR, label='RR')
    # 标注 x 最小、0、最大点（转向角）
    x_indices = [0, np.argmin(np.abs(vx_range)), -1]
    for arr, label in zip([steer_angles_FL, steer_angles_FR, steer_angles_RL, steer_angles_RR], ['FL', 'FR', 'RL', 'RR']):
        for idx in x_indices:
            ax1.scatter(vx_range[idx], arr[idx], color='blue', zorder=5)
            ax1.annotate(f'{label}\n({vx_range[idx]:.1f},{arr[idx]:.1f})',
                         (vx_range[idx], arr[idx]),
                         textcoords="offset points", xytext=(0,10), ha='center', color='blue', fontsize=8)
    ax1.set_title(f"Effect of Linear Velocity (vx) on Steering Angle (vy={v_y} m/s, wz={wz_fixed} rad/s, optimized={if_optimize})")
    ax1.set_ylabel("Steering Angle (degrees)")
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    ax1.axhline(0, color='black', linewidth=1)
    ax1.axvline(0, color='black', linewidth=1)

    ax2.plot(vx_range, wheel_speeds_FL, label='FL linear speed')
    ax2.plot(vx_range, wheel_speeds_FR, label='FR linear speed')
    ax2.plot(vx_range, wheel_speeds_RL, label='RL linear speed')
    ax2.plot(vx_range, wheel_speeds_RR, label='RR linear speed')
    # 标注 x 最小、0、最大点（线速度）
    for arr, label in zip([wheel_speeds_FL, wheel_speeds_FR, wheel_speeds_RL, wheel_speeds_RR], ['FL', 'FR', 'RL', 'RR']):
        for idx in x_indices:
            ax2.scatter(vx_range[idx], arr[idx], color='blue', zorder=5)
            ax2.annotate(f'{label}\n({vx_range[idx]:.1f},{arr[idx]:.3f})',
                         (vx_range[idx], arr[idx]),
                         textcoords="offset points", xytext=(0,10), ha='center', color='blue', fontsize=8)
    ax2.set_title("Wheel Linear Speed vs. vx (wz fixed)")
    ax2.set_xlabel("Linear Velocity vx (m/s)")
    ax2.set_ylabel("Wheel Linear Speed (m/s)")
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    ax2.axhline(0, color='black', linewidth=1)
    ax2.axvline(0, color='black', linewidth=1)

    plt.tight_layout()
    plt.show()

def analyze_swerve():
    if_optimize = True  # 是否启用优化
    # 配置参数
    trackwidth = 1.7
    wheelbase = 3.26
    vx_fixed = 0.0
    v_y = 0.0
    w_z_range = np.linspace(-8, 8, 5000)
    current_steering_angles = [0.0, 0.0, 0.0, 0.0]
    plot_vs_wz(trackwidth, wheelbase, vx_fixed, v_y, w_z_range, current_steering_angles, if_optimize=if_optimize)

    # vx变化分析
    wz_fixed = -0.1
    vx_range = np.linspace(-1, 1, 5000)
    plot_vs_vx(trackwidth, wheelbase, wz_fixed, v_y, vx_range, current_steering_angles, if_optimize=if_optimize)

if __name__ == "__main__":
    analyze_swerve()
