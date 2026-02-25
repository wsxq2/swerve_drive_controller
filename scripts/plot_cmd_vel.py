#!/usr/bin/env python3
"""
Plot cmd_vel (vx, vy, wz) over time from ROS2 topic or rosbag.

Usage:
    # Real-time subscription mode
    python3 plot_cmd_vel.py --mode live --topic /cmd_vel

    # Rosbag mode
    python3 plot_cmd_vel.py --mode bag --bag-path /path/to/rosbag --topic /cmd_vel
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import threading
import time

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import rosbag2_py
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("Warning: ROS2 packages not available. Only basic functionality will work.")


class CmdVelRecorder(Node):
    """ROS2 node to record cmd_vel data"""
    
    def __init__(self, topic_name='/cmd_vel'):
        super().__init__('cmd_vel_recorder')
        self.topic_name = topic_name
        
        # Data storage
        self.timestamps = []
        self.vx_data = []
        self.vy_data = []
        self.wz_data = []
        self.start_time = None
        self.data_lock = threading.Lock()
        
        # Create subscription
        self.subscription = self.create_subscription(
            Twist,
            topic_name,
            self.cmd_vel_callback,
            10
        )
        self.get_logger().info(f'Subscribed to {topic_name}')
    
    def cmd_vel_callback(self, msg):
        """Callback for cmd_vel messages"""
        current_time = self.get_clock().now()
        
        if self.start_time is None:
            self.start_time = current_time
        
        # Calculate relative time in seconds
        relative_time = (current_time - self.start_time).nanoseconds / 1e9
        
        with self.data_lock:
            self.timestamps.append(relative_time)
            self.vx_data.append(msg.linear.x)
            self.vy_data.append(msg.linear.y)
            self.wz_data.append(msg.angular.z)
    
    def get_data(self):
        """Get recorded data safely"""
        with self.data_lock:
            return (
                self.timestamps.copy(),
                self.vx_data.copy(),
                self.vy_data.copy(),
                self.wz_data.copy()
            )


def read_rosbag(bag_path, topic_name='/cmd_vel'):
    """Read cmd_vel data from rosbag file"""
    print(f"Reading rosbag from: {bag_path}")
    print(f"Topic: {topic_name}")
    
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}
    
    if topic_name not in type_map:
        available_topics = ', '.join(type_map.keys())
        raise ValueError(f"Topic {topic_name} not found in bag. Available topics: {available_topics}")
    
    timestamps = []
    vx_data = []
    vy_data = []
    wz_data = []
    start_time = None
    
    msg_type = get_message(type_map[topic_name])
    
    while reader.has_next():
        (topic, data, timestamp) = reader.read_next()
        
        if topic == topic_name:
            msg = deserialize_message(data, msg_type)
            
            if start_time is None:
                start_time = timestamp
            
            relative_time = (timestamp - start_time) / 1e9  # Convert to seconds
            
            timestamps.append(relative_time)
            vx_data.append(msg.linear.x)
            vy_data.append(msg.linear.y)
            wz_data.append(msg.angular.z)
    
    print(f"Read {len(timestamps)} messages from rosbag")
    return timestamps, vx_data, vy_data, wz_data


def plot_cmd_vel_data(timestamps, vx_data, vy_data, wz_data):
    """Plot cmd_vel data in three subplots"""
    if len(timestamps) == 0:
        print("No data to plot!")
        return
    
    # Convert wz from rad/s to deg/s
    wz_deg = np.array(wz_data) * 180.0 / np.pi
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot vx
    ax1.step(timestamps, vx_data, 'b-', where='post', linewidth=1.5, marker='o', markersize=3, alpha=0.6, label='vx')
    ax1.set_ylabel('vx (m/s)', fontsize=12)
    ax1.set_title('CMD_VEL Linear Velocity X vs Time', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.axhline(0, color='black', linewidth=1)
    ax1.legend(loc='upper right')
    
    # Plot vy
    ax2.step(timestamps, vy_data, 'g-', where='post', linewidth=1.5, marker='o', markersize=3, alpha=0.6, label='vy')
    ax2.set_ylabel('vy (m/s)', fontsize=12)
    ax2.set_title('CMD_VEL Linear Velocity Y vs Time', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.legend(loc='upper right')
    
    # Plot wz (in deg/s)
    ax3.step(timestamps, wz_deg, 'r-', where='post', linewidth=1.5, marker='o', markersize=3, alpha=0.6, label='wz')
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('wz (deg/s)', fontsize=12)
    ax3.set_title('CMD_VEL Angular Velocity Z vs Time', fontsize=14, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.axhline(0, color='black', linewidth=1)
    ax3.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Print statistics
    print("\n=== Data Statistics ===")
    print(f"Duration: {timestamps[-1]:.2f} seconds")
    print(f"Total messages: {len(timestamps)}")
    print(f"\nvx - Min: {min(vx_data):.3f}, Max: {max(vx_data):.3f}, Mean: {np.mean(vx_data):.3f} m/s")
    print(f"vy - Min: {min(vy_data):.3f}, Max: {max(vy_data):.3f}, Mean: {np.mean(vy_data):.3f} m/s")
    print(f"wz - Min: {min(wz_deg):.3f}, Max: {max(wz_deg):.3f}, Mean: {np.mean(wz_deg):.3f} deg/s")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot cmd_vel data from ROS2 topic or rosbag')
    parser.add_argument('--mode', type=str, choices=['live', 'bag'], required=True,
                        help='Mode: live (subscribe to topic) or bag (read from rosbag)')
    parser.add_argument('--topic', type=str, default='/cmd_vel',
                        help='Topic name (default: /cmd_vel)')
    parser.add_argument('--bag-path', type=str,
                        help='Path to rosbag file (required for bag mode)')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='Recording duration in seconds for live mode (default: 30)')
    
    args = parser.parse_args()
    
    if not ROS2_AVAILABLE:
        print("Error: ROS2 packages are not available. Please source your ROS2 environment.")
        return
    
    if args.mode == 'bag':
        # Rosbag mode
        if not args.bag_path:
            print("Error: --bag-path is required for bag mode")
            return
        
        timestamps, vx_data, vy_data, wz_data = read_rosbag(args.bag_path, args.topic)
        plot_cmd_vel_data(timestamps, vx_data, vy_data, wz_data)
    
    elif args.mode == 'live':
        # Live subscription mode
        rclpy.init()
        
        recorder = CmdVelRecorder(topic_name=args.topic)
        
        print(f"Recording cmd_vel data for {args.duration} seconds...")
        print("Press Ctrl+C to stop early and plot the data")
        
        # Spin in a separate thread
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(recorder)
        
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()
        
        try:
            # Wait for specified duration
            time.sleep(args.duration)
        except KeyboardInterrupt:
            print("\nStopped by user")
        
        # Get data and plot
        timestamps, vx_data, vy_data, wz_data = recorder.get_data()
        
        # Cleanup - proper order is important
        executor.shutdown()
        recorder.destroy_node()
        
        try:
            rclpy.shutdown()
        except:
            pass  # Ignore if already shutdown
        
        if len(timestamps) > 0:
            plot_cmd_vel_data(timestamps, vx_data, vy_data, wz_data)
        else:
            print("No data recorded. Make sure the topic is publishing.")


if __name__ == '__main__':
    main()
