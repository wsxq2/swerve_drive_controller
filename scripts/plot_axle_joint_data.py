#!/usr/bin/env python3
"""
转向轮和驱动轮关节数据可视化脚本

功能：
- 解析 ROS2 日志文件中的 axle_joint 和 drive_joint 数据
- 绘制每个轮子的位置、速度、加速度变化曲线
- 支持多种可视化方式

使用方法：
    python3 plot_axle_joint_data.py <log_file>                 # 显示转向轮图表
    python3 plot_axle_joint_data.py b.log --drive              # 显示驱动轮图表
    python3 plot_axle_joint_data.py b.log --combined           # 显示组合图表
    python3 plot_axle_joint_data.py b.log --save output.png    # 保存图表
"""

from enum import auto
import re
import sys
import argparse
import signal
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import numpy as np

def signal_handler(sig, frame):
    """处理 Ctrl+C 信号"""
    print("\n\n检测到 Ctrl+C，正在退出...")
    plt.close('all')  # 关闭所有图表
    sys.exit(0)


class JointDataParser:
    PATTERN = re.compile(
            r'\[(\d+\.\d+)\].*Joint \'(\w+_joint)\' ([a-z ]+) command: '
            r'([-\d.]+) -> ([-\d.]+)'
        )
    def __init__(self, pattern: re.Pattern = None):
        self.data = []
        if pattern is not None:
            self.PATTERN = pattern
    
    def _parse_line(self, line: str) -> None:
        """解析单行日志"""
        pattern = self.PATTERN
        match = pattern.search(line)
        if match:
            timestamp = float(match.group(1))
            joint_name = match.group(2)
            data_type = match.group(3)
            if data_type == 'position velocity':
                data_type = 'velocity'
            old_rad = float(match.group(4))
            new_rad = float(match.group(5))
            
            print(f"解析到 {data_type} 数据 - 时间: {timestamp:.3f}s, 关节: {joint_name}, "
                  f"旧值: {old_rad:.4f} rad , 新值: {new_rad:.4f} rad")
            self.data.append({
                'data_type': data_type,
                'timestamp': timestamp,
                'joint': joint_name,
                'value_rad': new_rad,
            })
    
    def parse_file(self, filepath: str) -> Dict[str, pd.DataFrame]:
        """解析日志文件"""
        print(f"正在解析关节日志数据...")
        
        with open(filepath, 'r') as f:
            for line in f:
                self._parse_line(line)
        
        # 找到所有数据类型中的最小时间戳（全局时间基准）
        min_timestamp = min([d['timestamp'] for d in self.data])
        
        tmp_data = {}
        for data_type in ['position', 'velocity', 'acceleration']:
            tmp_data[data_type] = [d for d in self.data if d['data_type'] == data_type]

        
        # 转换为 DataFrame，使用统一的时间基准
        dfs = {}
        for data_type in ['position', 'velocity', 'acceleration']:
            if tmp_data[data_type]:
                df = pd.DataFrame(tmp_data[data_type])
                # 将时间戳转换为相对时间（秒），使用全局最小时间戳
                df['time'] = df['timestamp'] - min_timestamp
                dfs[data_type] = df
                print(f"  {data_type}: 解析了 {len(df)} 条记录", flush=True)
            else:
                print(f"  {data_type}: 未找到数据")
                dfs[data_type] = pd.DataFrame()
        
        return dfs

class JointDataPlotter:
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data
    
    def _plot_single_type_with_joint(self, ax: plt.Axes, data_type: str, joint: str, color: str, label: str):
        """绘制单一类型数据"""
        df = self.data.get(data_type)
        
        if df is None or df.empty:
            ax.text(0.5, 0.5, f'No {data_type} data', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        joint_data = df[df['joint'] == joint]
        if not joint_data.empty:
            # 只绘制一条阶跃线，使用marker参数添加点标记
            ax.step(joint_data['time'].values, joint_data['value_rad'].values, 
                   label=label, color=color, linewidth=1.5, alpha=0.8, where='post',
                   marker='o', markersize=3, markerfacecolor=color, markeredgewidth=0)
    
    def _plot_single_type_for_axle_joint(self, ax: plt.Axes, data_type: str, only_joint: str = None):
        if only_joint:
            JOINTS = [only_joint]
            colors = [self.COLORS[0]]
        else:
            JOINTS = ['front_left_axle_joint', 'front_right_axle_joint', 
                      'rear_left_axle_joint', 'rear_right_axle_joint']
            colors = self.COLORS
        
        for joint, color in zip(JOINTS, colors):
            self._plot_single_type_with_joint(ax, data_type, joint, color, joint)
        ax.set_ylabel(data_type, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim(auto=True)

    def _plot_single_type_for_drive_joint(self, ax: plt.Axes, data_type: str, only_joint: str = None):
        if only_joint:
            JOINTS = [only_joint]
            colors = [self.COLORS[0]]  # 只使用一种颜色
        else:
            JOINTS = ['front_left_wheel_joint', 'front_right_wheel_joint', 
                      'rear_left_wheel_joint', 'rear_right_wheel_joint']
            colors = self.COLORS
        
        for joint, color in zip(JOINTS, colors):
            self._plot_single_type_with_joint(ax, data_type, joint, color, joint)
        ax.set_ylabel(data_type, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim(auto=True)
    
    def plot_wheel_and_axle_joint(self, save_path: str = None, interactive: bool = True) -> Figure:
        """绘制所有数据（转向轮和驱动轮）"""
        fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
        fig.suptitle('Swerve Drive Control Data (Axle + Drive)', fontsize=16, fontweight='bold')
        
        # 用于存储图例线条和对应的绘图线条的映射
        lined = {}
        
        # 转向轮数据
        axle_types = [
            ('position', axes[0]),
            ('velocity', axes[1]),
            ('acceleration', axes[2])
        ]
        
        for data_type, ax in axle_types:
            self._plot_single_type_for_axle_joint(ax, data_type)
        
        # 驱动轮数据
        drive_types = [
            ('velocity',  axes[3]),
            ('acceleration', axes[4])
        ]
        
        for data_type, ax in drive_types:
            self._plot_single_type_for_drive_joint(ax, data_type)
        
        axes[4].set_xlabel('Time (s)', fontsize=12)
        
        # 只为第一个子图添加图例
        legend = axes[0].legend(loc='upper right', fontsize=10, framealpha=0.9)
        legend.set_draggable(True)
        
        # 收集所有子图中相同标签的线条
        for legline in legend.get_lines():
            legline.set_picker(True)
            legline.set_pickradius(5)
            # 获取图例线条对应的标签
            label = legline.get_label()
            # 提取位置前缀（如 front_left_axle_joint -> front_left）
            prefix = label.replace('_axle_joint', '').replace('_wheel_joint', '')
            # 收集所有子图中具有相同位置前缀的线条（包括 axle 和 wheel）
            all_lines = []
            for ax in axes:
                for line in ax.get_lines():
                    line_label = line.get_label()
                    line_prefix = line_label.replace('_axle_joint', '').replace('_wheel_joint', '')
                    if line_prefix == prefix:
                        all_lines.append(line)
            lined[legline] = all_lines
        
        # 点击图例切换所有相关线条的可见性
        def on_pick(event):
            legline = event.artist
            all_lines = lined.get(legline)
            if all_lines:
                visible = not all_lines[0].get_visible()
                for line in all_lines:
                    line.set_visible(visible)
                legline.set_alpha(1.0 if visible else 0.2)
                fig.canvas.draw()
        
        fig.canvas.mpl_connect('pick_event', on_pick)
        
        plt.tight_layout()
        
        if interactive:
            print("显示组合图表...")
            plt.show()
        else:
            path = 'combined_joint_plot.png'
            if save_path:
                path = save_path
            print(f"保存组合图表到: {path}")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        return fig

def main():
    # 注册 Ctrl+C 信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(
        description='解析并可视化转向轮和驱动轮关节控制数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s a.log                          # 显示转向轮图表
  %(prog)s a.log --save axle_plot.png     # 保存转向轮图表
        """
    )
    
    parser.add_argument('logfile', help='日志文件路径')
    parser.add_argument('--save', '-s', help='保存图表到文件')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.logfile).exists():
        print(f"错误: 文件不存在 - {args.logfile}")
        sys.exit(1)
    
    # 解析日志
    parser = JointDataParser()
    data = parser.parse_file(args.logfile)
    
    plotter = JointDataPlotter(data)
    plotter.plot_wheel_and_axle_joint(args.save, interactive=not args.save)
    
    print("完成！")


if __name__ == '__main__':
    main()
