import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Tuple
import cv2

def plot_joint_trajectory(joint_history: np.ndarray, joint_names: List[str] = None, 
                         save_path: Optional[str] = None):
    """
    绘制关节轨迹
    
    Args:
        joint_history: 关节历史数据 (time_steps, num_joints)
        joint_names: 关节名称列表
        save_path: 保存路径
    """
    if joint_names is None:
        joint_names = [f"Joint {i+1}" for i in range(joint_history.shape[1])]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(joint_history.shape[1]):
        ax = axes[i]
        ax.plot(joint_history[:, i], linewidth=2)
        ax.set_title(joint_names[i])
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Joint Angle (rad)')
        ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_end_effector_trajectory(positions: np.ndarray, orientations: np.ndarray,
                                save_path: Optional[str] = None):
    """
    绘制末端执行器轨迹
    
    Args:
        positions: 位置历史 (time_steps, 3)
        orientations: 方向历史 (time_steps, 3)
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 位置轨迹
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                c='g', s=100, label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                c='r', s=100, label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('End Effector Position Trajectory')
    ax1.legend()
    ax1.grid(True)
    
    # 方向轨迹
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(orientations[:, 0], label='Roll', linewidth=2)
    ax2.plot(orientations[:, 1], label='Pitch', linewidth=2)
    ax2.plot(orientations[:, 2], label='Yaw', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Orientation (rad)')
    ax2.set_title('End Effector Orientation Trajectory')
    ax2.legend()
    ax2.grid(True)
    
    # 位置分量
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(positions[:, 0], label='X', linewidth=2)
    ax3.plot(positions[:, 1], label='Y', linewidth=2)
    ax3.plot(positions[:, 2], label='Z', linewidth=2)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Position (m)')
    ax3.set_title('Position Components')
    ax3.legend()
    ax3.grid(True)
    
    # 速度
    velocities = np.diff(positions, axis=0)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(np.linalg.norm(velocities, axis=1), 'r-', linewidth=2)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Velocity (m/step)')
    ax4.set_title('End Effector Velocity')
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_assembly_progress(task_history: List[Dict], save_path: Optional[str] = None):
    """
    绘制装配进度
    
    Args:
        task_history: 任务历史数据列表
        save_path: 保存路径
    """
    if not task_history:
        return
    
    # 提取数据
    steps = [state.get('step', i) for i, state in enumerate(task_history)]
    assembly_quality = [state.get('assembly_quality', 0.0) for state in task_history]
    insertion_depth = [state.get('insertion_depth', 0.0) for state in task_history]
    assembly_completed = [state.get('assembly_completed', False) for state in task_history]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 装配质量
    ax1 = axes[0, 0]
    ax1.plot(steps, assembly_quality, 'b-', linewidth=2)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Assembly Quality')
    ax1.set_title('Assembly Quality Progress')
    ax1.grid(True)
    ax1.set_ylim(0, 1)
    
    # 插入深度
    ax2 = axes[0, 1]
    ax2.plot(steps, insertion_depth, 'g-', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Insertion Depth (m)')
    ax2.set_title('Insertion Depth Progress')
    ax2.grid(True)
    
    # 完成状态
    ax3 = axes[1, 0]
    ax3.plot(steps, assembly_completed, 'r-', linewidth=2)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Assembly Completed')
    ax3.set_title('Assembly Completion Status')
    ax3.grid(True)
    ax3.set_ylim(-0.1, 1.1)
    
    # 奖励（如果有）
    if 'reward' in task_history[0]:
        rewards = [state.get('reward', 0.0) for state in task_history]
        ax4 = axes[1, 1]
        ax4.plot(steps, rewards, 'm-', linewidth=2)
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Reward')
        ax4.set_title('Reward Progress')
        ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_robot_workspace(ur5_model, save_path: Optional[str] = None):
    """
    绘制机器人工作空间
    
    Args:
        ur5_model: UR5模型实例
        save_path: 保存路径
    """
    # 生成工作空间采样点
    num_samples = 1000
    joint_samples = np.random.uniform(
        ur5_model.joint_lower_limits,
        ur5_model.joint_upper_limits,
        (num_samples, ur5_model.num_joints)
    )
    
    # 计算末端执行器位置
    end_effector_positions = []
    for joint_config in joint_samples:
        # 这里需要实现前向运动学来计算末端执行器位置
        # 简化版本，使用随机位置
        position = np.random.uniform(-1, 1, 3)
        end_effector_positions.append(position)
    
    end_effector_positions = np.array(end_effector_positions)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制工作空间点
    ax.scatter(end_effector_positions[:, 0], 
               end_effector_positions[:, 1], 
               end_effector_positions[:, 2], 
               c='b', alpha=0.6, s=1)
    
    # 绘制机器人基座
    ax.scatter([0], [0], [0], c='r', s=100, label='Robot Base')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('UR5 Robot Workspace')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def visualize_assembly_task(peg_position: np.ndarray, hole_position: np.ndarray,
                           peg_radius: float, hole_radius: float,
                           save_path: Optional[str] = None):
    """
    可视化装配任务
    
    Args:
        peg_position: 轴位置
        hole_position: 孔位置
        peg_radius: 轴半径
        hole_radius: 孔半径
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(12, 8))
    
    # 2D俯视图
    ax1 = fig.add_subplot(1, 2, 1)
    
    # 绘制孔
    hole_circle = Circle((hole_position[0], hole_position[1]), hole_radius, 
                        fill=False, color='blue', linewidth=2, label='Hole')
    ax1.add_patch(hole_circle)
    
    # 绘制轴
    peg_circle = Circle((peg_position[0], peg_position[1]), peg_radius, 
                       fill=True, color='red', alpha=0.7, label='Peg')
    ax1.add_patch(peg_circle)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Assembly Task (Top View)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_aspect('equal')
    
    # 3D视图
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # 绘制孔（圆柱体）
    theta = np.linspace(0, 2*np.pi, 100)
    z_hole = np.linspace(hole_position[2], hole_position[2] + 0.1, 2)
    theta_grid, z_grid = np.meshgrid(theta, z_hole)
    x_hole = hole_position[0] + hole_radius * np.cos(theta_grid)
    y_hole = hole_position[1] + hole_radius * np.sin(theta_grid)
    
    ax2.plot_surface(x_hole, y_hole, z_grid, alpha=0.3, color='blue')
    
    # 绘制轴（圆柱体）
    z_peg = np.linspace(peg_position[2], peg_position[2] + 0.1, 2)
    theta_grid, z_grid = np.meshgrid(theta, z_peg)
    x_peg = peg_position[0] + peg_radius * np.cos(theta_grid)
    y_peg = peg_position[1] + peg_radius * np.sin(theta_grid)
    
    ax2.plot_surface(x_peg, y_peg, z_grid, alpha=0.7, color='red')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('Assembly Task (3D View)')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_camera_image(image: np.ndarray, image_type: str = "rgb", 
                     save_path: Optional[str] = None):
    """
    绘制相机图像
    
    Args:
        image: 图像数据
        image_type: 图像类型 ("rgb", "depth", "segmentation")
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    
    if image_type == "rgb":
        plt.imshow(image)
        plt.title('RGB Camera Image')
    elif image_type == "depth":
        plt.imshow(image, cmap='viridis')
        plt.colorbar(label='Depth (m)')
        plt.title('Depth Camera Image')
    elif image_type == "segmentation":
        plt.imshow(image, cmap='tab20')
        plt.colorbar(label='Object ID')
        plt.title('Segmentation Image')
    else:
        plt.imshow(image)
        plt.title(f'{image_type.capitalize()} Image')
    
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def create_animation(joint_history: np.ndarray, save_path: str, 
                    fps: int = 30, joint_names: List[str] = None):
    """
    创建关节轨迹动画
    
    Args:
        joint_history: 关节历史数据
        save_path: 保存路径
        fps: 帧率
        joint_names: 关节名称
    """
    if joint_names is None:
        joint_names = [f"Joint {i+1}" for i in range(joint_history.shape[1])]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    lines = []
    for i in range(joint_history.shape[1]):
        ax = axes[i]
        line, = ax.plot([], [], 'b-', linewidth=2)
        lines.append(line)
        ax.set_title(joint_names[i])
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Joint Angle (rad)')
        ax.set_xlim(0, len(joint_history))
        ax.set_ylim(joint_history[:, i].min(), joint_history[:, i].max())
        ax.grid(True)
    
    plt.tight_layout()
    
    def animate(frame):
        for i, line in enumerate(lines):
            line.set_data(range(frame + 1), joint_history[:frame + 1, i])
        return lines
    
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, animate, frames=len(joint_history), 
                        interval=1000//fps, blit=True)
    
    # 保存动画
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()

def plot_reward_distribution(rewards: List[float], save_path: Optional[str] = None):
    """
    绘制奖励分布
    
    Args:
        rewards: 奖励列表
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 8))
    
    # 奖励直方图
    plt.subplot(2, 2, 1)
    plt.hist(rewards, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.grid(True, alpha=0.3)
    
    # 奖励累积分布
    plt.subplot(2, 2, 2)
    sorted_rewards = np.sort(rewards)
    cumulative_prob = np.arange(1, len(sorted_rewards) + 1) / len(sorted_rewards)
    plt.plot(sorted_rewards, cumulative_prob, 'b-', linewidth=2)
    plt.xlabel('Reward')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Reward Distribution')
    plt.grid(True, alpha=0.3)
    
    # 奖励时间序列
    plt.subplot(2, 2, 3)
    plt.plot(rewards, 'g-', linewidth=1, alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Over Episodes')
    plt.grid(True, alpha=0.3)
    
    # 奖励统计
    plt.subplot(2, 2, 4)
    stats_text = f"""
    Mean: {np.mean(rewards):.3f}
    Std: {np.std(rewards):.3f}
    Min: {np.min(rewards):.3f}
    Max: {np.max(rewards):.3f}
    Median: {np.median(rewards):.3f}
    """
    plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    plt.axis('off')
    plt.title('Reward Statistics')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
