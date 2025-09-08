#!/usr/bin/env python3
"""
UR5机械臂强化学习环境测试脚本
"""

import numpy as np
import time
from envs.ur5_assembly_env import UR5AssemblyEnv
from utils.visualization import plot_joint_trajectory, plot_assembly_progress
import cv2
import pybullet as p

def test_basic_functionality():
    """测试环境基本功能"""
    print("=== 测试环境基本功能 ===")
    
    # 创建环境，可选择是否启用相机视图
    show_camera = input("是否启用相机实时视图？(y/n): ").lower().startswith('y')
    
    env = UR5AssemblyEnv(
        render_mode="human", 
        max_steps=500,
        show_camera_view=show_camera
    )
    
    print(f"动作空间: {env.action_space}")
    print(f"观测空间: {env.observation_space}")
    
    # 重置环境
    obs, info = env.reset()
    print(f"初始观测形状: {obs['joint_state'].shape}")
    print(f"任务信息: {info}")
    
    # 测试随机动作
    print("\n=== 测试随机动作 ===")
    joint_history = []
    task_history = []
    
    for step in range(100):
        # 生成随机动作
        action = env.action_space.sample()
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 记录历史
        joint_history.append(obs['joint_state'])
        task_history.append(info)

        print(obs['joint_state'])
        
        print(f"步骤 {step}: 奖励={reward:.3f}, 完成={terminated}, 质量={info['assembly_quality']:.3f}")
        
        if terminated or truncated:
            break
            
        time.sleep(0.1)  # 减慢仿真速度 
    
    # 可视化结果
    joint_history = np.array(joint_history)
    plot_joint_trajectory(joint_history, save_path="joint_trajectory.png")
    plot_assembly_progress(task_history, save_path="assembly_progress.png")
    
    env.close()
    print("基本功能测试完成")

def test_different_action_types():
    """测试不同的动作类型"""
    print("\n=== 测试不同动作类型 ===")
    
    action_types = ["joint_position", "end_effector_pose", "joint_velocity"]
    
    for action_type in action_types:
        print(f"\n测试动作类型: {action_type}")
        
        try:
            env = UR5AssemblyEnv(render_mode="human", action_type=action_type, max_steps=200)
            obs, info = env.reset()
            
            print(f"动作空间维度: {env.action_space.shape}")
            print(f"动作空间范围: {env.action_space.low} 到 {env.action_space.high}")
            
            # 执行几个随机动作
            for step in range(50):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
                    
                time.sleep(0.1)
            
            env.close()
            print(f"{action_type} 动作类型测试完成")
            
        except Exception as e:
            print(f"{action_type} 动作类型测试失败: {e}")

def test_camera_functionality():
    """测试相机功能"""
    print("\n=== 测试相机功能 ===")
    
    # 测试RGB相机
    env_rgb = UR5AssemblyEnv(render_mode="rgb_array", max_steps=100)

    # 测试相机视图
    # 创建OpenCV窗口
    cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
    
    # 测试不同的相机位置
    camera_positions = [
        {"name": "俯视角度", "pos": [0, 0, 20], "target": [0, 0, 0]},
        {"name": "侧视角度", "pos": [1, 0, 10], "target": [0, 0, 0]},
        {"name": "前视角度", "pos": [0, -1, 1], "target": [0, 0, 0]},
        {"name": "45度俯视", "pos": [0.5, 0.5, 1.5], "target": [0, 0, 0]}
    ]
    
    for i, view in enumerate(camera_positions):
        print(f"\n=== 测试 {i+1}: {view['name']} ===")
        print(f"相机位置: {view['pos']}")
        print(f"相机目标: {view['target']}")

        env_rgb.camera.set_camera_pose(view['pos'], view['target'])
        cv2.imshow("Camera View", env_rgb.camera.get_rgb_image(env_rgb.physics_client_id))
        cv2.waitKey(1)
        print("图像已显示，按任意键继续到下一个位置...")
        input()


    obs, info = env_rgb.reset()
    
    if 'camera' in obs:
        rgb_image = obs['camera']
        print(f"RGB图像形状: {rgb_image.shape}")
        print(f"RGB图像数据类型: {rgb_image.dtype}")
        print(f"RGB图像值范围: {rgb_image.min()} - {rgb_image.max()}")
        
        # 保存图像
        env_rgb.save_camera_image("test_rgb.png", "rgb")
        print("RGB图像已保存为 test_rgb.png")
    else:
        print("RGB相机未启用")
    
    env_rgb.close()
    
    # 测试深度相机
    env_depth = UR5AssemblyEnv(render_mode="depth_array", max_steps=100)
    obs, info = env_depth.reset()
    
    if 'camera' in obs:
        depth_image = obs['camera']
        print(f"深度图像形状: {depth_image.shape}")
        print(f"深度图像数据类型: {depth_image.dtype}")
        print(f"深度图像值范围: {depth_image.min():.3f} - {depth_image.max():.3f}")
        
        # 保存图像
        env_depth.save_camera_image("test_depth.png", "depth")
        print("深度图像已保存为 test_depth.png")
    else:
        print("深度相机未启用")
    
    env_depth.close()

def test_task_parameters():
    """测试任务参数设置"""
    print("\n=== 测试任务参数设置 ===")
    
    env = UR5AssemblyEnv(render_mode="human", max_steps=300)
    
    # 获取默认任务信息
    default_task_info = env.get_task_info()
    print("默认任务参数:")
    for key, value in default_task_info.items():
        print(f"  {key}: {value}")
    
    # 修改任务参数
    env.set_task_parameters(
        peg_radius=0.015,  # 更小的轴
        hole_radius=0.02,  # 更小的孔
        position_threshold=0.005,  # 更严格的位置要求
        completion_bonus=20.0  # 更高的完成奖励
    )
    
    # 获取修改后的任务信息
    modified_task_info = env.get_task_info()
    print("\n修改后的任务参数:")
    for key, value in modified_task_info.items():
        print(f"  {key}: {value}")
    
    # 测试修改后的环境
    obs, info = env.reset()
    
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"步骤 {step}: 奖励={reward:.3f}, 质量={info['assembly_quality']:.3f}")
        
        if terminated or truncated:
            break
            
        time.sleep(0.1)
    
    env.close()

def test_environment_info():
    """测试环境信息获取"""
    print("\n=== 测试环境信息获取 ===")
    
    env = UR5AssemblyEnv(render_mode="human", max_steps=200)
    
    # 获取动作空间信息
    action_info = env.get_action_space_info()
    print("动作空间信息:")
    for key, value in action_info.items():
        print(f"  {key}: {value}")
    
    # 获取观测空间信息
    obs_info = env.get_observation_space_info()
    print("\n观测空间信息:")
    for key, value in obs_info.items():
        print(f"  {key}: {value}")
    
    # 获取任务信息
    task_info = env.get_task_info()
    print("\n任务信息:")
    for key, value in task_info.items():
        print(f"  {key}: {value}")
    
    env.close()
    

def main():
    """主函数"""
    print("开始UR5机械臂强化学习环境测试")
    
    try:
        # 基本功能测试
        test_basic_functionality()
        
        # # 不同动作类型测试
        # test_different_action_types()
        
        # 相机功能测试
        test_camera_functionality()
        
        # 任务参数测试
        test_task_parameters()
        
        # 环境信息测试
        test_environment_info()
        
        print("\n所有测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
