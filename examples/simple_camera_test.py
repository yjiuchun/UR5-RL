#!/usr/bin/env python3
"""
简单的相机视角测试脚本
"""

import numpy as np
import time
from envs.ur5_assembly_env import UR5AssemblyEnv

def simple_camera_test():
    """简单测试相机视角变化"""
    print("=== 简单相机视角测试 ===")
    
    # 创建环境
    env = UR5AssemblyEnv(
        render_mode="human", 
        show_camera_view=True,
        max_steps=100
    )
    
    print("环境已创建，相机视图窗口应该已打开")
    print("按任意键继续...")
    input()
    
    # 重置环境
    obs, info = env.reset()
    print("环境已重置")
    
    # 显示当前相机信息
    print("\n当前相机信息:")
    env.camera.debug_camera_info()
    
    # 测试1: 俯视角度
    print("\n=== 测试1: 俯视角度 ===")
    env.camera.set_camera_pose([0.0, 0.0, 2.0], [0.0, 0.0, 0.0])
    print("设置俯视角度完成")
    env.camera.debug_camera_info()
    print("观察相机视图，按任意键继续...")
    input()
    
    # 测试2: 侧视角度
    print("\n=== 测试2: 侧视角度 ===")
    env.camera.set_camera_pose([1.0, 0.0, 0.5], [0.0, 0.0, 0.5])
    print("设置侧视角度完成")
    env.camera.debug_camera_info()
    print("观察相机视图，按任意键继续...")
    input()
    
    # 测试3: 前视角度
    print("\n=== 测试3: 前视角度 ===")
    env.camera.set_camera_pose([0.0, -1.0, 0.5], [0.0, 0.0, 0.5])
    print("设置前视角度完成")
    env.camera.debug_camera_info()
    print("观察相机视图，按任意键继续...")
    input()
    
    # 测试4: 近距离观察
    print("\n=== 测试4: 近距离观察 ===")
    env.camera.set_camera_pose([0.2, 0.0, 0.3], [0.0, 0.0, 0.3])
    print("设置近距离观察完成")
    env.camera.debug_camera_info()
    print("观察相机视图，按任意键继续...")
    input()
    
    # 测试5: 远距离观察
    print("\n=== 测试5: 远距离观察 ===")
    env.camera.set_camera_pose([0.0, 0.0, 3.0], [0.0, 0.0, 0.0])
    print("设置远距离观察完成")
    env.camera.debug_camera_info()
    print("观察相机视图，按任意键继续...")
    input()
    
    # 关闭环境
    env.close()
    print("测试完成！")

def test_camera_with_robot_movement():
    """测试相机视角变化时机器人的运动"""
    print("\n=== 测试相机视角变化时机器人的运动 ===")
    
    # 创建环境
    env = UR5AssemblyEnv(
        render_mode="human", 
        show_camera_view=True,
        max_steps=200
    )
    
    print("环境已创建")
    print("按任意键继续...")
    input()
    
    # 重置环境
    obs, info = env.reset()
    print("环境已重置")
    
    # 测试不同视角下的机器人运动
    camera_views = [
        {"name": "俯视角度", "pos": [0.0, 0.0, 2.0], "target": [0.0, 0.0, 0.0]},
        {"name": "侧视角度", "pos": [1.0, 0.0, 0.5], "target": [0.0, 0.0, 0.5]},
        {"name": "前视角度", "pos": [0.0, -1.0, 0.5], "target": [0.0, 0.0, 0.5]},
        {"name": "45度俯视", "pos": [0.5, 0.5, 1.5], "target": [0.0, 0.0, 0.0]}
    ]
    
    for view in camera_views:
        print(f"\n=== {view['name']} ===")
        env.camera.set_camera_pose(view['pos'], view['target'])
        print(f"相机位置: {view['pos']}")
        print(f"相机目标: {view['target']}")
        
        # 执行一些动作来观察机器人运动
        print("执行机器人动作...")
        for step in range(30):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
                
            time.sleep(0.1)
        
        print("按任意键切换到下一个视角...")
        input()
    
    env.close()
    print("机器人运动测试完成！")

def main():
    """主函数"""
    print("开始简单相机视角测试")
    
    try:
        # 简单相机测试
        simple_camera_test()
        
        # 机器人运动测试
        test_camera_with_robot_movement()
        
        print("\n所有测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

