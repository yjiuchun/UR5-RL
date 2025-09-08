#!/usr/bin/env python3
"""
测试UR5机械臂环境的相机实时视图功能
"""

import numpy as np
import time
from envs.ur5_assembly_env import UR5AssemblyEnv

def test_camera_view():
    """测试相机实时视图功能"""
    print("=== 测试相机实时视图功能 ===")
    
    # 创建环境，启用相机视图
    env = UR5AssemblyEnv(
        render_mode="human", 
        show_camera_view=True,  # 启用相机实时视图
        max_steps=200
    )
    
    print("环境已创建，相机视图窗口应该已打开")
    print("按任意键继续...")
    input()
    env.save_camera_image("camera_image.png","rgb")
    
    # 重置环境
    obs, info = env.reset()
    print(f"环境已重置，观测形状: {obs['joint_state'].shape}")
    
    # 执行一些随机动作来观察相机视图
    print("开始执行随机动作，观察相机视图...")
    print("按 'q' 键退出，或等待动作执行完成")
    
    for step in range(100):
        # 生成随机动作
        action = env.action_space.sample()
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"步骤 {step}: 奖励={reward:.3f}, 质量={info['assembly_quality']:.3f}")
        
        if terminated or truncated:
            break
            
        # 减慢仿真速度，让用户观察相机视图
        time.sleep(0.2)
        
        # 检查用户输入
        try:
            import cv2
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("用户按了 'q' 键，退出测试")
                break
        except:
            pass
    
    # 关闭环境
    env.close()
    print("测试完成！")

def test_camera_view_without_gui():
    """测试无GUI模式下的相机视图"""
    print("\n=== 测试无GUI模式下的相机视图 ===")
    
    # 创建环境，无GUI但启用相机视图
    env = UR5AssemblyEnv(
        render_mode=None,  # 无GUI模式
        show_camera_view=True,  # 启用相机实时视图
        max_steps=100
    )
    
    print("环境已创建（无GUI模式），相机视图窗口应该已打开")
    print("按任意键继续...")
    input()
    
    # 重置环境
    obs, info = env.reset()
    print(f"环境已重置，观测形状: {obs['joint_state'].shape}")
    
    # 执行一些随机动作
    print("开始执行随机动作，观察相机视图...")
    
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"步骤 {step}: 奖励={reward:.3f}, 质量={info['assembly_quality']:.3f}")
        
        if terminated or truncated:
            break
            
        time.sleep(0.3)
    
    # 关闭环境
    env.close()
    print("测试完成！")

def main():
    """主函数"""
    print("开始测试UR5机械臂环境的相机实时视图功能")
    
    try:
        # 测试带GUI的相机视图
        test_camera_view()
        
        # 测试无GUI的相机视图
        test_camera_view_without_gui()
        
        print("\n所有测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
