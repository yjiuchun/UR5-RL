#!/usr/bin/env python3
"""
测试UR5机械臂环境的相机视角控制功能
"""

import numpy as np
import time
from envs.ur5_assembly_env import UR5AssemblyEnv

def test_camera_angles():
    """测试不同的相机视角"""
    print("=== 测试相机视角控制功能 ===")
    
    # 创建环境，启用相机视图
    env = UR5AssemblyEnv(
        render_mode="human", 
        show_camera_view=True,
        max_steps=500
    )
    
    print("环境已创建，相机视图窗口应该已打开")
    print("按任意键继续...")
    input()
    
    # 重置环境
    obs, info = env.reset()
    print(f"环境已重置，观测形状: {obs['joint_state'].shape}")
    
    # 显示当前相机信息
    camera_info = env.get_camera_info()
    print(f"当前相机信息: {camera_info}")
    
    # 测试不同的相机视角
    camera_views = [
        {
            "name": "俯视角度",
            "position": [0.0, 0.0, 2.0],
            "target": [0.0, 0.0, 0.0],
            "description": "从上方俯视机器人"
        },
        {
            "name": "侧视角度", 
            "position": [1.0, 0.0, 0.5],
            "target": [0.0, 0.0, 0.5],
            "description": "从侧面观察机器人"
        },
        {
            "name": "前视角度",
            "position": [0.0, -1.0, 0.5], 
            "target": [0.0, 0.0, 0.5],
            "description": "从前方观察机器人"
        },
        {
            "name": "45度俯视角度",
            "position": [0.5, 0.5, 1.5],
            "target": [0.0, 0.0, 0.0],
            "description": "从斜上方观察机器人"
        },
        {
            "name": "近距离观察",
            "position": [0.2, 0.0, 0.3],
            "target": [0.0, 0.0, 0.3],
            "description": "近距离观察机器人末端执行器"
        },
        {
            "name": "远距离观察",
            "position": [0.0, 0.0, 3.0],
            "target": [0.0, 0.0, 0.0],
            "description": "远距离俯视整个工作区域"
        }
    ]
    
    print("\n开始测试不同相机视角...")
    
    for i, view in enumerate(camera_views):
        print(f"\n{i+1}/{len(camera_views)}: {view['name']}")
        print(f"描述: {view['description']}")
        print(f"位置: {view['position']}")
        print(f"目标: {view['target']}")
        
        # 设置相机视角
        env.set_camera_view(view['position'], view['target'])
        
        # 等待用户观察
        print("观察相机视图，按任意键切换到下一个视角...")
        input()
        
        # 执行几个动作来观察不同视角下的机器人运动
        print("执行几个动作来观察机器人运动...")
        for step in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
                
            time.sleep(0.1)
        
        # 询问是否继续
        if i < len(camera_views) - 1:
            continue_test = input(f"\n是否继续测试下一个视角？(y/n): ").lower().startswith('y')
            if not continue_test:
                break
    
    # 显示最终相机信息
    final_camera_info = env.get_camera_info()
    print(f"\n最终相机信息: {final_camera_info}")
    
    # 关闭环境
    env.close()
    print("测试完成！")

def test_interactive_camera_control():
    """测试交互式相机控制"""
    print("\n=== 测试交互式相机控制 ===")
    
    # 创建环境
    env = UR5AssemblyEnv(
        render_mode="human",
        show_camera_view=True,
        max_steps=300
    )
    
    print("环境已创建，现在可以交互式控制相机视角")
    print("可用的命令:")
    print("  'p' - 显示当前相机信息")
    print("  '1' - 俯视角度")
    print("  '2' - 侧视角度") 
    print("  '3' - 前视角度")
    print("  '4' - 45度俯视角度")
    print("  '5' - 近距离观察")
    print("  '6' - 远距离观察")
    print("  'c' - 自定义视角")
    print("  'q' - 退出")
    
    env.reset()
    
    while True:
        command = input("\n请输入命令: ").lower()
        
        if command == 'q':
            break
        elif command == 'p':
            camera_info = env.get_camera_info()
            print(f"当前相机信息: {camera_info}")
        elif command == '1':
            env.set_camera_view([0.0, 0.0, 2.0], [0.0, 0.0, 0.0])
        elif command == '2':
            env.set_camera_view([1.0, 0.0, 0.5], [0.0, 0.0, 0.5])
        elif command == '3':
            env.set_camera_view([0.0, -1.0, 0.5], [0.0, 0.0, 0.5])
        elif command == '4':
            env.set_camera_view([0.5, 0.5, 1.5], [0.0, 0.0, 0.0])
        elif command == '5':
            env.set_camera_view([0.2, 0.0, 0.3], [0.0, 0.0, 0.3])
        elif command == '6':
            env.set_camera_view([0.0, 0.0, 3.0], [0.0, 0.0, 0.0])
        elif command == 'c':
            try:
                print("请输入相机位置 [x, y, z]:")
                pos_input = input("位置: ").strip()
                position = [float(x) for x in pos_input.strip('[]').split(',')]
                
                print("请输入目标点 [x, y, z]:")
                target_input = input("目标: ").strip()
                target = [float(x) for x in target_input.strip('[]').split(',')]
                
                env.set_camera_view(position, target)
                print("自定义视角已设置！")
            except Exception as e:
                print(f"输入格式错误: {e}")
                print("请使用格式: [x, y, z]")
        else:
            print("未知命令，请重新输入")
    
    env.close()
    print("交互式相机控制测试完成！")

def main():
    """主函数"""
    print("开始测试UR5机械臂环境的相机视角控制功能")
    
    try:
        # 测试预设相机视角
        test_camera_angles()
        
        # 测试交互式相机控制
        test_interactive_camera_control()
        
        print("\n所有测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

