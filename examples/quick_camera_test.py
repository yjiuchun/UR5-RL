#!/usr/bin/env python3
"""
快速相机视角测试
"""

import numpy as np
import time
from envs.ur5_assembly_env import UR5AssemblyEnv

def quick_test():
    """快速测试相机视角"""
    print("=== 快速相机视角测试 ===")
    
    # 创建环境
    env = UR5AssemblyEnv(
        render_mode="human", 
        show_camera_view=True,
        max_steps=50
    )
    
    print("环境已创建")
    
    # 重置环境
    obs, info = env.reset()
    print("环境已重置")
    
    # 显示当前相机信息
    print("当前相机信息:")
    env.camera.debug_camera_info()
    
    # 测试几个不同的视角
    views = [
        {"name": "俯视", "pos": [0.0, 0.0, 2.0], "target": [0.0, 0.0, 0.0]},
        {"name": "侧视", "pos": [1.0, 0.0, 0.5], "target": [0.0, 0.0, 0.5]},
        {"name": "前视", "pos": [0.0, -1.0, 0.5], "target": [0.0, 0.0, 0.5]}
    ]
    
    for view in views:
        print(f"\n设置 {view['name']} 视角...")
        env.camera.set_camera_pose(view['pos'], view['target'])
        print(f"位置: {view['pos']}, 目标: {view['target']}")
        
        # 执行几个动作
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.1)
        
        print("按任意键继续...")
        input()
    
    env.close()
    print("测试完成！")

if __name__ == "__main__":
    quick_test()

