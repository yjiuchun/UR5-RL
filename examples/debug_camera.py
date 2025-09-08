#!/usr/bin/env python3
"""
调试相机视角问题的脚本
"""

import numpy as np
import time
from envs.ur5_assembly_env import UR5AssemblyEnv

def debug_camera_positions():
    """调试相机位置设置"""
    print("=== 调试相机位置设置 ===")
    
    # 创建环境
    env = UR5AssemblyEnv(
        render_mode="human", 
        show_camera_view=True,
        max_steps=100
    )
    
    print("环境已创建")
    
    # 重置环境
    obs, info = env.reset()
    print("环境已重置")
    
    # 显示当前相机信息
    print(f"当前相机位置: {env.camera.camera_position}")
    print(f"当前相机目标: {env.camera.camera_target}")
    print(f"当前相机上方向: {env.camera.camera_up}")
    
    # 测试不同的相机位置
    test_positions = [
        {
            "name": "原始位置",
            "position": [0.0, 0.0, 1.0],
            "target": [0.0, 0.0, 0.0]
        },
        {
            "name": "俯视角度",
            "position": [0.0, 0.0, 2.0],
            "target": [0.0, 0.0, 0.0]
        },
        {
            "name": "侧视角度",
            "position": [1.0, 0.0, 0.5],
            "target": [0.0, 0.0, 0.5]
        },
        {
            "name": "前视角度",
            "position": [0.0, -1.0, 0.5],
            "target": [0.0, 0.0, 0.5]
        }
    ]
    
    for i, test in enumerate(test_positions):
        print(f"\n{i+1}. 测试 {test['name']}")
        print(f"   设置位置: {test['position']}")
        print(f"   设置目标: {test['target']}")
        
        # 设置相机位置
        env.camera.set_camera_pose(test['position'], test['target'])
        
        # 验证设置是否生效
        print(f"   实际位置: {env.camera.camera_position}")
        print(f"   实际目标: {env.camera.camera_target}")
        
        # 等待用户观察
        print("   观察相机视图，按任意键继续...")
        input()
        
        # 执行几个动作
        print("   执行几个动作...")
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.1)
    
    env.close()
    print("调试完成！")

def test_camera_matrix():
    """测试相机矩阵计算"""
    print("\n=== 测试相机矩阵计算 ===")
    
    from sensors.camera import Camera
    
    # 创建相机
    camera = Camera(640, 480)
    
    # 测试不同的位置
    positions = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 2.0],
        [1.0, 0.0, 0.5],
        [0.0, -1.0, 0.5]
    ]
    
    targets = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.5]
    ]
    
    for i, (pos, target) in enumerate(zip(positions, targets)):
        print(f"\n测试 {i+1}:")
        print(f"位置: {pos}")
        print(f"目标: {target}")
        
        # 设置相机位置
        camera.set_camera_pose(pos, target)
        
        # 获取视图矩阵
        view_matrix = camera.get_view_matrix()
        print(f"视图矩阵形状: {view_matrix.shape}")
        print(f"视图矩阵前几行:\n{view_matrix[:2]}")
        
        # 检查矩阵是否不同
        if i > 0:
            prev_matrix = camera.get_view_matrix()
            diff = np.abs(view_matrix - prev_matrix).max()
            print(f"与前一矩阵的最大差异: {diff}")

def test_pybullet_camera():
    """直接测试PyBullet相机功能"""
    print("\n=== 直接测试PyBullet相机功能 ===")
    
    import pybullet as p
    import pybullet_data
    
    # 连接PyBullet
    physics_client_id = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client_id)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 加载地面
    p.loadURDF("plane.urdf", physicsClientId=physics_client_id)
    
    # 加载一个简单的物体
    box_id = p.loadURDF("r2d2.urdf", [0, 0, 1], physicsClientId=physics_client_id)
    
    # 测试不同的相机位置
    camera_positions = [
        [0, 0, 2],
        [1, 0, 1],
        [0, -1, 1],
        [0.5, 0.5, 1.5]
    ]
    
    camera_targets = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    
    for i, (pos, target) in enumerate(zip(camera_positions, targets)):
        print(f"\n测试相机位置 {i+1}: {pos} -> {target}")
        
        # 计算视图矩阵
        view_matrix = p.computeViewMatrix(pos, target, [0, 0, 1])
        projection_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.01, 10.0)
        
        # 获取相机图像
        _, _, rgb_image, _, _ = p.getCameraImage(
            width=640,
            height=480,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=physics_client_id
        )
        
        print(f"图像形状: {np.array(rgb_image).shape}")
        print(f"图像数据类型: {type(rgb_image)}")
        
        # 等待用户观察
        input("按任意键继续到下一个位置...")
    
    p.disconnect(physics_client_id)
    print("PyBullet相机测试完成！")

def main():
    """主函数"""
    print("开始调试相机视角问题")
    
    try:
        # 调试相机位置设置
        debug_camera_positions()
        
        # 测试相机矩阵计算
        test_camera_matrix()
        
        # 直接测试PyBullet相机功能
        test_pybullet_camera()
        
        print("\n所有调试测试完成！")
        
    except Exception as e:
        print(f"调试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
