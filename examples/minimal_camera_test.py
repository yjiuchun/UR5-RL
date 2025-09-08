#!/usr/bin/env python3
"""
最小化相机测试脚本
"""

import numpy as np
import pybullet as p
import pybullet_data
import cv2

def test_pybullet_camera_directly():
    """直接测试PyBullet相机功能"""
    print("=== 直接测试PyBullet相机功能 ===")
    
    # 连接PyBullet
    physics_client_id = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client_id)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 加载地面
    p.loadURDF("plane.urdf", physicsClientId=physics_client_id)
    
    # 加载一个简单的物体
    box_id = p.loadURDF("r2d2.urdf", [0, 0, 1], physicsClientId=physics_client_id)
    
    # 创建OpenCV窗口
    cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
    
    # 测试不同的相机位置
    camera_positions = [
        {"name": "俯视角度", "pos": [0, 0, 2], "target": [0, 0, 0]},
        {"name": "侧视角度", "pos": [1, 0, 1], "target": [0, 0, 0]},
        {"name": "前视角度", "pos": [0, -1, 1], "target": [0, 0, 0]},
        {"name": "45度俯视", "pos": [0.5, 0.5, 1.5], "target": [0, 0, 0]}
    ]
    
    for i, view in enumerate(camera_positions):
        print(f"\n=== 测试 {i+1}: {view['name']} ===")
        print(f"相机位置: {view['pos']}")
        print(f"相机目标: {view['target']}")
        
        # 计算视图矩阵
        view_matrix = p.computeViewMatrix(view['pos'], view['target'], [0, 0, 1])
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
        
        # 转换图像格式
        rgb_image = np.array(rgb_image)
        rgb_image = rgb_image[:, :, :3]  # 只保留RGB通道
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        # 显示图像
        cv2.imshow("Camera View", bgr_image)
        cv2.waitKey(1)
        
        print("图像已显示，按任意键继续到下一个位置...")
        input()
    
    cv2.destroyAllWindows()
    p.disconnect(physics_client_id)
    print("PyBullet相机测试完成！")

def test_camera_class():
    """测试相机类"""
    print("\n=== 测试相机类 ===")
    
    from sensors.camera import Camera
    
    # 创建相机
    camera = Camera(640, 480)
    
    # 显示初始信息
    print("初始相机信息:")
    camera.debug_camera_info()
    
    # 测试不同的位置
    positions = [
        [0.0, 0.0, 2.0],
        [1.0, 0.0, 1.0],
        [0.0, -1.0, 1.0],
        [0.5, 0.5, 1.5]
    ]
    
    targets = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]
    
    for i, (pos, target) in enumerate(zip(positions, targets)):
        print(f"\n测试 {i+1}:")
        print(f"设置位置: {pos}")
        print(f"设置目标: {target}")
        
        # 设置相机位置
        camera.set_camera_pose(pos, target)
        
        # 显示设置后的信息
        print("设置后的相机信息:")
        camera.debug_camera_info()
        
        # 获取视图矩阵
        view_matrix = camera.get_view_matrix()
        print(f"视图矩阵形状: {view_matrix.shape}")
        print(f"视图矩阵前两行:\n{view_matrix[:2]}")
        
        print("按任意键继续...")
        input()

def main():
    """主函数"""
    print("开始最小化相机测试")
    
    try:
        # 直接测试PyBullet相机
        test_pybullet_camera_directly()
        
        # 测试相机类
        test_camera_class()
        
        print("\n所有测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
