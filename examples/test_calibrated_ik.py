import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append('/home/yjc/Project/UR5-RL')
sys.path.append('/home/yjc/Project/UR5-RL/ur_ikfast')

from envs.ur5_assembly_env import UR5AssemblyEnv
from models.calibrated_ur5_ik import CalibratedUR5IK
import time

def test_calibrated_ik():
    """测试校准后的IK类"""
    
    print("=== 测试校准后的UR5 IK类 ===\n")
    
    # 创建校准后的IK实例
    calibrated_ik = CalibratedUR5IK()
    
    # 创建环境
    env = UR5AssemblyEnv(render_mode=None)
    obs, info = env.reset()
    
    # 测试多个关节配置
    test_configurations = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 零位
        np.array([0.0, -np.pi/4, np.pi/4, 0.0, np.pi/2, 0.0]),  # 你使用的配置
        np.array([np.pi/6, -np.pi/3, np.pi/6, -np.pi/4, np.pi/3, np.pi/6]),  # 随机配置1
        np.array([-np.pi/4, np.pi/6, -np.pi/3, np.pi/4, -np.pi/6, np.pi/3]),  # 随机配置2
    ]
    
    print("校准信息:")
    calib_info = calibrated_ik.get_calibration_info()
    for key, value in calib_info.items():
        print(f"  {key}: {value}")
    print()
    
    for i, joint_angles in enumerate(test_configurations):
        print(f"--- 测试配置 {i+1} ---")
        print(f"关节角度: {joint_angles}")
        print(f"关节角度(度): {np.rad2deg(joint_angles)}")
        
        # 设置PyBullet中的关节位置
        env.ur5_model.set_joint_positions(joint_angles, env.physics_client_id)
        
        # 等待仿真稳定
        for _ in range(50):
            import pybullet as p
            p.stepSimulation(physicsClientId=env.physics_client_id)
        
        # 获取PyBullet的末端执行器位姿
        pybullet_pos, pybullet_orient = env.ur5_model.get_end_effector_pose(env.physics_client_id)
        
        # 获取校准后IKFast的末端执行器位姿
        calibrated_ikfast_pose = calibrated_ik.forward_kinematic(joint_angles)
        calibrated_ikfast_pos = calibrated_ikfast_pose[:3, 3]
        
        print(f"PyBullet位置: {pybullet_pos}")
        print(f"校准IKFast位置: {calibrated_ikfast_pos}")
        print(f"位置差异: {np.linalg.norm(pybullet_pos - calibrated_ikfast_pos):.6f} 米")
        
        # 测试逆向运动学
        print("\n--- 逆向运动学测试 ---")
        
        # 使用校准后的IKFast进行逆运动学
        calibrated_ik_solution = calibrated_ik.inverse_kinematic(calibrated_ikfast_pose, joint_angles)
        
        if calibrated_ik_solution is not None:
            print(f"校准IKFast IK解: {calibrated_ik_solution}")
            print(f"校准IKFast IK解(度): {np.rad2deg(calibrated_ik_solution)}")
            print(f"校准IKFast IK误差: {np.linalg.norm(joint_angles - calibrated_ik_solution):.6f}")
            
            # 验证校准后的IK解
            env.ur5_model.set_joint_positions(calibrated_ik_solution, env.physics_client_id)
            for _ in range(50):
                import pybullet as p
                p.stepSimulation(physicsClientId=env.physics_client_id)
            
            verify_pos, verify_orient = env.ur5_model.get_end_effector_pose(env.physics_client_id)
            print(f"校准IK解验证位置: {verify_pos}")
            print(f"目标位置: {pybullet_pos}")
            print(f"位置误差: {np.linalg.norm(verify_pos - pybullet_pos):.6f} 米")
        else:
            print("校准IKFast未找到解")
        
        print("\n" + "="*60 + "\n")
    
    env.close()

def test_ik_functions():
    """测试IK类的各种功能"""
    
    print("=== 测试IK类功能 ===\n")
    
    calibrated_ik = CalibratedUR5IK()
    
    # 测试关节配置
    joint_angles = np.array([0.0, -np.pi/4, np.pi/4, 0.0, np.pi/2, 0.0])
    
    print(f"测试关节角度: {joint_angles}")
    
    # 测试正向运动学
    ee_pose = calibrated_ik.forward_kinematic(joint_angles)
    print(f"末端执行器位姿矩阵:\n{ee_pose}")
    
    # 测试获取位置和方向
    position = calibrated_ik.get_end_effector_position(joint_angles)
    orientation = calibrated_ik.get_end_effector_orientation(joint_angles)
    print(f"末端执行器位置: {position}")
    print(f"末端执行器方向矩阵:\n{orientation}")
    
    # 测试逆向运动学
    ik_solution = calibrated_ik.inverse_kinematic(ee_pose, joint_angles)
    print(f"IK解: {ik_solution}")
    
    # 测试所有IK解
    all_solutions = calibrated_ik.get_all_ik_solutions(ee_pose)
    print(f"所有IK解数量: {len(all_solutions)}")
    if len(all_solutions) > 0:
        print(f"所有IK解:\n{all_solutions}")
    
    # 测试可达性
    is_reachable = calibrated_ik.is_reachable(ee_pose)
    print(f"位姿是否可达: {is_reachable}")
    
    # 测试关节限制
    lower_limits, upper_limits = calibrated_ik.get_joint_limits()
    print(f"关节下限: {lower_limits}")
    print(f"关节上限: {upper_limits}")
    
    # 测试关节角度有效性
    is_valid = calibrated_ik.is_joint_angles_valid(joint_angles)
    print(f"关节角度是否有效: {is_valid}")
    
    # 测试无效关节角度
    invalid_joints = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    is_invalid = calibrated_ik.is_joint_angles_valid(invalid_joints)
    print(f"无效关节角度是否有效: {is_invalid}")

if __name__ == "__main__":
    # 测试校准后的IK
    test_calibrated_ik()
    
    # 测试IK类功能
    test_ik_functions()

