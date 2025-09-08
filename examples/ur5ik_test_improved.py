import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append('/home/yjc/Project/UR5-RL')
sys.path.append('/home/yjc/Project/UR5-RL/ur_ikfast')

from envs.ur5_assembly_env import UR5AssemblyEnv
from models.calibrated_ur5_ik import CalibratedUR5IK
import time

if __name__ == "__main__":
    print("开始测试UR5环境...")
    
    # 测试GUI模式
    print("创建环境（GUI模式）...")
    env = UR5AssemblyEnv(render_mode='human')
    
    print("重置环境...")
    obs, info = env.reset()
    print(f"初始观测空间: {list(obs.keys())}")
    print(f"关节状态维度: {obs['joint_state'].shape}")
    print(f"任务状态维度: {obs['task_state'].shape}")
    
    print("执行动作...")
    action_sample = np.array([0.0, -np.pi/4, np.pi/4, 0.0, np.pi/2, 0.0])
    obs, reward, terminated, truncated, info = env.step(action_sample)
    
    # 获取PyBullet中的末端执行器位姿
    pos, angle = env.ur5_model.get_end_effector_pose(env.physics_client_id)
    print(f"PyBullet末端执行器位置: {pos}")
    print(f"PyBullet末端执行器角度: {angle}")
    
    time.sleep(5)
    
    print("关闭环境...")
    env.close()
    
    print("测试完成！")
    
    # 测试校准后的IK功能
    print("\n=== 测试校准后的IK功能 ===")
    try:
        # 创建校准后的IK实例
        calibrated_ik = CalibratedUR5IK()
        
        # 显示校准信息
        calib_info = calibrated_ik.get_calibration_info()
        print("校准信息:")
        for key, value in calib_info.items():
            print(f"  {key}: {value}")
        print()
        
        # 测试正向运动学
        test_joints = np.array([0.0, -np.pi/4, np.pi/4, 0.0, np.pi/2, 0.0])
        print(f"测试关节角度: {test_joints}")
        print(f"测试关节角度(度): {np.rad2deg(test_joints)}")
        
        # 获取校准后的末端执行器位姿
        ee_pose = calibrated_ik.forward_kinematic(test_joints)
        ee_position = calibrated_ik.get_end_effector_position(test_joints)
        ee_orientation = calibrated_ik.get_end_effector_orientation(test_joints)
        
        print(f"校准IKFast末端执行器位姿矩阵:\n{ee_pose}")
        print(f"校准IKFast末端执行器位置: {ee_position}")
        print(f"校准IKFast末端执行器方向矩阵:\n{ee_orientation}")
        
        # 测试逆向运动学
        print("\n--- 逆向运动学测试 ---")
        ik_solution = calibrated_ik.inverse_kinematic(ee_pose, test_joints)
        if ik_solution is not None:
            print(f"校准IKFast IK解: {ik_solution}")
            print(f"校准IKFast IK解(度): {np.rad2deg(ik_solution)}")
            print(f"IK误差: {np.linalg.norm(test_joints - ik_solution):.8f}")
        else:
            print("未找到IK解")
        
        # 测试所有IK解
        all_solutions = calibrated_ik.get_all_ik_solutions(ee_pose)
        print(f"所有IK解数量: {len(all_solutions)}")
        
        # 测试可达性
        is_reachable = calibrated_ik.is_reachable(ee_pose)
        print(f"位姿是否可达: {is_reachable}")
        
        # 测试关节限制
        lower_limits, upper_limits = calibrated_ik.get_joint_limits()
        print(f"关节限制范围: [{np.rad2deg(lower_limits)}, {np.rad2deg(upper_limits)}]")
        
        # 测试关节角度有效性
        is_valid = calibrated_ik.is_joint_angles_valid(test_joints)
        print(f"关节角度是否有效: {is_valid}")
        
        print("\n=== 对比测试 ===")
        # 重新创建环境进行对比测试
        env = UR5AssemblyEnv(render_mode=None)
        obs, info = env.reset()
        
        # 设置相同的关节角度
        env.ur5_model.set_joint_positions(test_joints, env.physics_client_id)
        
        # 等待仿真稳定
        for _ in range(50):
            import pybullet as p
            p.stepSimulation(physicsClientId=env.physics_client_id)
        
        # 获取PyBullet的末端执行器位姿
        pybullet_pos, pybullet_orient = env.ur5_model.get_end_effector_pose(env.physics_client_id)
        
        print(f"PyBullet位置: {pybullet_pos}")
        print(f"校准IKFast位置: {ee_position}")
        print(f"位置差异: {np.linalg.norm(pybullet_pos - ee_position):.6f} 米")
        
        # 验证IK解
        if ik_solution is not None:
            env.ur5_model.set_joint_positions(ik_solution, env.physics_client_id)
            for _ in range(50):
                import pybullet as p
                p.stepSimulation(physicsClientId=env.physics_client_id)
            
            verify_pos, verify_orient = env.ur5_model.get_end_effector_pose(env.physics_client_id)
            print(f"IK解验证位置: {verify_pos}")
            print(f"目标位置: {pybullet_pos}")
            print(f"验证位置误差: {np.linalg.norm(verify_pos - pybullet_pos):.6f} 米")
        
        env.close()
        
    except Exception as e:
        print(f"IK测试失败: {e}")
        import traceback
        traceback.print_exc()
