import numpy as np
import sys
import os
import pybullet as p

# 添加项目根目录到Python路径
sys.path.append('/home/yjc/Project/UR5-RL')
sys.path.append('/home/yjc/Project/UR5-RL/ur_ikfast')

from envs.ur5_assembly_env import UR5AssemblyEnv
import ur5e_ikfast
import time

class ur5_ik:
    def __init__(self):
        # Initialize kinematics for UR5 robot arm
        self.ur5_kin = ur5e_ikfast.PyKinematics()
        self.n_joints = self.ur5_kin.getDOF()

    def forward_kinematic(self, joint_angles):
        ee_pose = self.ur5_kin.forward(joint_angles)
        ee_pose = np.asarray(ee_pose).reshape(3, 4)  # 3x4 rigid transformation matrix
        return ee_pose

    def inverse_kinematic(self, ee_pose, cur_joint_angles):
        joint_configs = self.ur5_kin.inverse(ee_pose.reshape(-1).tolist())
        n_solutions = int(len(joint_configs) / self.n_joints)
        if n_solutions == 0:
            print('no solution found')
            return None
        joint_configs = np.asarray(joint_configs).reshape(n_solutions, self.n_joints)
        #  fix multi-solves problem
        move_ranges = []
        for joint_config in joint_configs:
            move_range = 0
            for n in range(self.n_joints):
                move_range += abs(joint_config[n]-cur_joint_angles[n])
            move_ranges.append(move_range)
        min_val = min(move_ranges)
        min_index = move_ranges.index(min_val)
        return joint_configs[min_index]

def simple_comparison():
    """简单的IKFast和PyBullet对比测试"""
    
    print("=== 简单IKFast vs PyBullet对比测试 ===\n")
    
    # 创建环境
    env = UR5AssemblyEnv(render_mode=None)
    obs, info = env.reset()
    
    # 创建IKFast实例
    ur5_ikfast = ur5_ik()
    
    # 测试你使用的关节配置
    joint_angles = np.array([0.0, -np.pi/4, np.pi/4, 0.0, np.pi/2, 0.0])
    print(f"测试关节角度: {joint_angles}")
    print(f"测试关节角度(度): {np.rad2deg(joint_angles)}")
    
    # 设置PyBullet中的关节位置
    env.ur5_model.set_joint_positions(joint_angles, env.physics_client_id)
    
    # 等待仿真稳定
    for _ in range(50):
        p.stepSimulation(physicsClientId=env.physics_client_id)
    
    # 获取PyBullet的末端执行器位姿
    pybullet_pos, pybullet_orient = env.ur5_model.get_end_effector_pose(env.physics_client_id)
    
    # 获取IKFast的末端执行器位姿
    ikfast_pose_matrix = ur5_ikfast.forward_kinematic(joint_angles)
    ikfast_pos = ikfast_pose_matrix[:3, 3]  # 位置
    
    print(f"\n--- 位置对比 ---")
    print(f"PyBullet位置: {pybullet_pos}")
    print(f"IKFast位置:   {ikfast_pos}")
    print(f"位置差异:     {np.linalg.norm(pybullet_pos - ikfast_pos):.6f} 米")
    
    print(f"\n--- 方向对比 ---")
    print(f"PyBullet方向(欧拉角): {pybullet_orient}")
    print(f"PyBullet方向(度):     {np.rad2deg(pybullet_orient)}")
    print(f"IKFast旋转矩阵:\n{ikfast_pose_matrix[:3, :3]}")
    
    # 测试逆向运动学
    print(f"\n--- 逆向运动学测试 ---")
    
    # 使用IKFast的位姿矩阵进行逆运动学
    ikfast_ik_solution = ur5_ikfast.inverse_kinematic(ikfast_pose_matrix, joint_angles)
    
    # 使用PyBullet的逆运动学
    pybullet_quat = p.getQuaternionFromEuler(pybullet_orient)
    pybullet_ik_solution = p.calculateInverseKinematics(
        env.ur5_model.robot_id,
        env.ur5_model.joint_ids[-1],
        pybullet_pos,
        pybullet_quat,
        jointDamping=[0.01] * 6,
        physicsClientId=env.physics_client_id
    )
    pybullet_ik_solution = np.array(pybullet_ik_solution[:6])  # 只取前6个关节
    
    print(f"原始关节角度: {joint_angles}")
    print(f"原始关节角度(度): {np.rad2deg(joint_angles)}")
    
    if ikfast_ik_solution is not None:
        print(f"IKFast IK解:   {ikfast_ik_solution}")
        print(f"IKFast IK解(度): {np.rad2deg(ikfast_ik_solution)}")
        print(f"IKFast IK误差: {np.linalg.norm(joint_angles - ikfast_ik_solution):.6f}")
    else:
        print("IKFast未找到解")
    
    print(f"PyBullet IK解: {pybullet_ik_solution}")
    print(f"PyBullet IK解(度): {np.rad2deg(pybullet_ik_solution)}")
    print(f"PyBullet IK误差: {np.linalg.norm(joint_angles - pybullet_ik_solution):.6f}")
    
    # 验证IK解
    print(f"\n--- 验证IK解 ---")
    if ikfast_ik_solution is not None:
        # 使用IKFast解进行正向运动学验证
        env.ur5_model.set_joint_positions(ikfast_ik_solution, env.physics_client_id)
        for _ in range(50):
            p.stepSimulation(physicsClientId=env.physics_client_id)
        
        verify_pos, verify_orient = env.ur5_model.get_end_effector_pose(env.physics_client_id)
        print(f"IKFast解验证位置: {verify_pos}")
        print(f"目标位置:         {pybullet_pos}")
        print(f"位置误差:         {np.linalg.norm(verify_pos - pybullet_pos):.6f} 米")
    
    # 使用PyBullet解进行验证
    env.ur5_model.set_joint_positions(pybullet_ik_solution, env.physics_client_id)
    for _ in range(50):
        p.stepSimulation(physicsClientId=env.physics_client_id)
    
    verify_pos, verify_orient = env.ur5_model.get_end_effector_pose(env.physics_client_id)
    print(f"PyBullet解验证位置: {verify_pos}")
    print(f"目标位置:           {pybullet_pos}")
    print(f"位置误差:           {np.linalg.norm(verify_pos - pybullet_pos):.6f} 米")
    
    env.close()

def test_multiple_configurations():
    """测试多个关节配置"""
    
    print("\n=== 多配置测试 ===\n")
    
    # 创建环境
    env = UR5AssemblyEnv(render_mode=None)
    obs, info = env.reset()
    
    # 创建IKFast实例
    ur5_ikfast = ur5_ik()
    
    # 测试多个配置
    test_configs = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, -np.pi/4, np.pi/4, 0.0, np.pi/2, 0.0]),
        np.array([np.pi/6, -np.pi/3, np.pi/6, -np.pi/4, np.pi/3, np.pi/6]),
    ]
    
    for i, joint_angles in enumerate(test_configs):
        print(f"--- 配置 {i+1} ---")
        
        # 设置关节位置
        env.ur5_model.set_joint_positions(joint_angles, env.physics_client_id)
        for _ in range(50):
            p.stepSimulation(physicsClientId=env.physics_client_id)
        
        # 获取位姿
        pybullet_pos, pybullet_orient = env.ur5_model.get_end_effector_pose(env.physics_client_id)
        ikfast_pose_matrix = ur5_ikfast.forward_kinematic(joint_angles)
        ikfast_pos = ikfast_pose_matrix[:3, 3]
        
        print(f"关节角度: {joint_angles}")
        print(f"PyBullet位置: {pybullet_pos}")
        print(f"IKFast位置:   {ikfast_pos}")
        print(f"位置差异:     {np.linalg.norm(pybullet_pos - ikfast_pos):.6f} 米")
        print()
    
    env.close()

if __name__ == "__main__":
    # 运行简单对比测试
    simple_comparison()
    
    # 运行多配置测试
    test_multiple_configurations()

