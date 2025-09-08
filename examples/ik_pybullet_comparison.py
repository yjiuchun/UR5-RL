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

def compare_kinematics():
    """对比IKFast和PyBullet的运动学结果"""
    
    print("=== IKFast vs PyBullet 运动学对比测试 ===\n")
    
    # 创建环境
    env = UR5AssemblyEnv(render_mode=None)
    obs, info = env.reset()
    
    # 创建IKFast实例
    ur5_ikfast = ur5_ik()
    
    # 测试多个关节配置
    test_configurations = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 零位
        np.array([0.0, -np.pi/4, np.pi/4, 0.0, np.pi/2, 0.0]),  # 你使用的配置
        np.array([np.pi/6, -np.pi/3, np.pi/6, -np.pi/4, np.pi/3, np.pi/6]),  # 随机配置1
        np.array([-np.pi/4, np.pi/6, -np.pi/3, np.pi/4, -np.pi/6, np.pi/3]),  # 随机配置2
    ]
    
    for i, joint_angles in enumerate(test_configurations):
        print(f"--- 测试配置 {i+1}: {joint_angles} ---")
        
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
        ikfast_orient_matrix = ikfast_pose_matrix[:3, :3]  # 旋转矩阵
        
        # 将旋转矩阵转换为欧拉角
        # 从3x4矩阵中提取3x3旋转矩阵
        rotation_matrix = ikfast_pose_matrix[:3, :3]
        # 使用scipy或其他方法将旋转矩阵转换为四元数，然后转换为欧拉角
        # 这里我们直接使用旋转矩阵的欧拉角表示
        ikfast_orient = np.array([0.0, 0.0, 0.0])  # 临时占位符
        
        print(f"PyBullet位置: {pybullet_pos}")
        print(f"IKFast位置:   {ikfast_pos}")
        print(f"位置差异:     {np.linalg.norm(pybullet_pos - ikfast_pos):.6f}")
        
        print(f"PyBullet角度: {pybullet_orient}")
        print(f"IKFast角度:   {ikfast_orient}")
        print(f"角度差异:     {np.linalg.norm(pybullet_orient - ikfast_orient):.6f}")
        
        # 测试逆向运动学
        print("\n--- 逆向运动学测试 ---")
        
        # 使用PyBullet的位置进行IKFast逆运动学
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
        
        if ikfast_ik_solution is not None:
            print(f"原始关节角度: {joint_angles}")
            print(f"IKFast IK解:   {ikfast_ik_solution}")
            print(f"PyBullet IK解: {pybullet_ik_solution}")
            print(f"IKFast IK误差: {np.linalg.norm(joint_angles - ikfast_ik_solution):.6f}")
            print(f"PyBullet IK误差: {np.linalg.norm(joint_angles - pybullet_ik_solution):.6f}")
        else:
            print("IKFast未找到解")
            print(f"PyBullet IK解: {pybullet_ik_solution}")
            print(f"PyBullet IK误差: {np.linalg.norm(joint_angles - pybullet_ik_solution):.6f}")
        
        print("\n" + "="*60 + "\n")
    
    env.close()

def test_coordinate_systems():
    """测试坐标系差异"""
    
    print("=== 坐标系差异测试 ===\n")
    
    # 创建环境
    env = UR5AssemblyEnv(render_mode=None)
    obs, info = env.reset()
    
    # 创建IKFast实例
    ur5_ikfast = ur5_ik()
    
    # 测试零位配置
    joint_angles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    env.ur5_model.set_joint_positions(joint_angles, env.physics_client_id)
    
    # 等待仿真稳定
    for _ in range(50):
        p.stepSimulation(physicsClientId=env.physics_client_id)
    
    # 获取PyBullet的基座和末端执行器位姿
    base_pos, base_quat = p.getBasePositionAndOrientation(env.ur5_model.robot_id, env.physics_client_id)
    base_orient = p.getEulerFromQuaternion(base_quat)
    
    ee_pos, ee_orient = env.ur5_model.get_end_effector_pose(env.physics_client_id)
    
    print(f"PyBullet基座位置: {base_pos}")
    print(f"PyBullet基座方向: {base_orient}")
    print(f"PyBullet末端位置: {ee_pos}")
    print(f"PyBullet末端方向: {ee_orient}")
    
    # 获取IKFast的位姿矩阵
    ikfast_pose_matrix = ur5_ikfast.forward_kinematic(joint_angles)
    print(f"IKFast位姿矩阵:\n{ikfast_pose_matrix}")
    
    # 检查URDF文件信息
    print(f"\nURDF文件路径: {env.ur5_model.urdf_path}")
    
    # 获取关节信息
    print("\n关节信息:")
    for i, joint_id in enumerate(env.ur5_model.joint_ids):
        joint_info = p.getJointInfo(env.ur5_model.robot_id, joint_id, env.physics_client_id)
        print(f"关节 {i}: {joint_info[1].decode('utf-8')}")
        print(f"  位置: {joint_info[12]}")
        print(f"  方向: {joint_info[13]}")
    
    env.close()

if __name__ == "__main__":
    # 运行对比测试
    compare_kinematics()
    
    # 运行坐标系测试
    test_coordinate_systems()
