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

def analyze_coordinate_difference():
    """分析坐标系差异"""
    
    print("=== 坐标系差异分析 ===\n")
    
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
    
    # 获取IKFast的位姿矩阵
    ikfast_pose_matrix = ur5_ikfast.forward_kinematic(joint_angles)
    ikfast_pos = ikfast_pose_matrix[:3, 3]
    
    print(f"PyBullet基座位置: {base_pos}")
    print(f"PyBullet基座方向: {base_orient}")
    print(f"PyBullet末端位置: {ee_pos}")
    print(f"PyBullet末端方向: {ee_orient}")
    print(f"IKFast末端位置:   {ikfast_pos}")
    print(f"IKFast位姿矩阵:\n{ikfast_pose_matrix}")
    
    # 计算位置偏移
    position_offset = ee_pos - ikfast_pos
    print(f"\n位置偏移: {position_offset}")
    print(f"位置偏移大小: {np.linalg.norm(position_offset):.6f} 米")
    
    # 检查URDF文件信息
    print(f"\nURDF文件路径: {env.ur5_model.urdf_path}")
    
    # 获取关节信息
    print("\n关节信息:")
    for i, joint_id in enumerate(env.ur5_model.joint_ids):
        joint_info = p.getJointInfo(env.ur5_model.robot_id, joint_id, env.physics_client_id)
        print(f"关节 {i}: {joint_info[1].decode('utf-8')}")
        print(f"  关节位置: {joint_info[12]}")
        print(f"  关节方向: {joint_info[13]}")
        print(f"  关节轴: {joint_info[13]}")
    
    env.close()
    
    return position_offset

def create_calibrated_ik():
    """创建校准后的IK类"""
    
    print("\n=== 创建校准后的IK类 ===\n")
    
    # 分析坐标系差异
    position_offset = analyze_coordinate_difference()
    
    class CalibratedUR5IK:
        def __init__(self):
            self.ur5_kin = ur5e_ikfast.PyKinematics()
            self.n_joints = self.ur5_kin.getDOF()
            # 使用分析得到的偏移量
            self.position_offset = position_offset
            
        def forward_kinematic(self, joint_angles):
            ee_pose = self.ur5_kin.forward(joint_angles)
            ee_pose = np.asarray(ee_pose).reshape(3, 4)
            # 应用位置偏移
            ee_pose[:3, 3] += self.position_offset
            return ee_pose
            
        def inverse_kinematic(self, ee_pose, cur_joint_angles):
            # 移除位置偏移
            ee_pose_corrected = ee_pose.copy()
            ee_pose_corrected[:3, 3] -= self.position_offset
            
            joint_configs = self.ur5_kin.inverse(ee_pose_corrected.reshape(-1).tolist())
            n_solutions = int(len(joint_configs) / self.n_joints)
            if n_solutions == 0:
                print('no solution found')
                return None
            joint_configs = np.asarray(joint_configs).reshape(n_solutions, self.n_joints)
            
            # 选择最接近当前关节角度的解
            move_ranges = []
            for joint_config in joint_configs:
                move_range = 0
                for n in range(self.n_joints):
                    move_range += abs(joint_config[n]-cur_joint_angles[n])
                move_ranges.append(move_range)
            min_val = min(move_ranges)
            min_index = move_ranges.index(min_val)
            return joint_configs[min_index]
    
    return CalibratedUR5IK

def test_calibrated_ik():
    """测试校准后的IK"""
    
    print("\n=== 测试校准后的IK ===\n")
    
    # 创建校准后的IK类
    CalibratedUR5IK = create_calibrated_ik()
    calibrated_ik = CalibratedUR5IK()
    
    # 创建环境
    env = UR5AssemblyEnv(render_mode=None)
    obs, info = env.reset()
    
    # 测试关节配置
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
    
    # 获取校准后IKFast的末端执行器位姿
    calibrated_ikfast_pose_matrix = calibrated_ik.forward_kinematic(joint_angles)
    calibrated_ikfast_pos = calibrated_ikfast_pose_matrix[:3, 3]
    
    print(f"\n--- 校准后位置对比 ---")
    print(f"PyBullet位置: {pybullet_pos}")
    print(f"校准IKFast位置: {calibrated_ikfast_pos}")
    print(f"位置差异: {np.linalg.norm(pybullet_pos - calibrated_ikfast_pos):.6f} 米")
    
    # 测试逆向运动学
    print(f"\n--- 校准后逆向运动学测试 ---")
    
    # 使用校准后的IKFast进行逆运动学
    calibrated_ik_solution = calibrated_ik.inverse_kinematic(calibrated_ikfast_pose_matrix, joint_angles)
    
    if calibrated_ik_solution is not None:
        print(f"校准IKFast IK解: {calibrated_ik_solution}")
        print(f"校准IKFast IK解(度): {np.rad2deg(calibrated_ik_solution)}")
        print(f"校准IKFast IK误差: {np.linalg.norm(joint_angles - calibrated_ik_solution):.6f}")
        
        # 验证校准后的IK解
        env.ur5_model.set_joint_positions(calibrated_ik_solution, env.physics_client_id)
        for _ in range(50):
            p.stepSimulation(physicsClientId=env.physics_client_id)
        
        verify_pos, verify_orient = env.ur5_model.get_end_effector_pose(env.physics_client_id)
        print(f"校准IK解验证位置: {verify_pos}")
        print(f"目标位置: {pybullet_pos}")
        print(f"位置误差: {np.linalg.norm(verify_pos - pybullet_pos):.6f} 米")
    else:
        print("校准IKFast未找到解")
    
    env.close()

if __name__ == "__main__":
    # 分析坐标系差异
    analyze_coordinate_difference()
    
    # 测试校准后的IK
    test_calibrated_ik()

