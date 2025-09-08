import numpy as np
import sys
import os

# 添加ur_ikfast路径
sys.path.append('/home/yjc/Project/UR5-RL/ur_ikfast')

import ur5e_ikfast

class CalibratedUR5IK:
    """
    校准后的UR5逆运动学类
    
    该类解决了IKFast和PyBullet之间的坐标系差异问题
    """
    
    def __init__(self):
        # 初始化IKFast
        self.ur5_kin = ur5e_ikfast.PyKinematics()
        self.n_joints = self.ur5_kin.getDOF()
        
        # 校准参数（通过分析得到的偏移量）
        # 这些参数补偿了IKFast和PyBullet之间的坐标系差异
        self.position_offset = np.array([-0.92606495, 0.5861432, 0.60597383])
        
    def forward_kinematic(self, joint_angles):
        """
        正向运动学：从关节角度计算末端执行器位姿
        
        Args:
            joint_angles: 关节角度数组 (6个关节)
            
        Returns:
            ee_pose: 4x4齐次变换矩阵
        """
        ee_pose = self.ur5_kin.forward(joint_angles)
        ee_pose = np.asarray(ee_pose).reshape(3, 4)
        
        # 应用位置偏移校准
        ee_pose[:3, 3] += self.position_offset
        
        # 转换为4x4齐次变换矩阵
        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[:3, :] = ee_pose
        
        return homogeneous_matrix
        
    def inverse_kinematic(self, ee_pose, cur_joint_angles=None):
        """
        逆向运动学：从末端执行器位姿计算关节角度
        
        Args:
            ee_pose: 4x4齐次变换矩阵或3x4变换矩阵
            cur_joint_angles: 当前关节角度，用于选择最接近的解
            
        Returns:
            joint_angles: 关节角度数组，如果无解则返回None
        """
        # 确保ee_pose是3x4矩阵
        if ee_pose.shape == (4, 4):
            ee_pose = ee_pose[:3, :]
        elif ee_pose.shape != (3, 4):
            raise ValueError("ee_pose must be either 3x4 or 4x4 matrix")
        
        # 移除位置偏移校准
        ee_pose_corrected = ee_pose.copy()
        ee_pose_corrected[:3, 3] -= self.position_offset
        
        # 计算逆运动学
        joint_configs = self.ur5_kin.inverse(ee_pose_corrected.reshape(-1).tolist())
        n_solutions = int(len(joint_configs) / self.n_joints)
        
        if n_solutions == 0:
            return None
            
        joint_configs = np.asarray(joint_configs).reshape(n_solutions, self.n_joints)
        
        # 如果没有提供当前关节角度，返回第一个解
        if cur_joint_angles is None:
            return joint_configs[0]
        
        # 选择最接近当前关节角度的解
        move_ranges = []
        for joint_config in joint_configs:
            move_range = np.sum(np.abs(joint_config - cur_joint_angles))
            move_ranges.append(move_range)
        
        min_index = np.argmin(move_ranges)
        return joint_configs[min_index]
        
    def get_end_effector_position(self, joint_angles):
        """
        获取末端执行器位置
        
        Args:
            joint_angles: 关节角度数组
            
        Returns:
            position: 末端执行器位置 [x, y, z]
        """
        ee_pose = self.forward_kinematic(joint_angles)
        return ee_pose[:3, 3]
        
    def get_end_effector_orientation(self, joint_angles):
        """
        获取末端执行器方向（旋转矩阵）
        
        Args:
            joint_angles: 关节角度数组
            
        Returns:
            orientation: 3x3旋转矩阵
        """
        ee_pose = self.forward_kinematic(joint_angles)
        return ee_pose[:3, :3]
        
    def get_all_ik_solutions(self, ee_pose):
        """
        获取所有逆运动学解
        
        Args:
            ee_pose: 4x4齐次变换矩阵或3x4变换矩阵
            
        Returns:
            solutions: 所有解的数组，如果没有解则返回空数组
        """
        # 确保ee_pose是3x4矩阵
        if ee_pose.shape == (4, 4):
            ee_pose = ee_pose[:3, :]
        elif ee_pose.shape != (3, 4):
            raise ValueError("ee_pose must be either 3x4 or 4x4 matrix")
        
        # 移除位置偏移校准
        ee_pose_corrected = ee_pose.copy()
        ee_pose_corrected[:3, 3] -= self.position_offset
        
        # 计算逆运动学
        joint_configs = self.ur5_kin.inverse(ee_pose_corrected.reshape(-1).tolist())
        n_solutions = int(len(joint_configs) / self.n_joints)
        
        if n_solutions == 0:
            return np.array([])
            
        return np.asarray(joint_configs).reshape(n_solutions, self.n_joints)
        
    def is_reachable(self, ee_pose):
        """
        检查末端执行器位姿是否可达
        
        Args:
            ee_pose: 4x4齐次变换矩阵或3x4变换矩阵
            
        Returns:
            is_reachable: 是否可达
        """
        solutions = self.get_all_ik_solutions(ee_pose)
        return len(solutions) > 0
        
    def get_joint_limits(self):
        """
        获取关节限制
        
        Returns:
            lower_limits: 关节下限
            upper_limits: 关节上限
        """
        # UR5的关节限制（弧度）
        lower_limits = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
        upper_limits = np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
        return lower_limits, upper_limits
        
    def is_joint_angles_valid(self, joint_angles):
        """
        检查关节角度是否在限制范围内
        
        Args:
            joint_angles: 关节角度数组
            
        Returns:
            is_valid: 是否有效
        """
        lower_limits, upper_limits = self.get_joint_limits()
        return np.all(joint_angles >= lower_limits) and np.all(joint_angles <= upper_limits)
        
    def get_calibration_info(self):
        """
        获取校准信息
        
        Returns:
            info: 校准信息字典
        """
        return {
            'position_offset': self.position_offset,
            'offset_magnitude': np.linalg.norm(self.position_offset),
            'n_joints': self.n_joints,
            'description': 'Calibrated UR5 IK with PyBullet coordinate system compensation'
        }

