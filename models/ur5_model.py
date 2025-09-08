import numpy as np
import pybullet as p
import pybullet_data
from typing import Tuple, List, Optional
import os

class UR5Model:
    """UR5机械臂模型类"""
    
    def __init__(self, urdf_path: Optional[str] = None):
        """
        初始化UR5机械臂模型
        
        Args:
            urdf_path: URDF文件路径，如果为None则使用默认路径
        """
        self.urdf_path = urdf_path or os.path.join(pybullet_data.getDataPath(), "ur5", "ur5.urdf")
        self.robot_id = None
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint', 
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        self.joint_ids = []
        self.num_joints = len(self.joint_names)
        
        # 关节限制
        self.joint_lower_limits = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
        self.joint_upper_limits = np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
        
        # 末端执行器偏移
        self.end_effector_offset = np.array([0, 0, 0.1])
        
        # 关节状态
        self.joint_positions = np.zeros(self.num_joints)
        self.joint_velocities = np.zeros(self.num_joints)
        self.joint_forces = np.zeros(self.num_joints)
        
    def load(self, physics_client_id: int, base_position: np.ndarray = None, base_orientation: np.ndarray = None):
        """
        加载机械臂模型到仿真环境
        
        Args:
            physics_client_id: PyBullet客户端ID
            base_position: 基座位置 [x, y, z]
            base_orientation: 基座方向 [roll, pitch, yaw]
        """
        if base_position is None:
            base_position = np.array([0, 0, 0])
        if base_orientation is None:
            base_orientation = np.array([0, 0, 0])
            
        # 转换方向为四元数
        base_quat = p.getQuaternionFromEuler(base_orientation)
        
        # 加载URDF
        self.robot_id = p.loadURDF(
            self.urdf_path,
            base_position,
            base_quat,
            useFixedBase=True,
            physicsClientId=physics_client_id
        )
        
        # 获取关节ID
        self.joint_ids = []
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=physics_client_id)
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=physics_client_id)
            joint_name = joint_info[1].decode('utf-8')
            if joint_name in self.joint_names:
                self.joint_ids.append(i)
            
        # 设置初始关节位置
        self.reset_joints(physics_client_id=physics_client_id)
        
    def reset_joints(self, joint_positions: np.ndarray = None, physics_client_id: int = None):
        """
        重置关节位置
        
        Args:
            joint_positions: 目标关节位置，如果为None则设为0
            physics_client_id: PyBullet客户端ID
        """
        if joint_positions is None:
            joint_positions = np.zeros(self.num_joints)
            
        for i, joint_id in enumerate(self.joint_ids):
            p.resetJointState(
                self.robot_id, 
                joint_id, 
                joint_positions[i],
                targetVelocity=0.0,
                physicsClientId=physics_client_id
            )
            
        self.joint_positions = joint_positions.copy()
        
    def get_joint_states(self, physics_client_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取关节状态
        
        Args:
            physics_client_id: PyBullet客户端ID
            
        Returns:
            joint_positions: 关节位置
            joint_velocities: 关节速度
            joint_forces: 关节力矩
        """
        joint_states = p.getJointStates(self.robot_id, self.joint_ids, physicsClientId=physics_client_id)
        
        self.joint_positions = np.array([state[0] for state in joint_states])
        self.joint_velocities = np.array([state[1] for state in joint_states])
        self.joint_forces = np.array([state[3] for state in joint_states])
        
        return self.joint_positions, self.joint_velocities, self.joint_forces
        
    def set_joint_positions(self, joint_positions: np.ndarray, physics_client_id: int):
        """
        设置关节位置
        
        Args:
            joint_positions: 目标关节位置
            physics_client_id: PyBullet客户端ID
        """
        # 限制关节范围
        joint_positions = np.clip(joint_positions, self.joint_lower_limits, self.joint_upper_limits)
        
        for i, joint_id in enumerate(self.joint_ids):
            p.setJointMotorControl2(
                self.robot_id,
                joint_id,
                p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                force=500,  # 增加最大力矩
                positionGain=0.1,  # 位置增益
                velocityGain=1.0,  # 速度增益
                physicsClientId=physics_client_id
            )
            
    def get_end_effector_pose(self, physics_client_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取末端执行器位姿
        
        Args:
            physics_client_id: PyBullet客户端ID
            
        Returns:
            position: 位置 [x, y, z]
            orientation: 方向 [roll, pitch, yaw]
        """
        # 获取末端执行器链接状态
        link_state = p.getLinkState(self.robot_id, self.joint_ids[-1], physicsClientId=physics_client_id)
        print(self.joint_ids[-1])
        position = np.array(link_state[0])
        orientation = p.getEulerFromQuaternion(link_state[1])
        
        return position, orientation
        
    def set_end_effector_pose(self, target_position: np.ndarray, target_orientation: np.ndarray, 
                             physics_client_id: int, max_iterations: int = 1000):
        """
        设置末端执行器位姿（使用逆运动学）
        
        Args:
            target_position: 目标位置 [x, y, z]
            target_orientation: 目标方向 [roll, pitch, yaw]
            physics_client_id: PyBullet客户端ID
            max_iterations: 最大迭代次数
        """
        # 转换方向为四元数
        target_quat = p.getQuaternionFromEuler(target_orientation)
        
        # 计算逆运动学
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            self.joint_ids[-1] - 1,
            target_position,
            target_orientation

        )
        print(joint_positions)
        # 设置关节位置
        self.set_joint_positions(joint_positions, physics_client_id)
        
    def get_jacobian(self, physics_client_id: int) -> np.ndarray:
        """
        获取雅可比矩阵
        
        Args:
            physics_client_id: PyBullet客户端ID
            
        Returns:
            jacobian: 雅可比矩阵 (6 x num_joints)
        """
        joint_positions, _, _ = self.get_joint_states(physics_client_id)
        
        # 计算雅可比矩阵
        jacobian = p.calculateJacobian(
            self.robot_id,
            self.joint_ids[-1],
            [0, 0, 0],  # 局部位置偏移
            joint_positions.tolist(),
            [0] * self.num_joints,  # 关节速度
            [0] * self.num_joints,  # 关节加速度
            physicsClientId=physics_client_id
        )
        
        return np.array(jacobian)
        
    def is_collision(self, physics_client_id: int) -> bool:
        """
        检查是否发生碰撞
        
        Args:
            physics_client_id: PyBullet客户端ID
            
        Returns:
            has_collision: 是否发生碰撞
        """
        # 检查与环境的碰撞
        collision_points = p.getContactPoints(bodyA=self.robot_id, physicsClientId=physics_client_id)
        return len(collision_points) > 0
