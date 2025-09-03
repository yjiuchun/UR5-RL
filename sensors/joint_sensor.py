import numpy as np
from typing import Dict, List, Tuple
import pybullet as p

class JointSensor:
    """关节传感器类"""
    
    def __init__(self, robot_id: int, joint_ids: List[int]):
        """
        初始化关节传感器
        
        Args:
            robot_id: 机器人ID
            joint_ids: 关节ID列表
        """
        self.robot_id = robot_id
        self.joint_ids = joint_ids
        self.num_joints = len(joint_ids)
        
        # 关节信息
        self.joint_names = []
        self.joint_lower_limits = np.zeros(self.num_joints)
        self.joint_upper_limits = np.zeros(self.num_joints)
        self.joint_max_velocities = np.zeros(self.num_joints)
        self.joint_max_forces = np.zeros(self.num_joints)
        
        # 获取关节信息
        self._get_joint_info()
        
        # 历史数据
        self.joint_position_history = []
        self.joint_velocity_history = []
        self.joint_force_history = []
        self.max_history_length = 100
        
    def _get_joint_info(self):
        """获取关节信息"""
        for i, joint_id in enumerate(self.joint_ids):
            joint_info = p.getJointInfo(self.robot_id, joint_id)
            self.joint_names.append(joint_info[1].decode('utf-8'))
            self.joint_lower_limits[i] = joint_info[8]
            self.joint_upper_limits[i] = joint_info[9]
            self.joint_max_velocities[i] = joint_info[11]
            self.joint_max_forces[i] = joint_info[10]
            
    def get_joint_states(self, physics_client_id: int) -> Dict[str, np.ndarray]:
        """
        获取关节状态
        
        Args:
            physics_client_id: PyBullet客户端ID
            
        Returns:
            joint_states: 包含关节位置、速度、力矩的字典
        """
        joint_states = p.getJointStates(self.robot_id, self.joint_ids, physicsClientId=physics_client_id)
        
        joint_positions = np.array([state[0] for state in joint_states])
        joint_velocities = np.array([state[1] for state in joint_states])
        joint_forces = np.array([state[3] for state in joint_states])
        
        # 更新历史数据
        self._update_history(joint_positions, joint_velocities, joint_forces)
        
        return {
            'positions': joint_positions,
            'velocities': joint_velocities,
            'forces': joint_forces
        }
        
    def _update_history(self, positions: np.ndarray, velocities: np.ndarray, forces: np.ndarray):
        """更新历史数据"""
        self.joint_position_history.append(positions.copy())
        self.joint_velocity_history.append(velocities.copy())
        self.joint_force_history.append(forces.copy())
        
        # 限制历史长度
        if len(self.joint_position_history) > self.max_history_length:
            self.joint_position_history.pop(0)
            self.joint_velocity_history.pop(0)
            self.joint_force_history.pop(0)
            
    def get_joint_angles_deg(self, physics_client_id: int) -> np.ndarray:
        """
        获取关节角度（度）
        
        Args:
            physics_client_id: PyBullet客户端ID
            
        Returns:
            joint_angles_deg: 关节角度（度）
        """
        joint_states = self.get_joint_states(physics_client_id)
        return np.rad2deg(joint_states['positions'])
        
    def get_joint_velocities_deg(self, physics_client_id: int) -> np.ndarray:
        """
        获取关节角速度（度/秒）
        
        Args:
            physics_client_id: PyBullet客户端ID
            
        Returns:
            joint_velocities_deg: 关节角速度（度/秒）
        """
        joint_states = self.get_joint_states(physics_client_id)
        return np.rad2deg(joint_states['velocities'])
        
    def get_joint_torques(self, physics_client_id: int) -> np.ndarray:
        """
        获取关节力矩
        
        Args:
            physics_client_id: PyBullet客户端ID
            
        Returns:
            joint_torques: 关节力矩
        """
        joint_states = self.get_joint_states(physics_client_id)
        return joint_states['forces']
        
    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取关节限制
        
        Returns:
            lower_limits: 关节下限
            upper_limits: 关节上限
        """
        return self.joint_lower_limits, self.joint_upper_limits
        
    def get_joint_limits_deg(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取关节限制（度）
        
        Returns:
            lower_limits_deg: 关节下限（度）
            upper_limits_deg: 关节上限（度）
        """
        return np.rad2deg(self.joint_lower_limits), np.rad2deg(self.joint_upper_limits)
        
    def is_joint_limit_violated(self, physics_client_id: int) -> np.ndarray:
        """
        检查关节是否超出限制
        
        Args:
            physics_client_id: PyBullet客户端ID
            
        Returns:
            limit_violated: 布尔数组，表示每个关节是否超出限制
        """
        joint_states = self.get_joint_states(physics_client_id)
        positions = joint_states['positions']
        
        lower_violated = positions < self.joint_lower_limits
        upper_violated = positions > self.joint_upper_limits
        
        return lower_violated | upper_violated
        
    def get_joint_workspace_center(self) -> np.ndarray:
        """
        获取关节工作空间中心
        
        Returns:
            workspace_center: 工作空间中心位置
        """
        return (self.joint_lower_limits + self.joint_upper_limits) / 2
        
    def get_joint_workspace_range(self) -> np.ndarray:
        """
        获取关节工作空间范围
        
        Returns:
            workspace_range: 工作空间范围
        """
        return self.joint_upper_limits - self.joint_lower_limits
        
    def get_joint_velocity_limits(self) -> np.ndarray:
        """
        获取关节速度限制
        
        Returns:
            velocity_limits: 关节速度限制
        """
        return self.joint_max_velocities
        
    def get_joint_force_limits(self) -> np.ndarray:
        """
        获取关节力矩限制
        
        Returns:
            force_limits: 关节力矩限制
        """
        return self.joint_max_forces
        
    def get_joint_energy(self, physics_client_id: int) -> float:
        """
        计算关节总能量（动能）
        
        Args:
            physics_client_id: PyBullet客户端ID
            
        Returns:
            total_energy: 总动能
        """
        joint_states = self.get_joint_states(physics_client_id)
        velocities = joint_states['velocities']
        
        # 假设每个关节的转动惯量为1（简化计算）
        moment_of_inertia = 1.0
        kinetic_energy = 0.5 * moment_of_inertia * np.sum(velocities**2)
        
        return kinetic_energy
        
    def get_joint_power(self, physics_client_id: int) -> float:
        """
        计算关节总功率
        
        Args:
            physics_client_id: PyBullet客户端ID
            
        Returns:
            total_power: 总功率
        """
        joint_states = self.get_joint_states(physics_client_id)
        velocities = joint_states['velocities']
        forces = joint_states['forces']
        
        power = np.sum(velocities * forces)
        return power
        
    def get_joint_statistics(self, physics_client_id: int) -> Dict[str, float]:
        """
        获取关节统计信息
        
        Args:
            physics_client_id: PyBullet客户端ID
            
        Returns:
            statistics: 统计信息字典
        """
        joint_states = self.get_joint_states(physics_client_id)
        positions = joint_states['positions']
        velocities = joint_states['velocities']
        forces = joint_states['forces']
        
        statistics = {
            'position_mean': np.mean(positions),
            'position_std': np.std(positions),
            'position_range': np.ptp(positions),
            'velocity_mean': np.mean(velocities),
            'velocity_std': np.std(velocities),
            'velocity_max': np.max(np.abs(velocities)),
            'force_mean': np.mean(forces),
            'force_std': np.std(forces),
            'force_max': np.max(np.abs(forces)),
            'energy': self.get_joint_energy(physics_client_id),
            'power': self.get_joint_power(physics_client_id)
        }
        
        return statistics
        
    def reset_history(self):
        """重置历史数据"""
        self.joint_position_history = []
        self.joint_velocity_history = []
        self.joint_force_history = []
        
    def get_history_data(self) -> Dict[str, np.ndarray]:
        """
        获取历史数据
        
        Returns:
            history_data: 历史数据字典
        """
        if not self.joint_position_history:
            return {}
            
        return {
            'positions': np.array(self.joint_position_history),
            'velocities': np.array(self.joint_velocity_history),
            'forces': np.array(self.joint_force_history)
        }
