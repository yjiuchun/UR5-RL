import numpy as np
import pybullet as p
from typing import Dict, Tuple, Optional
import math

class AssemblyTask:
    """轴孔装配任务类"""
    
    def __init__(self, physics_client_id: int):
        """
        初始化装配任务
        
        Args:
            physics_client_id: PyBullet客户端ID
        """
        self.physics_client_id = physics_client_id
        
        # 任务参数
        self.peg_radius = 0.02  # 轴半径
        self.hole_radius = 0.025  # 孔半径
        self.peg_length = 0.1  # 轴长度
        self.hole_depth = 0.08  # 孔深度
        
        # 目标位置
        self.target_hole_position = np.array([0.0, 0.0, 0.05])
        self.target_hole_orientation = np.array([0.0, 0.0, 0.0])
        
        # 任务状态
        self.peg_id = None
        self.hole_id = None
        self.assembly_completed = False
        self.assembly_quality = 0.0
        
        # 奖励参数
        self.distance_weight = 1.0
        self.orientation_weight = 0.5
        self.insertion_weight = 2.0
        self.completion_bonus = 10.0
        
        # 成功阈值
        self.position_threshold = 0.01  # 位置误差阈值
        self.orientation_threshold = 0.1  # 方向误差阈值（弧度）
        self.insertion_threshold = 0.05  # 插入深度阈值
        
    def create_environment(self, peg_position: np.ndarray = None, hole_position: np.ndarray = None):
        """
        创建任务环境
        
        Args:
            peg_position: 轴的初始位置
            hole_position: 孔的初始位置
        """
        if peg_position is None:
            peg_position = np.array([0.2, 0.0, 0.1])
        if hole_position is None:
            hole_position = self.target_hole_position
            
        # 创建轴（圆柱体）
        peg_collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=self.peg_radius,
            height=self.peg_length,
            physicsClientId=self.physics_client_id
        )
        
        peg_visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.peg_radius,
            length=self.peg_length,
            rgbaColor=[0.8, 0.2, 0.2, 1.0],  # 红色
            physicsClientId=self.physics_client_id
        )
        
        self.peg_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=peg_collision_shape,
            baseVisualShapeIndex=peg_visual_shape,
            basePosition=peg_position,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.physics_client_id
        )
        
        # 创建孔（圆柱体，挖空）
        hole_collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=self.hole_radius,
            height=self.hole_depth,
            physicsClientId=self.physics_client_id
        )
        
        hole_visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.hole_radius,
            length=self.hole_depth,
            rgbaColor=[0.2, 0.2, 0.8, 1.0],  # 蓝色
            physicsClientId=self.physics_client_id
        )
        
        self.hole_id = p.createMultiBody(
            baseMass=0.0,  # 静态物体
            baseCollisionShapeIndex=hole_collision_shape,
            baseVisualShapeIndex=hole_visual_shape,
            basePosition=hole_position,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.physics_client_id
        )
        
    def get_task_state(self) -> Dict:
        """
        获取任务状态
        
        Returns:
            task_state: 任务状态字典
        """
        if self.peg_id is None or self.hole_id is None:
            return {}
            
        # 获取轴和孔的位置和方向
        peg_pos, peg_quat = p.getBasePositionAndOrientation(
            self.peg_id, physicsClientId=self.physics_client_id
        )
        hole_pos, hole_quat = p.getBasePositionAndOrientation(
            self.hole_id, physicsClientId=self.physics_client_id
        )
        
        peg_orientation = p.getEulerFromQuaternion(peg_quat)
        hole_orientation = p.getEulerFromQuaternion(hole_quat)
        
        # 计算相对位置和方向
        relative_position = np.array(peg_pos) - np.array(hole_pos)
        relative_orientation = np.array(peg_orientation) - np.array(hole_orientation)
        
        # 计算插入深度
        insertion_depth = self._calculate_insertion_depth()
        
        # 计算装配质量
        self.assembly_quality = self._calculate_assembly_quality(relative_position, relative_orientation, insertion_depth)
        
        # 检查是否完成
        self.assembly_completed = self._check_completion(relative_position, relative_orientation, insertion_depth)
        
        return {
            'peg_position': np.array(peg_pos),
            'peg_orientation': np.array(peg_orientation),
            'hole_position': np.array(hole_pos),
            'hole_orientation': np.array(hole_orientation),
            'relative_position': relative_position,
            'relative_orientation': relative_orientation,
            'insertion_depth': insertion_depth,
            'assembly_quality': self.assembly_quality,
            'assembly_completed': self.assembly_completed
        }
        
    def _calculate_insertion_depth(self) -> float:
        """计算插入深度"""
        if self.peg_id is None or self.hole_id is None:
            return 0.0
            
        peg_pos, _ = p.getBasePositionAndOrientation(
            self.peg_id, physicsClientId=self.physics_client_id
        )
        hole_pos, _ = p.getBasePositionAndOrientation(
            self.hole_id, physicsClientId=self.physics_client_id
        )
        
        # 计算沿Z轴的插入深度
        insertion_depth = hole_pos[2] - peg_pos[2] + self.peg_length / 2
        
        return max(0.0, min(insertion_depth, self.hole_depth))
        
    def _calculate_assembly_quality(self, relative_position=None, relative_orientation=None, insertion_depth=None) -> float:
        """计算装配质量（0-1之间）"""
        if relative_position is None or relative_orientation is None or insertion_depth is None:
            return 0.0
            
        # 位置误差
        position_error = np.linalg.norm(relative_position[:2])  # 只考虑XY平面
        position_quality = max(0, 1 - position_error / self.hole_radius)
        
        # 方向误差
        orientation_error = np.linalg.norm(relative_orientation)
        orientation_quality = max(0, 1 - orientation_error / (np.pi / 4))
        
        # 插入深度质量
        insertion_quality = insertion_depth / self.hole_depth
        
        # 综合质量
        overall_quality = (position_quality * 0.4 + 
                          orientation_quality * 0.3 + 
                          insertion_quality * 0.3)
        
        return np.clip(overall_quality, 0, 1)
        
    def _check_completion(self, relative_position=None, relative_orientation=None, insertion_depth=None) -> bool:
        """检查任务是否完成"""
        if relative_position is None or relative_orientation is None or insertion_depth is None:
            return False
            
        # 检查位置误差
        position_error = np.linalg.norm(relative_position[:2])
        if position_error > self.position_threshold:
            return False
            
        # 检查方向误差
        orientation_error = np.linalg.norm(relative_orientation)
        if orientation_error > self.orientation_threshold:
            return False
            
        # 检查插入深度
        if insertion_depth < self.insertion_threshold:
            return False
            
        return True
        
    def calculate_reward(self, current_state: Dict, previous_state: Dict = None) -> float:
        """
        计算奖励值
        
        Args:
            current_state: 当前状态
            previous_state: 前一状态
            
        Returns:
            reward: 奖励值
        """
        if not current_state:
            return 0.0
            
        reward = 0.0
        
        # 距离奖励
        position_error = np.linalg.norm(current_state['relative_position'][:2])
        distance_reward = -self.distance_weight * position_error
        reward += distance_reward
        
        # 方向奖励
        orientation_error = np.linalg.norm(current_state['relative_orientation'])
        orientation_reward = -self.orientation_weight * orientation_error
        reward += orientation_reward
        
        # 插入深度奖励
        insertion_reward = self.insertion_weight * current_state['insertion_depth']
        reward += insertion_reward
        
        # 完成奖励
        if current_state['assembly_completed']:
            reward += self.completion_bonus
            
        # 进度奖励（与前一状态比较）
        if previous_state:
            progress_reward = (current_state['assembly_quality'] - 
                             previous_state['assembly_quality']) * 5.0
            reward += progress_reward
            
        return reward
        
    def reset_task(self, peg_position: np.ndarray = None, hole_position: np.ndarray = None):
        """
        重置任务
        
        Args:
            peg_position: 轴的初始位置
            hole_position: 孔的初始位置
        """
        # 移除现有物体
        if self.peg_id is not None:
            p.removeBody(self.peg_id, physicsClientId=self.physics_client_id)
            self.peg_id = None
            
        if self.hole_id is not None:
            p.removeBody(self.hole_id, physicsClientId=self.physics_client_id)
            self.hole_id = None
            
        # 重置状态
        self.assembly_completed = False
        self.assembly_quality = 0.0
        
        # 重新创建环境
        self.create_environment(peg_position, hole_position)
        
    def get_task_info(self) -> Dict:
        """
        获取任务信息
        
        Returns:
            task_info: 任务信息字典
        """
        return {
            'peg_radius': self.peg_radius,
            'hole_radius': self.hole_radius,
            'peg_length': self.peg_length,
            'hole_depth': self.hole_depth,
            'target_hole_position': self.target_hole_position,
            'target_hole_orientation': self.target_hole_orientation,
            'position_threshold': self.position_threshold,
            'orientation_threshold': self.orientation_threshold,
            'insertion_threshold': self.insertion_threshold,
            'distance_weight': self.distance_weight,
            'orientation_weight': self.orientation_weight,
            'insertion_weight': self.insertion_weight,
            'completion_bonus': self.completion_bonus
        }
        
    def set_task_parameters(self, **kwargs):
        """
        设置任务参数
        
        Args:
            **kwargs: 要设置的参数
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
    def is_valid_action(self, action: np.ndarray) -> bool:
        """
        检查动作是否有效
        
        Args:
            action: 动作向量
            
        Returns:
            is_valid: 动作是否有效
        """
        # 这里可以添加动作有效性检查逻辑
        # 例如检查关节限制、碰撞等
        return True
