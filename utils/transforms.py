import numpy as np
from typing import Tuple, List, Optional
import math

def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """
    欧拉角转四元数 (ZYX顺序)
    
    Args:
        euler: 欧拉角 [roll, pitch, yaw] (弧度)
        
    Returns:
        quaternion: 四元数 [x, y, z, w]
    """
    roll, pitch, yaw = euler
    
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    quaternion = np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy
    ])
    
    return quaternion

def quaternion_to_euler(quaternion: np.ndarray) -> np.ndarray:
    """
    四元数转欧拉角 (ZYX顺序)
    
    Args:
        quaternion: 四元数 [x, y, z, w]
        
    Returns:
        euler: 欧拉角 [roll, pitch, yaw] (弧度)
    """
    x, y, z, w = quaternion
    
    # 归一化四元数
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm > 0:
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    # 计算欧拉角
    roll = math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = math.asin(2 * (w*y - z*x))
    yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    
    return np.array([roll, pitch, yaw])

def rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    旋转矩阵转欧拉角 (ZYX顺序)
    
    Args:
        rotation_matrix: 3x3旋转矩阵
        
    Returns:
        euler: 欧拉角 [roll, pitch, yaw] (弧度)
    """
    # 检查万向锁
    if abs(rotation_matrix[2, 0]) >= 1:
        # 万向锁情况
        yaw = 0
        if rotation_matrix[2, 0] < 0:
            pitch = math.pi / 2
            roll = yaw + math.atan2(rotation_matrix[0, 1], rotation_matrix[1, 1])
        else:
            pitch = -math.pi / 2
            roll = -yaw + math.atan2(-rotation_matrix[0, 1], rotation_matrix[1, 1])
    else:
        pitch = -math.asin(rotation_matrix[2, 0])
        roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    
    return np.array([roll, pitch, yaw])

def euler_to_rotation_matrix(euler: np.ndarray) -> np.ndarray:
    """
    欧拉角转旋转矩阵 (ZYX顺序)
    
    Args:
        euler: 欧拉角 [roll, pitch, yaw] (弧度)
        
    Returns:
        rotation_matrix: 3x3旋转矩阵
    """
    roll, pitch, yaw = euler
    
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    
    rotation_matrix = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])
    
    return rotation_matrix

def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """
    四元数转旋转矩阵
    
    Args:
        quaternion: 四元数 [x, y, z, w]
        
    Returns:
        rotation_matrix: 3x3旋转矩阵
    """
    x, y, z, w = quaternion
    
    rotation_matrix = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    return rotation_matrix

def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    旋转矩阵转四元数
    
    Args:
        rotation_matrix: 3x3旋转矩阵
        
    Returns:
        quaternion: 四元数 [x, y, z, w]
    """
    trace = np.trace(rotation_matrix)
    
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
    elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
        s = math.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2
        w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        x = 0.25 * s
        y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
        s = math.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2
        w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        y = 0.25 * s
        z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
        w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        z = 0.25 * s
    
    return np.array([x, y, z, w])

def homogeneous_transform(translation: np.ndarray, rotation: np.ndarray, 
                         rotation_type: str = "euler") -> np.ndarray:
    """
    创建齐次变换矩阵
    
    Args:
        translation: 平移向量 [x, y, z]
        rotation: 旋转（欧拉角、四元数或旋转矩阵）
        rotation_type: 旋转类型 ("euler", "quaternion", "matrix")
        
    Returns:
        transform_matrix: 4x4齐次变换矩阵
    """
    if rotation_type == "euler":
        rotation_matrix = euler_to_rotation_matrix(rotation)
    elif rotation_type == "quaternion":
        rotation_matrix = quaternion_to_rotation_matrix(rotation)
    elif rotation_type == "matrix":
        rotation_matrix = rotation
    else:
        raise ValueError(f"Unsupported rotation type: {rotation_type}")
    
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation
    
    return transform_matrix

def inverse_homogeneous_transform(transform_matrix: np.ndarray) -> np.ndarray:
    """
    计算齐次变换矩阵的逆
    
    Args:
        transform_matrix: 4x4齐次变换矩阵
        
    Returns:
        inverse_matrix: 逆变换矩阵
    """
    rotation_matrix = transform_matrix[:3, :3]
    translation = transform_matrix[:3, 3]
    
    inverse_rotation = rotation_matrix.T
    inverse_translation = -inverse_rotation @ translation
    
    inverse_matrix = np.eye(4)
    inverse_matrix[:3, :3] = inverse_rotation
    inverse_matrix[:3, 3] = inverse_translation
    
    return inverse_matrix

def transform_point(point: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    使用齐次变换矩阵变换点
    
    Args:
        point: 3D点 [x, y, z]
        transform_matrix: 4x4齐次变换矩阵
        
    Returns:
        transformed_point: 变换后的点
    """
    homogeneous_point = np.append(point, 1)
    transformed_homogeneous = transform_matrix @ homogeneous_point
    return transformed_homogeneous[:3]

def transform_vector(vector: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    使用齐次变换矩阵变换向量（只变换方向）
    
    Args:
        vector: 3D向量 [x, y, z]
        transform_matrix: 4x4齐次变换矩阵
        
    Returns:
        transformed_vector: 变换后的向量
    """
    rotation_matrix = transform_matrix[:3, :3]
    return rotation_matrix @ vector

def interpolate_pose(start_pose: np.ndarray, end_pose: np.ndarray, 
                    t: float, pose_type: str = "euler") -> np.ndarray:
    """
    在两个位姿之间插值
    
    Args:
        start_pose: 起始位姿 [position(3), orientation(3)]
        end_pose: 结束位姿 [position(3), orientation(3)]
        t: 插值参数 (0-1)
        pose_type: 位姿类型 ("euler", "quaternion")
        
    Returns:
        interpolated_pose: 插值后的位姿
    """
    # 位置插值（线性）
    start_pos = start_pose[:3]
    end_pos = end_pose[:3]
    interpolated_pos = start_pos + t * (end_pos - start_pos)
    
    # 方向插值
    start_ori = start_pose[3:]
    end_ori = end_pose[3:]
    
    if pose_type == "euler":
        # 欧拉角插值（线性）
        interpolated_ori = start_ori + t * (end_ori - start_ori)
    elif pose_type == "quaternion":
        # 四元数插值（球面线性插值）
        start_quat = euler_to_quaternion(start_ori)
        end_quat = euler_to_quaternion(end_ori)
        interpolated_quat = slerp(start_quat, end_quat, t)
        interpolated_ori = quaternion_to_euler(interpolated_quat)
    else:
        raise ValueError(f"Unsupported pose type: {pose_type}")
    
    return np.concatenate([interpolated_pos, interpolated_ori])

def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    四元数球面线性插值
    
    Args:
        q1: 起始四元数
        q2: 结束四元数
        t: 插值参数 (0-1)
        
    Returns:
        interpolated_quaternion: 插值后的四元数
    """
    # 确保四元数在同一个半球
    if np.dot(q1, q2) < 0:
        q2 = -q2
    
    # 计算角度
    cos_theta = np.clip(np.dot(q1, q2), -1, 1)
    theta = math.acos(cos_theta)
    
    if abs(theta) < 1e-6:
        return q1
    
    # 球面线性插值
    sin_theta = math.sin(theta)
    w1 = math.sin((1 - t) * theta) / sin_theta
    w2 = math.sin(t * theta) / sin_theta
    
    return w1 * q1 + w2 * q2

def normalize_angle(angle: float) -> float:
    """
    将角度归一化到 [-π, π]
    
    Args:
        angle: 输入角度（弧度）
        
    Returns:
        normalized_angle: 归一化后的角度
    """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def angle_difference(angle1: float, angle2: float) -> float:
    """
    计算两个角度之间的差值
    
    Args:
        angle1: 第一个角度（弧度）
        angle2: 第二个角度（弧度）
        
    Returns:
        difference: 角度差值 [-π, π]
    """
    diff = angle2 - angle1
    return normalize_angle(diff)
