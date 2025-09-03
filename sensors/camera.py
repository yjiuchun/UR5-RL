import numpy as np
import pybullet as p
from typing import Tuple, Optional
import cv2

class Camera:
    """相机传感器类"""
    
    def __init__(self, width: int = 640, height: int = 480, fov: float = 60.0):
        """
        初始化相机
        
        Args:
            width: 图像宽度
            height: 图像高度
            fov: 视场角（度）
        """
        self.width = width
        self.height = height
        self.fov = fov
        
        # 相机内参
        self.aspect = width / height
        self.near = 0.01
        self.far = 10.0
        
        # 投影矩阵
        self.projection_matrix = p.computeProjectionMatrixFOV(
            self.fov, self.aspect, self.near, self.far
        )
        
        # 相机位置和方向
        self.camera_position = np.array([0.5, 0.0, 0.8])
        self.camera_target = np.array([0.0, 0.0, 0.0])
        self.camera_up = np.array([0.0, 0.0, 1.0])
        
    def set_camera_pose(self, position: np.ndarray, target: np.ndarray, up: np.ndarray = None):
        """
        设置相机位姿
        
        Args:
            position: 相机位置 [x, y, z]
            target: 相机目标点 [x, y, z]
            up: 相机上方向 [x, y, z]
        """
        self.camera_position = np.array(position)
        self.camera_target = np.array(target)
        if up is not None:
            self.camera_up = np.array(up)
            
    def get_view_matrix(self) -> np.ndarray:
        """
        获取视图矩阵
        
        Returns:
            view_matrix: 4x4视图矩阵
        """
        view_matrix = p.computeViewMatrix(
            self.camera_position,
            self.camera_target,
            self.camera_up
        )
        return np.array(view_matrix).reshape(4, 4)
        
    def get_rgb_image(self, physics_client_id: int) -> np.ndarray:
        """
        获取RGB图像
        
        Args:
            physics_client_id: PyBullet客户端ID
            
        Returns:
            rgb_image: RGB图像 (height, width, 3)
        """
        # 获取RGB图像
        _, _, rgb_image, _, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.get_view_matrix(),
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=physics_client_id
        )
        
        # 转换格式
        rgb_image = np.array(rgb_image)
        rgb_image = rgb_image[:, :, :3]  # 只保留RGB通道
        
        return rgb_image
        
    def get_depth_image(self, physics_client_id: int) -> np.ndarray:
        """
        获取深度图像
        
        Args:
            physics_client_id: PyBullet客户端ID
            
        Returns:
            depth_image: 深度图像 (height, width)
        """
        # 获取深度图像
        _, _, _, depth_image, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.get_view_matrix(),
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=physics_client_id
        )
        
        # 转换深度值
        depth_image = np.array(depth_image)
        depth_image = self.far * self.near / (self.far - (self.far - self.near) * depth_image)
        
        return depth_image
        
    def get_point_cloud(self, physics_client_id: int) -> np.ndarray:
        """
        获取点云数据
        
        Args:
            physics_client_id: PyBullet客户端ID
            
        Returns:
            point_cloud: 点云数据 (height*width, 3)
        """
        depth_image = self.get_depth_image(physics_client_id)
        
        # 计算相机内参
        fx = self.width / (2 * np.tan(np.radians(self.fov / 2)))
        fy = self.height / (2 * np.tan(np.radians(self.fov / 2)))
        cx = self.width / 2
        cy = self.height / 2
        
        # 生成像素坐标
        y, x = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing='ij')
        
        # 计算3D坐标
        z = depth_image
        x = (x - cx) * z / fx
        y = (y - cy) * z / fy
        
        # 组合为点云
        point_cloud = np.stack([x, y, z], axis=-1)
        point_cloud = point_cloud.reshape(-1, 3)
        
        # 移除无效点
        valid_mask = point_cloud[:, 2] > 0
        point_cloud = point_cloud[valid_mask]
        
        return point_cloud
        
    def get_segmentation_image(self, physics_client_id: int) -> np.ndarray:
        """
        获取分割图像
        
        Args:
            physics_client_id: PyBullet客户端ID
            
        Returns:
            segmentation_image: 分割图像 (height, width)
        """
        # 获取分割图像
        _, _, _, _, segmentation_image = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.get_view_matrix(),
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=physics_client_id
        )
        
        return np.array(segmentation_image)
        
    def get_observation(self, physics_client_id: int) -> dict:
        """
        获取完整的相机观测
        
        Args:
            physics_client_id: PyBullet客户端ID
            
        Returns:
            observation: 包含RGB、深度等信息的字典
        """
        observation = {
            'rgb': self.get_rgb_image(physics_client_id),
            'depth': self.get_depth_image(physics_client_id),
            'segmentation': self.get_segmentation_image(physics_client_id),
            'point_cloud': self.get_point_cloud(physics_client_id)
        }
        
        return observation
        
    def save_image(self, image: np.ndarray, filename: str):
        """
        保存图像到文件
        
        Args:
            image: 要保存的图像
            filename: 文件名
        """
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
                
        cv2.imwrite(filename, image)
        
    def visualize_point_cloud(self, point_cloud: np.ndarray, output_file: str = None):
        """
        可视化点云数据
        
        Args:
            point_cloud: 点云数据
            output_file: 输出文件路径
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 随机采样点以加快显示速度
            if len(point_cloud) > 10000:
                indices = np.random.choice(len(point_cloud), 10000, replace=False)
                point_cloud = point_cloud[indices]
                
            ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], 
                      c=point_cloud[:, 2], cmap='viridis', s=0.1)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Point Cloud Visualization')
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
        except ImportError:
            print("matplotlib not available for point cloud visualization")
