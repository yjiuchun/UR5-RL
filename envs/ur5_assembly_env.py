import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
import time
import os
import pybullet_data
from models.ur5_model import UR5Model
from sensors.camera import Camera
from sensors.joint_sensor import JointSensor
from tasks.assembly_task import AssemblyTask


# UR5_URDF_PATH = '/home/yjc/Project/UR5-RL/examples/UR5_Robotiq85_description/src/ur5_description/urdf/ur5_robotiq_85.urdf'
UR5_URDF_PATH = '/home/yjc/Project/UR5-RL/models/ur5e/ur5e.urdf'



class UR5AssemblyEnv(gym.Env):
    """UR5机械臂轴孔装配强化学习环境"""
    
    def __init__(self, 
                 render_mode: str = "human",
                 camera_width: int = 640,
                 camera_height: int = 480,
                 max_steps: int = 1000,
                 action_type: str = "joint_position",
                 show_camera_view: bool = False):
        """
        初始化环境
        
        Args:
            render_mode: 渲染模式 ("human", "rgb_array", "depth_array", None)
            camera_width: 相机图像宽度
            camera_height: 相机图像高度
            max_steps: 最大步数
            action_type: 动作类型 ("joint_position", "end_effector_pose", "joint_velocity")
            show_camera_view: 是否显示相机实时画面
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.max_steps = max_steps
        self.action_type = action_type
        self.show_camera_view = show_camera_view
        
        # 物理仿真
        self.physics_client_id = None
        self.time_step = 1.0 / 240.0  # 仿真时间步长
        
        # 机械臂和传感器
        self.ur5_model = None
        self.camera = None
        self.joint_sensor = None
        self.table = None
        # 任务
        self.assembly_task = None
        
        # 环境状态
        self.current_step = 0
        self.previous_task_state = None
        
        # 相机视图窗口
        self.camera_window_name = None
        if self.show_camera_view:
            import cv2
            self.camera_window_name = "UR5 Camera View"
            cv2.namedWindow(self.camera_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.camera_window_name, self.camera_width, self.camera_height)
        
        # 动作和观测空间
        self._setup_spaces()


        # 初始化环境
        self._initialize_environment()
        
    def _setup_spaces(self):
        """设置动作和观测空间"""
        if self.action_type == "joint_position":
            # 关节位置控制：6个关节的角度
            self.action_space = spaces.Box(
                low=np.array([-2*np.pi] * 6),
                high=np.array([2*np.pi] * 6),
                dtype=np.float32
            )
        elif self.action_type == "end_effector_pose":
            # 末端执行器位姿控制：位置(3) + 方向(3)
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0, 0.0, -np.pi, -np.pi, -np.pi]),
                high=np.array([1.0, 1.0, 1.0, np.pi, np.pi, np.pi]),
                dtype=np.float32
            )
        elif self.action_type == "joint_velocity":
            # 关节速度控制：6个关节的角速度
            self.action_space = spaces.Box(
                low=np.array([-np.pi] * 6),
                high=np.array([np.pi] * 6),
                dtype=np.float32
            )
        else:
            raise ValueError(f"Unsupported action type: {self.action_type}")
            
        # 观测空间
        # 关节状态：位置(6) + 速度(6) + 力矩(6)
        joint_obs = spaces.Box(
            low=np.array([-2*np.pi] * 6 + [-np.pi] * 6 + [-100] * 6),
            high=np.array([2*np.pi] * 6 + [np.pi] * 6 + [100] * 6),
            dtype=np.float32
        )
        
        # 任务状态：相对位置(3) + 相对方向(3) + 插入深度(1) + 装配质量(1)
        task_obs = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -np.pi, -np.pi, -np.pi, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, np.pi, np.pi, np.pi, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # 相机观测（如果启用）
        if self.render_mode in ["rgb_array", "depth_array"]:
            if self.render_mode == "rgb_array":
                camera_obs = spaces.Box(
                    low=0, high=255,
                    shape=(self.camera_height, self.camera_width, 3),
                    dtype=np.uint8
                )
            else:  # depth_array
                camera_obs = spaces.Box(
                    low=0.0, high=10.0,
                    shape=(self.camera_height, self.camera_width),
                    dtype=np.float32
                )
            
            # 组合观测空间
            self.observation_space = spaces.Dict({
                'joint_state': joint_obs,
                'task_state': task_obs,
                'camera': camera_obs
            })
        else:
            # 只包含关节和任务状态
            self.observation_space = spaces.Dict({
                'joint_state': joint_obs,
                'task_state': task_obs
            })
            
    def _initialize_environment(self):
        """初始化仿真环境"""
        # 连接PyBullet
        if self.render_mode == "human":
            self.physics_client_id = p.connect(p.GUI)
        else:
            self.physics_client_id = p.connect(p.DIRECT)
            
        # 设置仿真参数
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client_id)
        p.setTimeStep(self.time_step, physicsClientId=self.physics_client_id)
        p.setRealTimeSimulation(0, physicsClientId=self.physics_client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # 加载地面
        p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), "plane.urdf"),
            physicsClientId=self.physics_client_id
        )

        # 加载桌子
        self.table = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, 0.0],baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))


        
        # 初始化机械臂
        self.ur5_model = UR5Model(UR5_URDF_PATH)
        self.ur5_model.load(self.physics_client_id,base_position=[0, 0.0, 0.63],base_orientation=[0, 0, 0])
        
        # 初始化传感器
        self.camera = Camera(self.camera_width, self.camera_height)
        self.joint_sensor = JointSensor(self.ur5_model.robot_id, self.ur5_model.joint_ids)
        
        # 初始化任务
        self.assembly_task = AssemblyTask(self.physics_client_id)
        self.assembly_task.create_environment()
        
        # 设置相机位置
        self.camera.set_camera_pose(
            position=[0.0, 0.75, 2.0],
            target=[0.0001, 0.75, 0.0]
        )
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 重置选项
            
        Returns:
            observation: 初始观测
            info: 信息字典
        """
        super().reset(seed=seed)
        
        # 重置步数
        self.current_step = 0
        
        # 重置机械臂
        self.ur5_model.reset_joints()
        
        # 重置任务
        self.assembly_task.reset_task()
        
        # 重置传感器历史
        self.joint_sensor.reset_history()
        
        # 重置前一状态
        self.previous_task_state = None
        
        # 等待仿真稳定
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.physics_client_id)
            
        # 更新相机视图（如果启用）
        if self.show_camera_view:
            self._update_camera_view()
            
        # 获取初始观测
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
        
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        执行一步动作
        
        Args:
            action: 动作向量
            
        Returns:
            observation: 新的观测
            reward: 奖励值
            terminated: 是否终止
            truncated: 是否截断
            info: 信息字典
        """
        # 执行动作
        self._execute_action(action)
        
        # 仿真步进
        for _ in range(100):  # 每个动作执行多个仿真步
            p.stepSimulation(physicsClientId=self.physics_client_id)
            if self.render_mode == "human":
                time.sleep(self.time_step)
                
        # 更新相机视图（如果启用）
        if self.show_camera_view:
            self._update_camera_view()
                
        # 获取当前状态
        current_task_state = self.assembly_task.get_task_state()
        
        # 计算奖励
        reward = self.assembly_task.calculate_reward(
            current_task_state, self.previous_task_state
        )
        
        # 更新状态
        self.previous_task_state = current_task_state.copy()
        self.current_step += 1
        
        # 检查终止条件
        terminated = current_task_state.get('assembly_completed', False)
        truncated = self.current_step >= self.max_steps
        
        # 获取观测和信息
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
        
    def _execute_action(self, action: np.ndarray):
        """执行动作"""
        if self.action_type == "joint_position":
            # 关节位置控制
            self.ur5_model.set_joint_positions(action, self.physics_client_id)
            
        elif self.action_type == "end_effector_pose":
            # 末端执行器位姿控制
            position = action[:3]
            orientation = action[3:]
            self.ur5_model.set_end_effector_pose(
                position, orientation, self.physics_client_id
            )
            
        elif self.action_type == "joint_velocity":
            # 关节速度控制
            for i, joint_id in enumerate(self.ur5_model.joint_ids):
                p.setJointMotorControl2(
                    self.ur5_model.robot_id,
                    joint_id,
                    p.VELOCITY_CONTROL,
                    targetVelocity=action[i],
                    physicsClientId=self.physics_client_id
                )
                
    def _get_observation(self) -> Dict:
        """获取观测"""
        # 关节状态
        joint_states = self.joint_sensor.get_joint_states(self.physics_client_id)

        joint_obs = np.concatenate([
            joint_states['positions'],
            joint_states['velocities'],
            joint_states['forces']
        ])
        
        # 任务状态
        task_state = self.assembly_task.get_task_state()
        if task_state:
            task_obs = np.concatenate([
                task_state['relative_position'],
                task_state['relative_orientation'],
                [task_state['insertion_depth']],
                [task_state['assembly_quality']]
            ])
        else:
            task_obs = np.zeros(8)
            
        observation = {
            'joint_state': joint_obs.astype(np.float32),
            'task_state': task_obs.astype(np.float32)
        }
        
        # 相机观测
        if self.render_mode in ["rgb_array", "depth_array"]:
            if self.render_mode == "rgb_array":
                camera_obs = self.camera.get_rgb_image(self.physics_client_id)
            else:  # depth_array
                camera_obs = self.camera.get_depth_image(self.physics_client_id)
                
            observation['camera'] = camera_obs
            
        return observation
        
    def _get_info(self) -> Dict:
        """获取信息字典"""
        task_state = self.assembly_task.get_task_state()
        
        info = {
            'step': self.current_step,
            'max_steps': self.max_steps,
            'assembly_completed': task_state.get('assembly_completed', False),
            'assembly_quality': task_state.get('assembly_quality', 0.0),
            'insertion_depth': task_state.get('insertion_depth', 0.0)
        }
        
        # 添加关节统计信息
        joint_stats = self.joint_sensor.get_joint_statistics(self.physics_client_id)
        info.update(joint_stats)
        
        return info
        
    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            # GUI模式，不需要额外渲染
            return None
        elif self.render_mode == "rgb_array":
            return self.camera.get_rgb_image(self.physics_client_id)
        elif self.render_mode == "depth_array":
            return self.camera.get_depth_image(self.physics_client_id)
        else:
            return None
            
    def close(self):
        """关闭环境"""
        # 关闭相机视图窗口
        if self.show_camera_view and self.camera_window_name:
            try:
                import cv2
                cv2.destroyWindow(self.camera_window_name)
            except Exception as e:
                print(f"Warning: Failed to close camera window: {e}")
                
        if self.physics_client_id is not None:
            p.disconnect(physicsClientId=self.physics_client_id)
            
    def get_action_space_info(self) -> Dict:
        """获取动作空间信息"""
        return {
            'action_type': self.action_type,
            'action_dim': self.action_space.shape[0],
            'action_low': self.action_space.low,
            'action_high': self.action_space.high
        }
        
    def get_observation_space_info(self) -> Dict:
        """获取观测空间信息"""
        info = {}
        for key, space in self.observation_space.spaces.items():
            info[key] = {
                'shape': space.shape,
                'dtype': str(space.dtype),
                'low': space.low if hasattr(space, 'low') else None,
                'high': space.high if hasattr(space, 'high') else None
            }
        return info
        
    def set_task_parameters(self, **kwargs):
        """设置任务参数"""
        if self.assembly_task:
            self.assembly_task.set_task_parameters(**kwargs)
            
    def get_task_info(self) -> Dict:
        """获取任务信息"""
        if self.assembly_task:
            return self.assembly_task.get_task_info()
        return {}
        
    def _update_camera_view(self):
        """更新相机视图窗口"""
        if not self.show_camera_view or not self.camera_window_name:
            return
            
        try:
            import cv2
            
            # 获取RGB图像
            rgb_image = self.camera.get_rgb_image(self.physics_client_id)
            
            # 转换BGR格式（OpenCV使用BGR）
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            # 显示图像
            cv2.imshow(self.camera_window_name, bgr_image)
            cv2.waitKey(1)  # 更新窗口，1ms延迟
            
        except Exception as e:
            print(f"Warning: Failed to update camera view: {e}")
            
    def save_camera_image(self, filename: str, image_type: str = "rgb"):
        """保存相机图像"""
        if self.camera:
            if image_type == "rgb":
                image = self.camera.get_rgb_image(self.physics_client_id)
            elif image_type == "depth":
                image = self.camera.get_depth_image(self.physics_client_id)
            else:
                raise ValueError(f"Unsupported image type: {image_type}")
                
            self.camera.save_image(image, filename)
