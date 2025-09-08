
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append('/home/yjc/Project/UR5-RL')
sys.path.append('/home/yjc/Project/UR5-RL/ur_ikfast')

from models.ur5_model import UR5Model
import pybullet as p
import time
import pybullet_data

if __name__ == "__main__":
    # 连接PyBullet
    physics_client_id = p.connect(p.GUI)
    
    # 设置物理仿真参数
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client_id)
    p.setTimeStep(1.0/240.0, physicsClientId=physics_client_id)
    p.setRealTimeSimulation(0, physicsClientId=physics_client_id)  # 关闭实时仿真，手动控制
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # 加载地面
    p.loadURDF("plane.urdf", physicsClientId=physics_client_id)
    
    # 创建UR5模型
    ur5_model = UR5Model('/home/yjc/Project/UR5-RL/models/ur5e/ur5e.urdf')
    ur5_model.load(physics_client_id)
    
    print("机械臂关节ID:", ur5_model.joint_ids)
    print("机械臂关节名称:", ur5_model.joint_names)
    
    # 获取初始关节状态
    initial_positions, initial_velocities, initial_forces = ur5_model.get_joint_states(physics_client_id)
    print("初始关节位置:", initial_positions)
    
    # 设置目标关节位置
    target_positions = np.array([0, -np.pi/4, np.pi/4, -np.pi/2, 0, 0])
    target_pos = np.array([0, 0, 1.0])
    target_angle = np.array([0, 0, 0])
    print("设置目标末端位置:", target_pos, target_angle)
    
    # 设置关节位置
    ur5_model.set_end_effector_pose(target_pos, target_angle, physics_client_id)
    
    # 运行仿真让机械臂移动到目标位置
    print("开始仿真...")
    for i in range(1000):  # 运行1000步仿真
        p.stepSimulation(physicsClientId=physics_client_id)
        if i % 100 == 0:  # 每100步打印一次状态
            current_positions, current_velocities, current_forces = ur5_model.get_joint_states(physics_client_id)
            print(f"步数 {i}: 当前关节位置 = {current_positions}")
        time.sleep(1.0/240.0)  # 控制仿真速度
    
    # 最终状态
    final_positions, final_velocities, final_forces = ur5_model.get_joint_states(physics_client_id)
    print("============================================================")
    print("最终末端位姿",ur5_model.get_end_effector_pose(physics_client_id))
    
    # 保持窗口打开
    print("按任意键退出...")
    input()
    
    # 断开连接
    p.disconnect(physics_client_id)