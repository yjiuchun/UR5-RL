import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append('/home/yjc/Project/UR5-RL')
sys.path.append('/home/yjc/Project/UR5-RL/ur_ikfast')

from envs.ur5_assembly_env import UR5AssemblyEnv
import time



if __name__ == "__main__":
    print("开始测试UR5环境...")
    
    # 测试无GUI模式
    env = UR5AssemblyEnv(render_mode="human")
    
    print("重置环境...")
    obs, info = env.reset()
    print(f"初始观测空间: {list(obs.keys())}")
    print(f"关节状态维度: {obs['joint_state'].shape}")
    print(f"任务状态维度: {obs['task_state'].shape}")

    print("执行零动作...")
    action_sample = np.array([0.0, 0, 0, 0.0, 0, 0.0])
    obs, reward, terminated, truncated, info = env.step(action_sample)
    pos = []
    angle = []
    pos, angle = env.ur5_model.get_end_effector_pose(env.physics_client_id)
    # pos = pos-np.array([0, 0.0, 0.63])

    pos1 = pos
    angle1 = angle
    print(f"末端执行器位置: {pos}")
    print(f"末端执行器角度: {angle}")

    time.sleep(2)

    action_sample = np.array([0.7, 0.5, 0.2, 0, 0, 0])
    obs, reward, terminated, truncated, info = env.step(action_sample)
    print(env.ur5_model.get_joint_states(env.physics_client_id))
    pos, angle = env.ur5_model.get_end_effector_pose(env.physics_client_id)
    # pos = pos-np.array([0, 0.0, 0.63])


    time.sleep(2)

    print("设置末端执行器位姿...==============")
    env.ur5_model.set_end_effector_pose(pos1, angle1, env.physics_client_id)
    pos, angle = env.ur5_model.get_end_effector_pose(env.physics_client_id)
    print(f"末端执行器位置: {pos}")
    print(f"末端执行器角度: {angle}")

    time.sleep(20)
    
    
