import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append('/home/yjc/Project/UR5-RL')
sys.path.append('/home/yjc/Project/UR5-RL/ur_ikfast')

from envs.ur5_assembly_env import UR5AssemblyEnv
import ur5_ikfast
import time

class ur5_ik:
    def __init__(self):
        # Initialize kinematics for UR5 robot arm
        self.ur5_kin = ur5_ikfast.PyKinematics()
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



if __name__ == "__main__":
    print("开始测试UR5环境...")
    
    # 测试无GUI模式
    print("创建环境（无GUI模式）...")
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
    pos = pos-np.array([0, 0.0, 0.63])
    print(f"末端执行器位置: {pos}")
    print(f"末端执行器角度: {angle}")
    

    
    print("测试完成！")
    
    # 测试IK功能
    print("\n测试IK功能...")
    try:
        ur5_module = ur5_ik()
        print(f"UR5自由度: {ur5_module.n_joints}")
        
        # 测试正向运动学
        test_joints = np.array([0.0, 0, 0, 0.0, 0, 0.0])
        print("第一次测试：",test_joints)
        ee_pose = ur5_module.forward_kinematic(test_joints)
        print(f"ik:末端执行器位姿矩阵:\n{ee_pose}")
        pos, angle = env.ur5_model.get_end_effector_pose(env.physics_client_id)
        pos = pos-np.array([0, 0.0, 0.63])
        print(f"Pybullet末端执行器位置: {pos}")  
        test_joints = np.array([np.pi/2, 0, -np.pi/2, 0.0, 0, 0.0])
        time.sleep(5)
        print("第二次测试：",test_joints)
        ee_pose = ur5_module.forward_kinematic(test_joints)
        print(f"ik:末端执行器位姿矩阵:\n{ee_pose}")
        env.step(test_joints*2)
        pos, angle = env.ur5_model.get_end_effector_pose(env.physics_client_id)
        pos = pos-np.array([0, 0.0, 0.63])
        print(f"Pybullet末端执行器位置: {pos}")  
        pos = pos-np.array([0.7, 0.5, 0.2])
        angle = angle-np.array([0, 0, 0])
        env.ur5_model.set_end_effector_pose(pos, angle, env.physics_client_id)


        # 测试逆向运动学
        ik_solution = ur5_module.inverse_kinematic(ee_pose, test_joints)
        if ik_solution is not None:
            print(f"IK解: {ik_solution}")
        else:
            print("未找到IK解")
            
    except Exception as e:
        print(f"IK测试失败: {e}")
    time.sleep(100)
    
    print("关闭环境...")
    env.close()
# ur5_module = ur5_ik()
# print(ur5_module.forward_kinematic([-2.71316499e-01,  8.07371497e-01, -1.87837207e+00 , 1.07100058e+00
#  ,-2.71316499e-01 ,-3.55271368e-15]))