#!/usr/bin/env python3
"""
UR5机械臂强化学习训练脚本示例
"""

import numpy as np
import time
from envs.ur5_assembly_env import UR5AssemblyEnv
from utils.visualization import plot_reward_distribution, plot_assembly_progress

class RandomAgent:
    """随机智能体（用于演示）"""
    
    def __init__(self, action_space):
        self.action_space = action_space
        
    def select_action(self, observation):
        """选择动作"""
        return self.action_space.sample()

class SimplePolicyAgent:
    """简单策略智能体（基于启发式规则）"""
    
    def __init__(self, action_space, action_type="joint_position"):
        self.action_space = action_space
        self.action_type = action_type
        
        # 定义一些预定义的动作
        if action_type == "joint_position":
            self.predefined_actions = [
                np.array([0, 0, 0, 0, 0, 0]),  # 零位
                np.array([np.pi/4, -np.pi/4, 0, 0, 0, 0]),  # 前伸
                np.array([-np.pi/4, -np.pi/4, 0, 0, 0, 0]),  # 后伸
                np.array([0, -np.pi/2, 0, 0, 0, 0]),  # 下探
            ]
        else:
            self.predefined_actions = []
            
    def select_action(self, observation):
        """选择动作"""
        if self.predefined_actions and np.random.random() < 0.3:
            # 30%概率选择预定义动作
            return np.random.choice(self.predefined_actions)
        else:
            # 70%概率选择随机动作
            return self.action_space.sample()

def train_random_agent(env, num_episodes=100):
    """训练随机智能体"""
    print("=== 训练随机智能体 ===")
    
    agent = RandomAgent(env.action_space)
    episode_rewards = []
    episode_lengths = []
    assembly_success_rate = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        step_count = 0
        
        while True:
            # 选择动作
            action = agent.select_action(obs)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        # 记录结果
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        
        if info.get('assembly_completed', False):
            assembly_success_rate += 1
            
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: 奖励={total_reward:.2f}, 步数={step_count}, "
                  f"完成={info.get('assembly_completed', False)}")
    
    # 计算统计信息
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    success_rate = assembly_success_rate / num_episodes
    
    print(f"\n随机智能体训练结果:")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"平均步数: {avg_length:.1f}")
    print(f"成功率: {success_rate:.2%}")
    
    return episode_rewards, episode_lengths, success_rate

def train_simple_policy_agent(env, num_episodes=100):
    """训练简单策略智能体"""
    print("\n=== 训练简单策略智能体 ===")
    
    agent = SimplePolicyAgent(env.action_space, env.action_type)
    episode_rewards = []
    episode_lengths = []
    assembly_success_rate = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        step_count = 0
        
        while True:
            # 选择动作
            action = agent.select_action(obs)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        # 记录结果
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        
        if info.get('assembly_completed', False):
            assembly_success_rate += 1
            
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: 奖励={total_reward:.2f}, 步数={step_count}, "
                  f"完成={info.get('assembly_completed', False)}")
    
    # 计算统计信息
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    success_rate = assembly_success_rate / num_episodes
    
    print(f"\n简单策略智能体训练结果:")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"平均步数: {avg_length:.1f}")
    print(f"成功率: {success_rate:.2%}")
    
    return episode_rewards, episode_lengths, success_rate

def compare_agents(env, num_episodes=50):
    """比较不同智能体的性能"""
    print("\n=== 比较不同智能体性能 ===")
    
    # 训练随机智能体
    random_rewards, random_lengths, random_success = train_random_agent(env, num_episodes)
    
    # 训练简单策略智能体
    policy_rewards, policy_lengths, policy_success = train_simple_policy_agent(env, num_episodes)
    
    # 比较结果
    print(f"\n=== 性能比较 ===")
    print(f"{'智能体类型':<15} {'平均奖励':<12} {'平均步数':<12} {'成功率':<10}")
    print("-" * 55)
    print(f"{'随机智能体':<15} {np.mean(random_rewards):<12.2f} {np.mean(random_lengths):<12.1f} {random_success:<10.2%}")
    print(f"{'策略智能体':<15} {np.mean(policy_rewards):<12.2f} {np.mean(policy_lengths):<12.1f} {policy_success:<10.2%}")
    
    # 可视化比较
    plot_reward_distribution(random_rewards, save_path="random_agent_rewards.png")
    plot_reward_distribution(policy_rewards, save_path="policy_agent_rewards.png")
    
    return {
        'random': {'rewards': random_rewards, 'lengths': random_lengths, 'success': random_success},
        'policy': {'rewards': policy_rewards, 'lengths': policy_lengths, 'success': policy_success}
    }

def test_different_environments():
    """测试不同环境配置"""
    print("\n=== 测试不同环境配置 ===")
    
    # 测试不同的动作类型
    action_types = ["joint_position", "end_effector_pose"]
    results = {}
    
    for action_type in action_types:
        print(f"\n测试动作类型: {action_type}")
        
        try:
            env = UR5AssemblyEnv(render_mode="human", action_type=action_type, max_steps=300)
            
            # 训练随机智能体
            rewards, lengths, success = train_random_agent(env, num_episodes=30)
            
            results[action_type] = {
                'rewards': rewards,
                'lengths': lengths,
                'success': success
            }
            
            env.close()
            
        except Exception as e:
            print(f"动作类型 {action_type} 测试失败: {e}")
    
    # 比较不同动作类型
    if len(results) > 1:
        print(f"\n=== 不同动作类型性能比较 ===")
        print(f"{'动作类型':<20} {'平均奖励':<12} {'平均步数':<12} {'成功率':<10}")
        print("-" * 60)
        
        for action_type, result in results.items():
            print(f"{action_type:<20} {np.mean(result['rewards']):<12.2f} "
                  f"{np.mean(result['lengths']):<12.1f} {result['success']:<10.2%}")

def main():
    """主函数"""
    print("开始UR5机械臂强化学习训练")
    
    try:
        # 创建环境
        env = UR5AssemblyEnv(render_mode="human", max_steps=500)
        
        print(f"环境信息:")
        print(f"动作空间: {env.action_space}")
        print(f"观测空间: {env.observation_space}")
        print(f"最大步数: {env.max_steps}")
        
        # 比较不同智能体
        results = compare_agents(env, num_episodes=30)
        
        # 测试不同环境配置
        test_different_environments()
        
        print("\n训练完成！")
        print("生成的可视化文件:")
        print("- random_agent_rewards.png: 随机智能体奖励分布")
        print("- policy_agent_rewards.png: 策略智能体奖励分布")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    main()
