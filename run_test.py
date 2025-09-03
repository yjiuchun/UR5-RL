#!/usr/bin/env python3
"""
简单的环境测试启动脚本
"""

import sys
import os

# 添加项目路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """主函数"""
    print("UR5机械臂强化学习环境")
    print("=" * 50)
    
    try:
        # 尝试导入环境
        from envs.ur5_assembly_env import UR5AssemblyEnv
        print("✓ 环境导入成功")
        
        # 创建环境（无GUI模式，快速测试）
        print("创建环境...")
        env = UR5AssemblyEnv(render_mode=None, max_steps=100)
        print("✓ 环境创建成功")
        
        # 测试基本功能
        print("测试基本功能...")
        obs, info = env.reset()
        print(f"✓ 环境重置成功，观测形状: {obs['joint_state'].shape}")
        
        # 测试随机动作
        print("测试随机动作...")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ 动作执行成功，奖励: {reward:.3f}")
        
        # 显示环境信息
        print("\n环境信息:")
        print(f"动作空间: {env.action_space}")
        print(f"观测空间: {env.observation_space}")
        print(f"任务信息: {env.get_task_info()}")
        
        env.close()
        print("\n✓ 所有测试通过！环境工作正常。")
        
    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        print("请确保已安装所有依赖包: pip install -r requirements.txt")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
