# UR5机械臂深度强化学习环境

这是一个用于UR5机械臂轴孔装配任务的深度强化学习环境。

## 项目结构

```
ur5_rl/
├── envs/                    # 环境定义
│   ├── __init__.py
│   ├── ur5_env.py          # 主要环境类
│   └── ur5_assembly_env.py # 轴孔装配任务环境
├── models/                  # 机械臂模型
│   ├── __init__.py
│   └── ur5_model.py        # UR5机械臂模型
├── sensors/                 # 传感器模块
│   ├── __init__.py
│   ├── camera.py           # 相机传感器
│   └── joint_sensor.py     # 关节传感器
├── tasks/                   # 任务定义
│   ├── __init__.py
│   └── assembly_task.py    # 轴孔装配任务
├── utils/                   # 工具函数
│   ├── __init__.py
│   ├── transforms.py       # 坐标变换
│   └── visualization.py    # 可视化工具
├── examples/                # 示例代码
│   ├── __init__.py
│   ├── test_env.py         # 环境测试
│   └── train_agent.py      # 训练智能体
├── requirements.txt         # 依赖包
└── README.md               # 项目说明
```

## 安装

1. 克隆项目：
```bash
git clone <repository_url>
cd ur5_rl
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 基本环境使用
```python
from envs.ur5_assembly_env import UR5AssemblyEnv

env = UR5AssemblyEnv()
obs = env.reset()
action = env.action_space.sample()
obs, reward, done, info = env.step(action)
```

### 训练智能体
```python
python examples/train_agent.py
```

## 环境特性

- **物理仿真**: 使用PyBullet进行高精度物理仿真
- **视觉感知**: 支持RGB和深度相机输入
- **任务定义**: 可配置的轴孔装配任务
- **奖励设计**: 基于任务完成度的奖励函数
- **接口标准**: 符合Gymnasium标准的环境接口

## 任务描述

轴孔装配任务要求机械臂将轴插入到指定的孔中，任务包括：
1. 抓取轴
2. 移动到孔的位置
3. 精确插入
4. 验证装配质量

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。
