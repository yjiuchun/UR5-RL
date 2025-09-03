# UR5机械臂强化学习环境 - 快速开始指南

## 🚀 快速安装

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 测试环境
```bash
python run_test.py
```

如果看到 "✓ 所有测试通过！环境工作正常。" 说明环境安装成功。

## 🎯 基本使用

### 创建环境
```python
from envs.ur5_assembly_env import UR5AssemblyEnv

# 创建带GUI的环境
env = UR5AssemblyEnv(render_mode="human")

# 创建无GUI的环境（用于训练）
env = UR5AssemblyEnv(render_mode=None)
```

### 基本操作
```python
# 重置环境
obs, info = env.reset()

# 执行动作
action = env.action_space.sample()  # 随机动作
obs, reward, terminated, truncated, info = env.step(action)

# 关闭环境
env.close()
```

## 🔧 环境配置

### 动作类型
- `"joint_position"`: 关节位置控制（6维）
- `"end_effector_pose"`: 末端执行器位姿控制（6维）
- `"joint_velocity"`: 关节速度控制（6维）

### 渲染模式
- `"human"`: 显示GUI窗口
- `"rgb_array"`: 返回RGB图像
- `"depth_array"`: 返回深度图像
- `None`: 无渲染（最快）

### 任务参数
```python
env.set_task_parameters(
    peg_radius=0.02,        # 轴半径
    hole_radius=0.025,      # 孔半径
    position_threshold=0.01, # 位置误差阈值
    completion_bonus=10.0    # 完成奖励
)
```

## 📊 观测空间

### 关节状态 (18维)
- 关节位置 (6维)
- 关节速度 (6维)
- 关节力矩 (6维)

### 任务状态 (8维)
- 相对位置 (3维)
- 相对方向 (3维)
- 插入深度 (1维)
- 装配质量 (1维)

### 相机观测 (可选)
- RGB图像: (480, 640, 3)
- 深度图像: (480, 640)

## 🎮 示例代码

### 运行完整测试
```bash
python examples/test_env.py
```

### 运行训练示例
```bash
python examples/train_agent.py
```

## 📁 项目结构

```
ur5_rl/
├── envs/                    # 环境定义
│   ├── ur5_assembly_env.py # 主要环境类
│   └── __init__.py
├── models/                  # 机械臂模型
│   ├── ur5_model.py        # UR5机械臂模型
│   └── __init__.py
├── sensors/                 # 传感器模块
│   ├── camera.py           # 相机传感器
│   ├── joint_sensor.py     # 关节传感器
│   └── __init__.py
├── tasks/                   # 任务定义
│   ├── assembly_task.py    # 轴孔装配任务
│   └── __init__.py
├── utils/                   # 工具函数
│   ├── transforms.py       # 坐标变换
│   ├── visualization.py    # 可视化工具
│   └── __init__.py
├── examples/                # 示例代码
│   ├── test_env.py         # 环境测试
│   ├── train_agent.py      # 训练智能体
│   └── __init__.py
├── requirements.txt         # 依赖包
├── README.md               # 项目说明
├── QUICKSTART.md           # 快速开始指南
└── run_test.py             # 快速测试脚本
```

## 🐛 常见问题

### 1. PyBullet安装失败
```bash
# 尝试使用conda安装
conda install -c conda-forge pybullet

# 或者从源码安装
pip install git+https://github.com/bulletphysics/bullet3.git
```

### 2. 相机图像显示问题
确保安装了OpenCV：
```bash
pip install opencv-python
```

### 3. 可视化问题
确保安装了matplotlib：
```bash
pip install matplotlib
```

## 🔍 下一步

1. **熟悉环境**: 运行 `python run_test.py` 测试基本功能
2. **探索示例**: 查看 `examples/` 目录中的代码
3. **自定义任务**: 修改 `tasks/assembly_task.py` 中的参数
4. **实现智能体**: 基于现有代码实现自己的强化学习算法

## 📚 学习资源

- [PyBullet文档](https://pybullet.org/wordpress/)
- [Gymnasium文档](https://gymnasium.farama.org/)
- [UR5技术文档](https://www.universal-robots.com/articles/ur/interface-communication/real-time-data-exchange-rtde-guide/)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！
