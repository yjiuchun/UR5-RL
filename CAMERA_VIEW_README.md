# UR5机械臂环境相机实时视图功能

## 概述

UR5机械臂强化学习环境现在支持实时相机视图功能，可以在单独的窗口中显示机器人相机的实时画面，帮助用户更好地观察机器人的工作状态和环境情况。

## 功能特性

- **实时相机视图**: 在独立窗口中显示相机的实时RGB图像
- **可配置选项**: 通过 `show_camera_view` 参数控制是否启用
- **兼容性**: 与现有的渲染模式完全兼容
- **性能优化**: 最小化对仿真性能的影响

## 使用方法

### 1. 基本用法

```python
from envs.ur5_assembly_env import UR5AssemblyEnv

# 启用相机实时视图
env = UR5AssemblyEnv(
    render_mode="human",
    show_camera_view=True,  # 启用相机视图
    camera_width=640,
    camera_height=480
)
```

### 2. 相机视角控制

#### 预设视角

```python
# 俯视角度（从上方观察）
env.set_camera_view([0.0, 0.0, 2.0], [0.0, 0.0, 0.0])

# 侧视角度（从侧面观察）
env.set_camera_view([1.0, 0.0, 0.5], [0.0, 0.0, 0.5])

# 前视角度（从前方观察）
env.set_camera_view([0.0, -1.0, 0.5], [0.0, 0.0, 0.5])

# 45度俯视角度
env.set_camera_view([0.5, 0.5, 1.5], [0.0, 0.0, 0.0])

# 近距离观察
env.set_camera_view([0.2, 0.0, 0.3], [0.0, 0.0, 0.3])

# 远距离观察
env.set_camera_view([0.0, 0.0, 3.0], [0.0, 0.0, 0.0])
```

#### 自定义视角

```python
# 设置自定义相机位置和目标点
env.set_camera_view(
    position=[x, y, z],  # 相机位置
    target=[x, y, z],    # 目标点
    up=[0, 0, 1]        # 可选：相机上方向
)

# 获取当前相机信息
camera_info = env.get_camera_info()
print(camera_info)
```

### 2. 参数说明

- `show_camera_view`: 布尔值，控制是否显示相机实时视图
  - `True`: 启用相机视图，会打开一个独立的OpenCV窗口
  - `False`: 禁用相机视图（默认值）

- `camera_width` 和 `camera_height`: 控制相机图像的分辨率

### 3. 渲染模式兼容性

相机实时视图功能与所有渲染模式兼容：

- `render_mode="human"`: GUI模式 + 相机视图
- `render_mode="rgb_array"`: 无GUI + 相机视图
- `render_mode="depth_array"`: 无GUI + 相机视图
- `render_mode=None`: 无GUI + 相机视图

## 测试示例

### 1. 运行相机视图测试

```bash
cd /home/yjc/Project/UR5-RL
PYTHONPATH=/home/yjc/Project/UR5-RL python examples/test_camera_view.py
```

### 2. 运行相机视角控制测试

```bash
cd /home/yjc/Project/UR5-RL
PYTHONPATH=/home/yjc/Project/UR5-RL python examples/test_camera_angles.py
```

### 3. 运行完整环境测试（可选择相机视图）

```bash
cd /home/yjc/Project/UR5-RL
PYTHONPATH=/home/yjc/Project/UR5-RL python examples/test_env.py
```

## 技术实现

### 1. 相机视图更新

- 在环境的 `step()` 和 `reset()` 方法中自动更新相机视图
- 使用OpenCV的 `imshow()` 和 `waitKey()` 实现实时显示
- 图像格式从RGB转换为BGR（OpenCV标准）

### 2. 窗口管理

- 自动创建和销毁相机视图窗口
- 支持窗口大小调整
- 在环境关闭时自动清理资源

### 3. 错误处理

- 包含异常处理，确保相机视图错误不会影响仿真
- 提供警告信息帮助调试

## 注意事项

1. **依赖要求**: 需要安装OpenCV (`pip install opencv-python`)
2. **性能影响**: 启用相机视图会略微增加计算开销
3. **窗口管理**: 相机视图窗口独立于PyBullet的GUI窗口
4. **退出控制**: 在相机视图窗口中按 'q' 键可以退出测试

## 故障排除

### 1. 相机视图不显示

- 检查是否正确设置了 `show_camera_view=True`
- 确认OpenCV已正确安装
- 检查是否有其他程序占用了显示资源

### 2. 图像显示异常

- 检查相机参数设置（位置、目标点、FOV等）
- 确认仿真环境已正确初始化
- 查看控制台是否有错误信息

### 3. 性能问题

- 降低相机分辨率
- 减少相机视图更新频率
- 在不需要时禁用相机视图

## 扩展功能

### 1. 多相机支持

可以轻松扩展支持多个相机视图：

```python
# 在环境中添加多个相机
self.cameras = {
    'front': Camera(width=320, height=240),
    'side': Camera(width=320, height=240),
    'top': Camera(width=320, height=240)
}
```

### 2. 图像处理

可以在显示前添加图像处理功能：

```python
def _update_camera_view(self):
    # 获取原始图像
    rgb_image = self.camera.get_rgb_image(self.physics_client_id)
    
    # 添加图像处理（如边缘检测、目标识别等）
    processed_image = self.process_image(rgb_image)
    
    # 显示处理后的图像
    self.display_image(processed_image)
```

### 3. 录制功能

可以添加图像录制功能：

```python
def start_recording(self, filename):
    """开始录制相机画面"""
    self.recording = True
    self.video_writer = cv2.VideoWriter(
        filename, cv2.VideoWriter_fourcc(*'XVID'), 
        30, (self.camera_width, self.camera_height)
    )
```

## 总结

相机实时视图功能为UR5机械臂强化学习环境提供了更好的可视化支持，使用户能够：

- 实时观察机器人的工作状态
- 更好地理解环境变化
- 调试和优化算法
- 记录和分析机器人行为

通过简单的参数设置，用户可以根据需要启用或禁用这个功能，确保在需要时获得最佳的观察体验。
