#!/bin/bash

# UR5 IK Test 运行脚本
# 自动设置环境变量并运行ur5ik_test.py

echo "激活ur5_pybullet环境..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ur5_pybullet

echo "设置Python路径..."
export PYTHONPATH=/home/yjc/Project/UR5-RL:/home/yjc/Project/UR5-RL/ur_ikfast:$PYTHONPATH

echo "运行ur5ik_test.py..."
cd /home/yjc/Project/UR5-RL
python examples/ur5ik_test.py

echo "脚本执行完成！"
