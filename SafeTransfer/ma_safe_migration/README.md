# MA-Safe Migration Learning

基于 Safety-Gymnasium 的多智能体安全迁移实验工程（SafeTransfer ）。

## 项目现状（2026-03）

- 环境主链路已统一到 Safety-Gymnasium MuJoCo Builder（`N>=2`）。
- 已接入 FSRL：可在当前环境维度上训练行为策略并用于离线数据采集。
- 已有离线采集主入口：`run_data_collection.py`（支持 `random` / `variable_ckpt` / `fsrl_ckpt`）。
- 新方案：采集范式切换为 CTCE（把多智能体拼接成 super-agent），继续使用 FSRL 单智能体 PPO。
- 目标任务：用 CTCE 采集到的数据进行值分解，并用于后续 Offline Safe RL 训练。
- 下一阶段：完成 CTCE 适配、固化值分解所需数据字段、接入 DSRL/OSRL 基线。

## 重要说明（CTCE 新口径）

当前计划采用 CTCE（Centralized Training, Centralized Execution）采集范式：

- 将所有 agent 的观测拼接为一个 super observation。
- 将所有 agent 的动作拼接为一个 super action。
- 使用 FSRL 单智能体 PPO 直接在 super-agent 空间训练/采样。

CTCE 在本项目中的定位：用于统一数据采集接口；下游训练仍以值分解 Offline Safe RL 为核心目标。

该方案目标是优先保证采集链路统一、简化工程复杂度，不在当前阶段修改 FSRL 算法源码。

注意：截至当前代码版本，运行逻辑仍是“单受控 agent + 其他 agent 占位策略”。CTCE 为新的执行计划，详见 `docs/NEXT_STEPS.md`。

## 环境命名

- 训练别名：`sg_ant_goal_n{N}`（如 `sg_ant_goal_n6`）
- 注册环境：`SafetyAntMultiGoalN{N}-v0`

## 安装

```bash
conda create -n ma_safe python=3.8
conda activate ma_safe
pip install -r requirements.txt
pip install -e ../SRL/safety-gymnasium-main
pip install mujoco glfw xmltodict gymnasium-robotics
```

```bash
最稳妥安装（建议）

系统依赖：

sudo apt-get updatesudo apt-get install -y git build-essential cmake pkg-config \  libgl1-mesa-glx libgl1-mesa-dev libglew-dev libosmesa6-dev \  libglfw3 libglfw3-dev patchelf ffmpeg xvfb

Conda 环境：

conda create -n ma_safe python=3.8 -yconda activate ma_safepython -m pip install --upgrade pip setuptools wheel

PyTorch（CPU 版，最省事）：

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

项目依赖 + 本地三方包：

cd /your/path/SafeTransfer/ma_safe_migrationpip install -r requirements.txtpip install -e ../SRL/safety-gymnasium-mainpip install -e third_party/FSRLpip install -e third_party/DSRLpip install mujoco glfw xmltodict gymnasium-robotics


如果你用 NVIDIA GPU（替换上面 CPU 版 torch）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

快速验证
python verify_safety_gym_backend.py --num-agents 6 --steps 20 --render-mode rgb_array
如果
```

## 最小工作流

### 1) 验证环境

```bash
python verify_safety_gym_backend.py --num-agents 6 --steps 20 --render-mode rgb_array
```

### 2) 训练 FSRL 行为策略（维度对齐）

```bash
python experiments/train_fsrl_behavior_on_safetransfer.py --backend safety_gym --num-agents 6 --epoch 10 --step-per-epoch 5000 --training-num 2 --testing-num 1 --device cpu
```

### 3) 使用 FSRL checkpoint 采集离线数据

```bash
python run_data_collection.py --backend safety_gym --num-agents 6 --num-episodes 200 --policy-type fsrl_ckpt --policy-checkpoint ./third_party/FSRL/logs_local/<run_name> --policy-device cpu --output-dir ./data/6agents/fsrl_collect_v1
```

## 常用脚本

- `verify_safety_gym_backend.py`：后端/渲染/step 验证
- `test_environment.py`：环境回归测试
- `experiments/train_fsrl_behavior_on_safetransfer.py`：训练可对齐 FSRL 策略
- `run_data_collection.py`：离线数据采集

## 文档索引

- `docs/PROJECT_STRUCTURE.md`：目录与模块职责
- `docs/NEXT_STEPS.md`：CTCE 方案下的推进计划

## 环境创建示例

```python
from envs.env_factory import make_marl_env

env = make_marl_env(num_agents=6, backend='safety_gym', render_mode='rgb_array')
obs, info = env.reset(seed=42)
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
obs, rewards, costs, terminations, truncations, infos = env.step(actions)
env.close()
```
