# MA-Safe Transfer Learning

环境: Safety-Gymnasium （可变多Agent）的多智能体安全迁移实验工程（SafeTransfer ）。

## 实验现状（2026-03）

- 环境已统一到 Safety-Gymnasium MuJoCo Builder（`N>=2`）。
- 已接入 FSRL：可在当前环境维度上训练行为策略并用于离线数据采集。
- 已切换离线采集主入口：`third_party/FSRL/examples/customized/collect_dataset.py`（FSRL 原生脚本 + CTCE 环境接入）。
- 新方案：采集范式切换为 CTCE（把多智能体拼接成 super-agent），统一使用 FSRL TRPOlag。
- 目标任务：用 CTCE 采集到的数据进行值分解，并用于后续 Offline Safe RL 训练。
- 下一阶段：完成 CTCE 适配、固化值分解所需数据字段、接入 DSRL/OSRL 基线。

## 使用CTCE的方式收集Trajectory

当前计划采用 CTCE（Centralized Training, Centralized Execution）采集范式：

- 将所有 agent 的观测拼接为一个 super observation。
- 将所有 agent 的动作拼接为一个 super action。
- 使用 FSRL TRPOlag 直接在 super-agent 空间训练/采样。

**CTCE** ：用于统一数据采集接口；offlineSafeRL训练以值分解为核心目标。

## 环境命名

- 训练别名：`sg_ant_goal_n{N}`（如 `sg_ant_goal_n6`）
- 注册环境：`SafetyAntMultiGoalN{N}-v0`

## 安装

```bash
conda create -n SafeMARL python=3.8
conda activate SafeMARL
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

### 2) 训练 TRPOlag 策略（CTCE）

```bash
python third_party/FSRL/examples/customized/collect_dataset.py --backend safety_gym --num_agents 6 --collect_only False --epoch 10 --step_per_epoch 5000 --training_num 2 --testing_num 1 --device cpu --name ctce_n6_train --logdir ./third_party/FSRL/logs_local --logger_backend tensorboard --vector_env auto --memory_guard auto
```

跨平台建议：

- Windows：建议保持默认 `--vector_env auto --memory_guard auto`（会自动选择更稳妥的单进程向量环境并限制并发）。
- Linux/GPU：建议显式使用 `--vector_env shmem --memory_guard off`，并按机器资源提高 `--training_num`/`--testing_num`。

### 3) 使用 FSRL checkpoint 采集离线数据

```bash
python third_party/FSRL/examples/customized/collect_dataset.py --backend safety_gym --num_agents 6 --collect_only True --num_episodes 200 --policy_checkpoint ./third_party/FSRL/logs_local/fast-safe-rl/ctce-n6-cost-5-100/<run_name>/checkpoint/model.pt --output_dir ./data/6agents/ctce_collect --dataset_name offline_dataset_ctce.hdf5 --seed 11
```

### 4) 导出为 OSRL 可读数据

```bash
python third_party/OSRL/examples/tools/export_ctce_hdf5_to_osrl.py --input-hdf5 ./data/6agents/ctce_collect/<run_tag>/offline_dataset_ctce.hdf5 --output-npz ./data/6agents/ctce_collect/<run_tag>/offline_dataset_osrl.npz --num-agents 6 --policy-tag fsrl_ctce --checkpoint-id <run_name> --seed 11
```

### 5) 用 DSRL 脚本处理 trajectory 并绘制 cost-reward 图

```bash
python third_party/DSRL/scripts/process_dataset.py "./third_party/FSRL/logs_local/fast-safe-rl/ctce-n3-cost-5-100/<run_tag>/offline_dataset_ctce.hdf5" --output ctce_n3 --dir ./third_party/DSRL/processed --cmin 0 --cmax 120 --rmin -200 --rmax 200 --save
```

输出：

- `fig/ctce_n3-120_before_filter.png`（过滤前 cost-reward 散点图）
- `fig/ctce_n3-120_after_filter.png`（过滤后 cost-reward 散点图）
- `third_party/DSRL/processed/ctce_n3-120-<traj_num>.hdf5`（过滤后的离线数据）

建议：先用该步骤检查轨迹 reward/cost 覆盖范围，再决定下游训练使用原始数据还是过滤后数据。

## 常用脚本

- `verify_safety_gym_backend.py`：后端/渲染/step 验证
- `test_environment.py`：环境回归测试
- `third_party/FSRL/examples/customized/collect_dataset.py`：TRPOlag 训练与 CTCE 离线数据采集（唯一主入口）
- `third_party/OSRL/examples/tools/export_ctce_hdf5_to_osrl.py`：导出 OSRL/DSRL 可消费 NPZ 数据

历史入口说明：

- `experiments/train_fsrl_behavior_on_safetransfer.py` 为旧训练脚本，后续将移除，不建议继续使用。

## 文档索引

- `docs/PROJECT_STRUCTURE.md`：目录与模块职责
- `docs/NEXT_STEPS.md`：CTCE 方案下的推进计划

## 下一步目标（offline训练）

当前短期目标：使用离线数据完成 Offline （值分解主线）最小闭环。

1. 固化训练输入版本：
   - 原始 `offline_dataset_ctce.hdf5`
   - 过滤版 `third_party/DSRL/processed/*.hdf5`
2. 建立统一评估口径：episode reward、episode cost、constraint violation rate。
3. 先跑一个最小离线值分解基线，验证训练稳定性与可复现命令。
4. 对比两组数据（原始 vs 过滤）对 offlineRLMARL 的影响。

## 可变多智能体环境创建示例

```python
from envs.env_factory import make_marl_env

env = make_marl_env(num_agents=6, backend='safety_gym', render_mode='rgb_array')
obs, info = env.reset(seed=42)
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
obs, rewards, costs, terminations, truncations, infos = env.step(actions)
env.close()
```
