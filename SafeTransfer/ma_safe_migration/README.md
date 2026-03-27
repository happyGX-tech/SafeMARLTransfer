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

### 6) 使用 FISOR 进行离线训练（中心化 Critic + 分布式 Actor）

本节整合 `train_offline.py` 当前链路和最常用操作，适用于本地 HDF5 离线数据训练。

#### 6.1 关键参数

- `--config`：配置文件（默认可用 `configs/train_config.py:fisor`）。
- `--dataset_path`：本地 HDF5 路径（传入后优先本地数据，不走远程 URL）。
- `--custom_env_name`：环境名覆盖（支持 `sg_ant_goal_nN` 或 `SafetyAntMultiGoalN{N}-v0`）。
- `--num_agents`：显式智能体数量（可选）。
- `--ratio`：数据采样比例（降内存、做快速验证）。

#### 6.2 当前训练链路

1. 环境创建：
   - `PointRobot` 走旧分支。
   - 其他环境若检测到 CTCE 条件（环境名/数据名/num_agents 任一可解析）则走 CTCE 分支。
2. 一致性校验：
   - 会联合校验 `--num_agents`、`--custom_env_name` 中的 N、`--dataset_path` 文件名中的 `ctce_nN`。
   - 三者冲突会直接报错并停止。
3. 数据加载：
   - 本地 HDF5 优先加载。
   - 自动进行 obs/action shape 匹配检查与 rewards/costs/dones 维度规范化。
4. 训练策略：
   - 非 CTCE：`sample_jax()` + 周期评估。
   - CTCE：`sample()`（numpy）降低设备内存压力，并默认跳过在线 `evaluate()`。

#### 6.3 PowerShell 最小操作

在 `third_party/FISOR` 目录执行：

```powershell
conda activate SafeMARL
cd "d:\RL-lab\safe RL\SafeMARL\SafeTransfer\ma_safe_migration\third_party\FISOR"
$env:XLA_PYTHON_CLIENT_PREALLOCATE="false"
```

标准离线训练（非 CTCE）：

```powershell
python launcher/examples/train_offline.py --env_id 0 --config configs/train_config.py:fisor
```

CTCE 离线训练（推荐显式参数）：

```powershell
python launcher/examples/train_offline.py `
  --config configs/train_config.py:fisor `
  --custom_env_name sg_ant_goal_n4 `
  --num_agents 4 `
  --dataset_path "D:\RL-lab\safe RL\SafeMARL\SafeTransfer\ma_safe_migration\third_party\DSRL\processed\ctce_n4-120-951.hdf5" `
  --ratio 0.3 `
  --project "ctce_debug" `
  --experiment_name "ctce_n4_fisor"
```

#### 6.4 常见问题

- `num_agents mismatch`：统一检查环境名 N、`--num_agents`、数据文件名中的 `ctce_nN`。
- 显存/内存压力大：先降 `--ratio`，再降 batch（CTCE 已有 batch=256 保护）。
- 导入错误版本 `safety_gymnasium`：优先在项目指定 conda 环境执行，避免 base 环境干扰。

#### 6.5 最小自检

- 控制台持续出现训练步进日志（非初始化后立即报错）。
- `results/<group>/<experiment_name>/config.json` 生成。
- wandb 可见 `train/*` 指标持续更新。

## 常用脚本

- `verify_safety_gym_backend.py`：后端/渲染/step 验证
- `test_environment.py`：环境回归测试
- `third_party/FSRL/examples/customized/collect_dataset.py`：TRPOlag 训练与 CTCE 离线数据采集（唯一主入口）
- `third_party/OSRL/examples/tools/export_ctce_hdf5_to_osrl.py`：导出 OSRL/DSRL 可消费 NPZ 数据

## 文档索引

- `docs/PROJECT_STRUCTURE.md`：目录与模块职责
- `docs/FSRL_DSRL_OSRL_PIPELINE.md`：CTCE 数据采集到离线训练的数据管线
- `docs/FISOR基线.md`：FISOR 理论拆解与 MA/CTDE 扩展设计

## FISOR 集成放置建议

为了与现有 FSRL/DSRL/OSRL 并行管理，并减少对上游镜像代码的污染，建议把 FISOR 仓库 clone 到：

```bash
third_party/FISOR/
```

理由：

- 与 `third_party/FSRL`、`third_party/DSRL`、`third_party/OSRL` 同层，第三方基线管理方式一致。
- 便于后续 pin commit、做补丁和复现实验。
- 可直接复用当前 `data/`、`envs/`、`docs/` 的工程产物，不打散主仓库业务代码。

## 目录冗余与命名决策

- `safety-gymnasium-main` 不需要强制移入 `ma_safe_migration/`；只要环境里已正确安装 `safety_gymnasium` 包，脚本即可运行。
- 为了兼容本地源码调试，适配器支持以下兜底路径：
  - `SRL/safety-gymnasium-main`（当前默认历史布局）
  - `third_party/safety-gymnasium-main`（可选同层布局）
  - `external/safety-gymnasium-main`（可选改名布局）
  - 或通过环境变量 `SAFETY_GYM_LOCAL_PATH` 指向任意本地源码路径
- `third_party/` 目前建议保留命名；若改名，需要同步更新安装命令、训练命令与文档路径引用。

## 查看训练情况：

```bash
tensorboard --logdir "D:\RL-lab\safe RL\SafeMARL\SafeTransfer\ma_safe_migration\third_party\FSRL\logs_local" --port 6006
然后浏览器打开
```

## 下一步目标（offline训练）

当前短期目标：使用离线数据完成 Offline （值分解主线）最小闭环。

1. 固化训练输入版本：
   - 原始 `offline_dataset_ctce.hdf5`
   - 过滤版 `third_party/DSRL/processed/*.hdf5`
2. 建立统一评估口径：episode reward、episode cost、constraint violation rate。
3. 先跑一个最小离线值分解基线，验证训练稳定性与可复现命令。
4. 对比两组数据（原始 vs 过滤）对 offlineRLMARL 的影响。

## 本次架构修改说明（中心化 Critic + 分布式 Actor）

为降低 CTCE 下 actor 训练不稳定和不收敛风险，已在 FISOR 中完成如下改造：

1. Critic 保持中心化：
   - reward critic/value 继续使用全局拼接状态和联合动作。
   - safety critic/value 继续使用全局拼接状态和联合动作。
2. Actor 改为分布式：
   - 一个共享参数的 diffusion actor，仅接收每个 agent 的局部观测切片，预测对应局部动作。
   - 每条离线样本先由中心化 critic 计算全局安全-奖励权重，再广播到每个局部 actor 样本做加权行为克隆。
3. 执行时动作生成：
   - 每个 agent 用局部观测独立采样候选局部动作。
   - 拼接成联合动作后，仍由中心化 critic 进行打分与筛选（maxq/minqc）。

当前入口脚本已自动在 CTCE 模式启用该结构，无需额外命令参数：

1. `third_party/FISOR/launcher/examples/train_offline.py`：CTCE 环境时自动传入 `decentralized_actor=True` 与 `num_agents`。
2. `third_party/FISOR/jaxrl5/agents/fisor/fisor.py`：新增分布式 actor 切分训练与采样逻辑。
3. `third_party/FISOR/configs/train_config.py`：增加 `decentralized_actor`、`num_agents` 配置项（默认兼容旧流程）。

## GitHub 托管迁移到服务器（含环境配置）

建议方式：本地提交到 GitHub，服务器直接 clone 后重建环境。

### A. 本地提交到 GitHub

1. 进入仓库根目录（建议 SafeMARL 作为根）。
2. 提交并推送：

```powershell
git add .
git commit -m "feat: centralized critic + decentralized actor for CTCE FISOR"
git push
```

### B. 服务器拉取

```bash
cd /workspace
git clone https://github.com/<your_user>/<your_repo>.git
cd <your_repo>/SafeTransfer/ma_safe_migration
```

### C. 服务器环境配置（Ubuntu）

1. 系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y \
  git build-essential cmake pkg-config \
  libgl1-mesa-glx libgl1-mesa-dev libglew-dev libosmesa6-dev \
  libglfw3 libglfw3-dev patchelf ffmpeg xvfb
```

2. Conda 环境：

```bash
conda create -n SafeMARL python=3.8 -y
conda activate SafeMARL
python -m pip install --upgrade pip setuptools wheel
```

3. PyTorch（CPU 稳妥版）：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

4. 项目依赖：

```bash
pip install -r requirements.txt
pip install -e ../../SRL/safety-gymnasium-main
pip install -e third_party/FSRL
pip install -e third_party/DSRL
pip install -e third_party/FISOR
pip install mujoco glfw xmltodict gymnasium-robotics
```

5. 无头渲染（建议写入 ~/.bashrc）：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

### D. 迁移后最小验证

1. 环境验证：

```bash
python verify_safety_gym_backend.py --num-agents 4 --steps 20 --render-mode rgb_array --randomize-layout
```

2. CTCE 离线训练（会自动启用中心化 critic + 分布式 actor）：

```bash
cd third_party/FISOR
python launcher/examples/train_offline.py --config configs/train_config.py:fisor --custom_env_name sg_ant_goal_n4 --num_agents 4 --dataset_path "<your_dataset_path>.hdf5" --experiment_name ctce_n4
```

## 可变多智能体环境创建示例

```python
from envs.env_factory import make_marl_env

env = make_marl_env(num_agents=6, backend='safety_gym', render_mode='rgb_array')
obs, info = env.reset(seed=42)
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
obs, rewards, costs, terminations, truncations, infos = env.step(actions)
env.close()
```
