# MA-Safe Transfer Learning

基于 Safety-Gymnasium（可变多 Agent）的多智能体安全迁移实验工程（SafeTransfer）。

## 1. 项目搭建与链路总览

本项目建议按“双环境 + 一条主链路”理解：

1. `SafeMARL` 环境：环境验证、FSRL 训练采样、OSRL 导出、DSRL 处理。
2. `FISOR` 环境：离线训练（中心化 Critic + 分布式 Actor）。
3. 主链路：环境验证 -> FSRL 训练 -> 采集 HDF5 -> OSRL/DSRL 处理 -> FISOR 离线训练。

适用目录（建议保持相对路径不变）：

```text
SafeMARL/
  SafeTransfer/ma_safe_migration
  SRL/safety-gymnasium-main
```

## 2. 环境命名规则

- 训练别名：`sg_ant_goal_n{N}`（例如 `sg_ant_goal_n6`）。
- 注册环境：`SafetyAntMultiGoalN{N}-v0`。

## 3. 环境创建说明（清晰版）

以下命令默认在 `SafeTransfer/ma_safe_migration` 目录执行。

### 3.1 系统依赖（Ubuntu）

```bash
sudo apt-get update
sudo apt-get install -y \
  git build-essential cmake pkg-config \
  libgl1-mesa-glx libgl1-mesa-dev libglew-dev libosmesa6-dev \
  libglfw3 libglfw3-dev patchelf ffmpeg xvfb
```

### 3.2 创建双 Conda 环境

```bash
conda create -n SafeMARL python=3.8 -y
conda create -n FISOR python=3.8 -y
conda env list
```

### 3.3 安装 `SafeMARL` 依赖（FSRL/OSRL/DSRL）

```bash
conda activate SafeMARL
python -m pip install --upgrade pip setuptools wheel

# 二选一：CPU / CUDA
# CPU:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# CUDA(示例 cu121):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
pip install -e ../../SRL/safety-gymnasium-main
pip install -e third_party/FSRL
pip install -e third_party/DSRL
pip install mujoco glfw xmltodict gymnasium-robotics
```

### 3.4 安装 `FISOR` 依赖（离线训练）具体请参考FISOR安装目录

```bash
conda activate FISOR
python -m pip install --upgrade pip setuptools wheel

# 二选一：CPU / CUDA
# CPU:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# CUDA(示例 cu121):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
pip install -e ../../SRL/safety-gymnasium-main
#这里的环境配置有问题，需要修改
pip install mujoco glfw xmltodict gymnasium-robotics
```

### 3.5 GPU 与渲染自检

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

## 4. 训练步骤

### Step 0) 环境可用性验证

```bash
conda activate SafeMARL
python verify_safety_gym_backend.py --num-agents 6 --steps 20 --render-mode rgb_array --randomize-layout
```

通过标准：无 ImportError，能正常 step 并输出 rewards/costs。

### Step 1) FSRL 训练 CTCE 策略

```bash
python third_party/FSRL/examples/customized/collect_dataset.py \
  --backend safety_gym \
  --num_agents 6 \
  --collect_only False \
  --epoch 10 \
  --step_per_epoch 5000 \
  --training_num 2 \
  --testing_num 1 \
  --device cpu \
  --name ctce_n6_train \
  --logdir ./third_party/FSRL/logs_local \
  --logger_backend tensorboard \
  --vector_env auto \
  --memory_guard auto
```

跨平台建议：

- Windows：保持 `--vector_env auto --memory_guard auto`。
- Linux/GPU：可改 `--vector_env shmem --memory_guard off` 并提高并行数。

### Step 2) 用 checkpoint 采集 HDF5 离线数据

```bash
python third_party/FSRL/examples/customized/collect_dataset.py \
  --backend safety_gym \
  --num_agents 6 \
  --collect_only True \
  --num_episodes 200 \
  --policy_checkpoint ./third_party/FSRL/logs_local/fast-safe-rl/ctce-n6-cost-5-100/<run_name>/checkpoint/model.pt \
  --output_dir ./data/6agents/ctce_collect \
  --dataset_name offline_dataset_ctce.hdf5 \
  --seed 11
```

### Step 3) 导出 OSRL 可读数据（NPZ）

```bash
python third_party/OSRL/examples/tools/export_ctce_hdf5_to_osrl.py \
  --input-hdf5 ./data/6agents/ctce_collect/<run_tag>/offline_dataset_ctce.hdf5 \
  --output-npz ./data/6agents/ctce_collect/<run_tag>/offline_dataset_osrl.npz \
  --num-agents 6 \
  --policy-tag fsrl_ctce \
  --checkpoint-id <run_name> \
  --seed 11
```

### Step 4) DSRL 数据处理与过滤（可选）

```bash
python third_party/DSRL/scripts/process_dataset.py \
  "./third_party/FSRL/logs_local/fast-safe-rl/ctce-n3-cost-5-100/<run_tag>/offline_dataset_ctce.hdf5" \
  --output ctce_n3 \
  --dir ./third_party/DSRL/processed \
  --cmin 0 --cmax 120 --rmin -200 --rmax 200 --save
```

典型输出：

- `fig/ctce_n3-120_before_filter.png`
- `fig/ctce_n3-120_after_filter.png`
- `third_party/DSRL/processed/ctce_n3-120-<traj_num>.hdf5`

### Step 5) FISOR 离线训练（CTCE）

在 `third_party/FISOR` 目录执行：

重要说明：

- 训练命令是 `launcher/examples/train_offline.py`。
- `launcher/examples/eval_offline.py` 只做评估，不会更新模型参数。

```powershell
conda activate FISOR
cd "d:\RL-lab\safe RL\SafeMARL\SafeTransfer\ma_safe_migration\third_party\FISOR"
$env:XLA_PYTHON_CLIENT_PREALLOCATE="false"

python launcher/examples/train_offline.py `
  --config configs/train_config.py:fisor `
  --custom_env_name sg_ant_goal_n4 `
  --num_agents 4 `
  --dataset_path "D:\RL-lab\safe RL\SafeMARL\SafeTransfer\ma_safe_migration\third_party\DSRL\processed\ctce_n4-120-951.hdf5" `
  --ratio 1.0 `
  --project "ctce_debug" `
  --experiment_name "ctce_n4_fisor" `
  --save_interval 50000 `
  --ctce_eval True `
  --wandb_mode online
```

Linux 服务器等价命令（推荐）：

```bash
cd /home/work3/SafeMARL/SafeMARLTransfer/SafeTransfer/ma_safe_migration/third_party/FISOR && \
python launcher/examples/train_offline.py \
  --config configs/train_config.py:fisor \
  --custom_env_name sg_ant_goal_n4 \
  --num_agents 4 \
  --dataset_path /home/work3/SafeMARL/SafeMARLTransfer/SafeTransfer/ma_safe_migration/third_party/DSRL/processed/ctce_n4-120-951.hdf5 \
  --ratio 1.0 \
  --project FISOR \
  --experiment_name ctce_n4_fisor \
  --save_interval 50000 \
  --ctce_eval=True \
  --wandb_mode online
```

### Step 6) FISOR 离线评估与可视化（CTCE）

在 `third_party/FISOR` 目录执行。建议优先使用一行命令，避免 shell 换行导致参数失效。

注意：下面命令是评估与可视化，不是训练。

1) 批量评估同一实验目录下全部 `model*.pickle`：

```bash
cd /home/work3/SafeMARL/SafeMARLTransfer/SafeTransfer/ma_safe_migration/third_party/FISOR && \
python launcher/examples/eval_offline.py \
  --model_location /home/work3/SafeMARL/SafeMARLTransfer/SafeTransfer/ma_safe_migration/third_party/FISOR/results/sg_ant_goal_n4/ctce_n4_fisor_2026-03-28_s135_915 \
  --evaluate_all=True \
  --eval_episodes=20 \
  --wandb_project FISOR \
  --wandb_mode online \
  --summary_filename eval_summary_n4.json
```

可选参数：

- `--wandb_entity <your_entity>`：仅在该 entity 确实存在时再传。
- `--wandb_mode offline`：网络或权限不稳定时建议离线。

评估输出：

- 每个 checkpoint 的指标汇总写入 `eval_summary.json`。
- 终端会逐个打印 `return / cost / episode_len`。

2) 生成 CTCE 可视化图（Value Slice）：

```bash
cd /home/work3/SafeMARL/SafeMARLTransfer/SafeTransfer/ma_safe_migration/third_party/FISOR && \
python launcher/viz/viz_map.py \
  --model_location /home/work3/SafeMARL/SafeMARLTransfer/SafeTransfer/ma_safe_migration/third_party/FISOR/results/sg_ant_goal_n4/ctce_n4_fisor_2026-03-28_s135_915 \
  --x_index 0 \
  --y_index 1 \
  --span 3.0 \
  --grid_size 121 \
  --output_image_name viz_map_ctce_model3.png
```

推荐：直接画“最安全模型”与“最高收益模型”的可行域边界（更容易比较）

```bash
cd /home/work3/SafeMARL/SafeMARLTransfer/SafeTransfer/ma_safe_migration/third_party/FISOR && \
python launcher/viz/viz_map.py \
  --model_location /home/work3/SafeMARL/SafeMARLTransfer/SafeTransfer/ma_safe_migration/third_party/FISOR/results/sg_ant_goal_n4/ctce_n4_fisor_2026-03-28_s135_915 \
  --model_file /home/work3/SafeMARL/SafeMARLTransfer/SafeTransfer/ma_safe_migration/third_party/FISOR/results/sg_ant_goal_n4/ctce_n4_fisor_2026-03-28_s135_915/model3.pickle \
  --x_index 0 --y_index 1 --span 3.0 --grid_size 161

cd /home/work3/SafeMARL/SafeMARLTransfer/SafeTransfer/ma_safe_migration/third_party/FISOR && \
python launcher/viz/viz_map.py \
  --model_location /home/work3/SafeMARL/SafeMARLTransfer/SafeTransfer/ma_safe_migration/third_party/FISOR/results/sg_ant_goal_n4/ctce_n4_fisor_2026-03-28_s135_915 \
  --model_file /home/work3/SafeMARL/SafeMARLTransfer/SafeTransfer/ma_safe_migration/third_party/FISOR/results/sg_ant_goal_n4/ctce_n4_fisor_2026-03-28_s135_915/model4.pickle \
  --x_index 0 --y_index 1 --span 3.0 --grid_size 161
```

说明：

- `viz_map.py` 会按 critic 类型自动选择边界阈值：`qc` 用 `cost_limit`，`hj` 用 0。
- 如需手动指定边界阈值（例如固定按 10），可加 `--boundary_value 10`。
- `eval_offline.py` 可用 `--summary_filename` 自定义结果 JSON 文件名（支持绝对路径）。
- `viz_map.py` 可用 `--output_image_name` 自定义输出图片名（支持绝对路径）。

可视化输出：

- 图片保存为 `imgs/viz_map_ctce.png`。

3) 一次性指定目录（可不传 `--model_location`）：

```bash
export MODEL_LOCATION=/home/work3/SafeMARL/SafeMARLTransfer/SafeTransfer/ma_safe_migration/third_party/FISOR/results/sg_ant_goal_n4/ctce_n4_fisor_2026-03-28_s135_915
cd /home/work3/SafeMARL/SafeMARLTransfer/SafeTransfer/ma_safe_migration/third_party/FISOR
python launcher/examples/eval_offline.py --evaluate_all=True --eval_episodes=20 --wandb_project FISOR --wandb_mode online
python launcher/viz/viz_map.py --x_index 0 --y_index 1 --span 3.0 --grid_size 121
```

## 5. FISOR 训练链路说明（当前实现）

本项目已在 CTCE 场景启用“中心化 Critic + 分布式 Actor”：

1. Critic 保持中心化：使用全局拼接观测和联合动作。
2. Actor 改为分布式：共享参数，按 agent 局部切片训练。
3. 执行阶段：先生成各 agent 局部动作，再拼接并由中心化 Critic 打分筛选。

### 图像可视化含义说明（`viz_map_ctce.png`）

该图是 CTCE 下的“状态切片价值图”（Value Slice）：

1. 横轴与纵轴：

- 分别对应 `obs[x_index]` 和 `obs[y_index]`（例如图中是 `obs[0]` 与 `obs[1]`）。
- 其余观测维度固定为一次 `env.reset()` 得到的基准状态。

2. 颜色：

- 颜色值来自 `safe_value` 网络在该网格点的输出。
- 这是局部切片下的安全价值分布，用于看模型在该二维子空间的风险/安全趋势。

3. 如何读图：

- 颜色变化平滑，说明该切片上价值函数连续性较好。
- 同色带越密，表示该区域价值变化更快（对状态扰动更敏感）。
- 若 `critic_type=qc`，通常可理解为“预测累计 cost”相关量，值越小一般越安全。
- 若 `critic_type=hj`，常以 0 等值线作为可行域边界（正负号可区分安全/不安全侧）。

4. 重要注意：

- 这是“二维切片”，不是全状态空间全貌。
- 不同 `x_index/y_index` 会展示不同局部结构，建议至少比较 3 组索引后再下结论。

入口行为：

- `third_party/FISOR/launcher/examples/train_offline.py` 在 CTCE 下自动注入 `decentralized_actor=True` 与 `num_agents`。
- `third_party/FISOR/configs/train_config.py` 提供兼容默认项：`decentralized_actor=False`、`num_agents=1`。

## 6. 常见问题（搭建与链路）

1. `num_agents mismatch`
   - 检查三处是否一致：`--num_agents`、`--custom_env_name` 的 N、数据文件名 `ctce_nN`。
2. 显存/内存压力大
   - 优先减小 `--ratio`，再降低 batch 或并发。
3. `safety_gymnasium` 导入异常
   - 确认在目标 conda 环境中执行，且已安装 `-e ../../SRL/safety-gymnasium-main`。

## 7. 迁移到服务器（GitHub 托管）

推荐流程：

1. 本地提交并 `git push`。
2. 服务器 `git clone`。
3. 按本 README 第 3 节重建双环境。
4. 按第 4 节执行最小链路验证。

```bash
cd /workspace
git clone https://github.com/<your_user>/<your_repo>.git
cd <your_repo>/SafeTransfer/ma_safe_migration
```

## 8. 常用脚本

- `verify_safety_gym_backend.py`：后端/渲染/step 验证。
- `test_environment.py`：环境回归测试。
- `third_party/FSRL/examples/customized/collect_dataset.py`：FSRL 训练与 CTCE 采集主入口。
- `third_party/OSRL/examples/tools/export_ctce_hdf5_to_osrl.py`：导出 OSRL/DSRL 可消费 NPZ 数据。
- `third_party/FISOR/launcher/examples/train_offline.py`：FISOR 离线训练入口。

## 9. 文档索引

- `docs/PROJECT_STRUCTURE.md`：目录与模块职责。
- `docs/FSRL_DSRL_OSRL_PIPELINE.md`：CTCE 采集到离线训练的数据管线。
- `docs/FISOR基线.md`：FISOR 理论拆解与 MA/CTDE 扩展设计。

## 10. 可变多智能体环境创建示例

```python
from envs.env_factory import make_marl_env

env = make_marl_env(num_agents=6, backend='safety_gym', render_mode='rgb_array')
obs, info = env.reset(seed=42)
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
obs, rewards, costs, terminations, truncations, infos = env.step(actions)
env.close()
```
