# FISOR Oracle 训练模式设计与使用说明

为了解决分布式多智能体协作中 centralized cost (单一全局代价)导致惩罚被摊平甚至丢失每个智能体个体违反安全约束特征的问题，本项目为 FISOR 扩展了 `oracle_cost_mode` 训练模式。

## 核心设计

**核心思路：多维度的 Cost 网络输出**

与完全去中心化的局部网络不同，`fisor_oracle` 继续保持 Cost Critic 的网络参数共享（以提高多智能体学习效率），但要求该网络同时输出 `num_agents` 个维度的代价 Q 和 V。
即：网络接收系统完整观测（或所有智能体观测及动作），输出 `(batch_size, num_agents)` 形状的估值矩阵。更新过程则结合数据集中保存的针对每个智能体独立的真实代价 `costs_per_agent` 进行维度对齐的目标计算。

如此一来，网络能有针对性地对所有违规个体进行精确归因，大幅减少信息丢失风险；而 Actor 在更新过程中也会获取多维度的 advantage 权重。

## 环境和数据修改

此网络模式高度依赖环境能够计算和提供每个智能体独立的惩罚数据，并在离线数据集 (`HDF5`) 中以 `costs_per_agent` 保存（维度如 `(B, num_agents)` 或以 `N` 个独立的 field 处理），而非原本简单的聚合标量 `(B,)`。

使用此模式的必要前置条件：

- 数据内包含属性 `'costs_per_agent'`，可以通过定制化 FSRL 或相关收集脚本在收集时添加存留。

## 如何使用 `fisor_oracle`

### 1. 配置准备

在 `third_party/FISOR/configs/train_config.py` 中已经内置了预设配置 `"fisor_oracle"`，它指定 `agent_kwargs` 中的 `oracle_cost_mode=True` 与实际对应系统的 `num_agents`，修改此处参数对应你的环境人数即可。

### 2. 训练指令

你可以与原本训练离线模型一致启动，但在命令行参数指定新的 config name：

```powershell
conda activate FISOR
cd "d:\RL-lab\safe RL\SafeMARL\SafeTransfer\ma_safe_migration\third_party\FISOR"
$env:XLA_PYTHON_CLIENT_PREALLOCATE="false"

python launcher/examples/train_offline.py --config configs/train_config.py:fisor_oracle --custom_env_name sg_ant_goal_n4 --num_agents 4 --dataset_path /home/work3/SafeMARL/SafeMARLTransfer/SafeTransfer/ma_safe_migration/third_party/DSRL/processed/ctce_n4-120-951.hdf5 --ratio 1.0 --project FISOR --experiment_name ctce_n4_fisor_oracle --max_steps 500000
```

### 3. 可视化与评估

针对 `oracle_cost_mode` 版本，现已完全兼容常规的测试与可视化脚本。

**批量评估所有模型 (eval_offline.py)**

```bash
python launcher/examples/eval_offline.py \
  --model_location /home/work3/SafeMARL/SafeMARLTransfer/SafeTransfer/ma_safe_migration/third_party/FISOR/results/sg_ant_goal_n4/ctce_n4_fisor_oracle_2026-X-X_sX_X \
  --evaluate_all=True \
  --eval_episodes=20 \
  --wandb_project FISOR \
  --wandb_mode online \
  --summary_filename eval_summary_n4.json
```

**生成可视化语义双图 (viz_map.py)**

```bash
python launcher/viz/viz_map.py \
  --model_location /home/work3/SafeMARL/SafeMARLTransfer/SafeTransfer/ma_safe_migration/third_party/FISOR/results/sg_ant_goal_n4/ctce_n4_fisor_oracle_2026-X-X_sX_X \
  --model_file /home/work3/SafeMARL/SafeMARLTransfer/SafeTransfer/ma_safe_migration/third_party/FISOR/results/sg_ant_goal_n4/ctce_n4_fisor_oracle_2026-X-X_sX_X/model3.pickle \
  --semantic_xy=True \
  --semantic_dual_panel=True \
  --semantic_agent_index=0 \
  --semantic_steps=8000 \
  --semantic_episodes=10 \
  --output_image_name viz_map_ctce_semantic_xy_dual_model3.png
```

*注意：`viz_map.py` 的 `--semantic_agent_index=0` 将自动选择 agent_0 视角的 cost value 进行可视化切片。*

> **警告**：如果在未调整或未获得 `costs_per_agent` 的原始数据集上使用 `fisor_oracle` 模式，那么训练将退回寻找 `cost_per_agent` 作为替代，若均未找到，框架将停止使用该维度特设机制，引发字段缺失错误。请务必检查你的 `.hdf5` 中含有该字段。
