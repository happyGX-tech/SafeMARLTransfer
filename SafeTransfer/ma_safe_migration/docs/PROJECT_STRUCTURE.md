# 项目结构（2026-03）

本目录聚焦三层能力：

1. 多智能体环境与后端适配（Safety-Gymnasium）
2. 在线策略驱动的离线数据采集（使用 FSRL 库）
3. 值分解的 Offline Safe RL 训练对接（规划接入 DSRL/OSRL）

## 顶层目录

```text
ma_safe_migration/
├── envs/                        # 环境工厂、后端适配、FSRL 单智能体包装
├── data_collection/             # 轨迹采集、数据集结构、导出格式
├── migration/                   # 可变智能体策略网络与迁移组件
├── experiments/                 # 训练/实验脚本（含 FSRL 行为策略训练）
├── docs/                        # 仅保留项目结构与下一步计划
├── third_party/                 # FSRL / DSRL / pyrallis（上游镜像）
├── run_data_collection.py       # 离线数据采集 CLI（random/variable_ckpt/fsrl_ckpt）
├── verify_safety_gym_backend.py # 后端与渲染回归入口
├── test_environment.py          # 环境测试
├── README.md
└── requirements.txt
```

## 模块职责

### envs/

- `env_factory.py`：统一环境入口 `make_marl_env`，默认走 safety_gym 主链。
- `safety_gym_adapter.py`：`sg_ant_goal_n{N}` 与 `SafetyAntMultiGoalN{N}-v0` 映射与注册。
- `fsrl_single_agent_wrapper.py`：把多智能体环境包装成 FSRL 可训练的单智能体接口。

### data_collection/

- `data_collector.py`：多智能体 rollout、轨迹记录、统计输出。
- `offline_dataset.py`：HDF5/Pickle 存储和集中式视图导出。

### experiments/

- `train_fsrl_behavior_on_safetransfer.py`：在 SafeTransfer 维度上训练 FSRL 行为策略。

## 当前主链路（现状）

```text
verify_safety_gym_backend.py
  -> envs.env_factory.make_marl_env
  -> envs.safety_gym_adapter / safety_gymnasium.make
  -> MuJoCo Builder reset/step/render

experiments/train_fsrl_behavior_on_safetransfer.py
  -> envs.fsrl_single_agent_wrapper.FSRLSingleAgentEnv
  -> fsrl.agent.PPOLagAgent
  -> third_party/FSRL/logs_local/.../checkpoint

run_data_collection.py --policy-type fsrl_ckpt
  -> FSRLCheckpointPolicy
  -> 多agent逐个前向 + 字典动作 step
  -> data_collection/offline_dataset.py
```

## 新方案主链路（目标）

```text
CTCE 聚合包装层（待新增）
  -> 拼接 all-agent observations 为 super observation
  -> FSRL 单智能体 PPO（不改算法源码）
  -> 输出 super action 并拆分回各 agent action
  -> data_collection/offline_dataset.py
  -> 值分解 Offline Safe RL 训练输入
```

## 已知架构边界

- 当前代码仍是“单受控 agent 训练 + 多agent逐个推理”。
- 新计划已切换为 CTCE super-agent 采集范式（见 `docs/NEXT_STEPS.md`）。
