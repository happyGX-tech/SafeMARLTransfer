# 项目结构（2026-03）

本目录聚焦三层能力：

1. 多智能体环境（Safety-Gymnasium）
2. 基于 FSRL Collector 的 CTCE 离线数据采集
3. 值分解的 Offline Safe RL 训练对接（接入 DSRL/OSRL）

## 目录

```text
ma_safe_migration/
├── envs/                        # 环境工厂、后端适配、FSRL 单智能体包装
├── migration/                   # 可变智能体策略网络与迁移组件
├── experiments/                 # 历史实验脚本（逐步下线）
├── docs/                        # 仅保留项目结构与下一步计划
├── third_party/                 # FSRL / DSRL / pyrallis（上游镜像）
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

### experiments/

- `train_fsrl_behavior_on_safetransfer.py`：历史训练入口（PPOLag），不再作为主流程（先）。

### third_party/

- `FSRL/examples/customized/collect_dataset.py`：TRPOlag 训练与 CTCE 采集统一入口（支持 collect_only + checkpoint 采样）。
- `OSRL/examples/tools/export_ctce_hdf5_to_osrl.py`：HDF5 转 NPZ 导出工具。
