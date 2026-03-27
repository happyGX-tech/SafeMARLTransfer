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
├── docs/                        # 项目结构、数据流程、FISOR 方案文档
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

### third_party/

- `FSRL/examples/customized/collect_dataset.py`：TRPOlag 训练与 CTCE 采集统一入口（支持 collect_only + checkpoint 采样）。
- `OSRL/examples/tools/export_ctce_hdf5_to_osrl.py`：HDF5 转 NPZ 导出工具。

## FISOR 放置建议

- 推荐将 FISOR clone 在 `third_party/FISOR/`。
- 建议保持“上游原仓 + 本地最小补丁”模式，避免改动扩散到 `envs/` 与数据管线主代码。

## 目录命名与位置建议

- `safety-gymnasium-main` 不要求必须移入 `ma_safe_migration/`；运行期核心依赖是已安装的 `safety_gymnasium` Python 包。
- 当前适配器已支持多位置本地源码兜底：`SRL/safety-gymnasium-main`、`third_party/safety-gymnasium-main`、`external/safety-gymnasium-main`，并支持环境变量 `SAFETY_GYM_LOCAL_PATH` 覆盖。
- `third_party/` 建议保留命名，不建议立即改名；它是行业常见第三方依赖目录语义，改名会增加文档、命令、脚本维护成本。
