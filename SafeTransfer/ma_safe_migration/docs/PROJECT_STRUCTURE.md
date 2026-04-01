# 项目结构（2026-03）

本目录聚焦三层能力：

1. 多智能体环境（Safety-Gymnasium）
2. 基于 FSRL Collector 的 CTCE 离线数据采集
3. 值分解的 Offline Safe RL 训练对接（接入 DSRL/OSRL）

## 当前状态补充（2026-03-31）

- FISOR 训练入口为 `third_party/FISOR/launcher/examples/train_offline.py`，CTCE 可视化入口为 `third_party/FISOR/launcher/viz/viz_map.py`。
- `third_party/FISOR/results/sg_ant_goal_n4/` 下存在多次实验目录，但当前工作区内多数目录仅有 `config.json`，缺少 `model*.pickle` 与可视化图片产物，说明训练流程存在“未形成可复用产物”的风险。
- 当前 CTCE 训练常用配置来自 `configs/train_config.py:fisor`（`critic_type=hj`、`cost_scale=25`）。结合 CTCE 数据的成本分布，HJ 标签在 `jaxrl5/data/dsrl_datasets.py` 中会被二值化为 `{25, -1}`，容易出现可行边界退化（边界几乎不穿越 0）。
- CTCE 语义图 `viz_map.py --semantic_xy` 由 rollout 采样点三角剖分得到，图中大块区域是采样凸包内插值，不代表完整世界网格真值；请与 GT hazard overlay 联合解读，避免把可视化插值误差当作训练失败结论。

## 目录

```text
ma_safe_migration/
├── envs/                        # 环境工厂、后端适配、FSRL 单智能体包装
├── migration/                   # 可变智能体策略网络与迁移组件
├── docs/                        # 项目结构、数据流程、FISOR 方案文档
├── third_party/                 # FSRL / DSRL / FISOR / pyrallis（上游镜像）
│   ├── FISOR/                   # Offline Safe RL 主体（训练/评估/可视化）
│   │   ├── launcher/examples/   # train_offline.py / eval_offline.py
│   │   ├── launcher/viz/        # viz_map.py (CTCE value & semantic xy)
│   │   ├── configs/             # train_config.py（fisor/fisor_ctce_raw）
│   │   └── results/             # 训练输出目录（checkpoint、可视化图等）
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
- `FISOR/launcher/examples/train_offline.py`：Offline 训练入口，支持本地 HDF5 直接训练、CTCE 自动识别、周期 checkpoint 保存。
- `FISOR/launcher/viz/viz_map.py`：CTCE 可行域可视化（obs 切片 + 语义 x-y 双图），支持 GT hazard 叠加。
- `FISOR/jaxrl5/data/dsrl_datasets.py`：离线数据读取与 HJ/QC 标签构造逻辑（含本地 HDF5 ratio 采样）。

## FISOR 训练链路（建议）

1. 用 `FSRL/examples/customized/collect_dataset.py` 采集并固定 `num_agents` 的 CTCE 数据。
2. 用 `DSRL/scripts/process_dataset.py` 生成过滤后的 HDF5，并检查 reward-cost 覆盖。
3. 使用 `FISOR/launcher/examples/train_offline.py` 训练；优先确保 `env_name`、`dataset_kwargs.num_agents`、`dataset 文件名中的 ctce_nN` 三者一致。
4. 训练结束后确认 `results/<group>/<exp>/` 至少包含 `config.json` 与 `model*.pickle`。
5. 用 `FISOR/launcher/viz/viz_map.py` 做语义图和边界图；若边界异常，优先回查 critic_type、cost_scale 和数据集 cost 分布。

## 已知风险与排查优先级

1. 产物缺失风险（高）：实验目录只有 `config.json` 时，优先排查训练中断、环境崩溃、权限或磁盘问题。
2. HJ 标签退化风险（高）：当 CTCE 数据中 step cost 绝大多数大于 0，`hj + cost_scale=25` 可能使可行边界学习退化，建议对比 `fisor_ctce_raw(qc)` 配置并重新可视化。
3. 可视化解释偏差（中）：`semantic_xy` 图仅在 rollout 覆盖区域可信，未覆盖区域不应据此判断策略可行性。

## FISOR 放置建议

- 推荐将 FISOR clone 在 `third_party/FISOR/`。
- 建议保持“上游原仓 + 本地最小补丁”模式，避免改动扩散到 `envs/` 与数据管线主代码。

## 目录命名与位置建议

- `safety-gymnasium-main` 不要求必须移入 `ma_safe_migration/`；运行期核心依赖是已安装的 `safety_gymnasium` Python 包。
- 当前适配器已支持多位置本地源码兜底：`SRL/safety-gymnasium-main`、`third_party/safety-gymnasium-main`、`external/safety-gymnasium-main`，并支持环境变量 `SAFETY_GYM_LOCAL_PATH` 覆盖。
- `third_party/` 建议保留命名，不建议立即改名；它是行业常见第三方依赖目录语义，改名会增加文档、命令、脚本维护成本。
