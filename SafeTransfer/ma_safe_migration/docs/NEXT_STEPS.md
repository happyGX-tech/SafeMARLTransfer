# 下一步推进清单（CTCE 方案，2026-03）

## 0. 本轮决策

当前采集方案统一为 CTCE（Centralized Training, Centralized Execution）：

1. 把所有 agent 的观测拼接为 super observation。
2. 把所有 agent 的动作拼接为 super action。
3. 使用 FSRL TRPOlag 在 super-agent 空间训练与采样。
4. FSRL 算法库保持原样，不修改 TRPOlag 内核源码。

补充定位：CTCE 仅用于采集阶段统一接口；最终训练目标是基于采集数据做值分解 Offline Safe RL。

## 1. 现状

当前代码主链已切换到 CTCE 单入口：

1. 训练端：`third_party/FSRL/examples/customized/collect_dataset.py` 使用 TRPOlag 进行 CTCE 训练。
2. 采样端：`third_party/FSRL/examples/customized/collect_dataset.py` 在 `collect_only=True` 时使用同一 TRPOlag 结构进行 CTCE 采样。
3. 导出端：`third_party/OSRL/examples/tools/export_ctce_hdf5_to_osrl.py` 统一导出 OSRL/DSRL 可消费 NPZ。

## 2. 阶段计划

### P0：定义 CTCE 接口与验收基线

1. 明确固定 agent 顺序（例如 agent_0...agent_{N-1}）。
2. 明确聚合规则：
   - super_obs = concat(obs_i)
   - super_act = concat(act_i)
   - reward = sum(reward_i)
   - cost = sum(cost_i)
   - done = any(done_i)
3. 写清维度与切片索引表，作为实现与调试标准。

验收：

1. 手工跑通 reset/step 并确认拼接与拆分一致。
2. 与原环境动作边界一致，无越界错误。

### P1：落地 CTCE 包装与 TRPOlag 单入口

1. 新增或改造环境包装器，使其输出 super observation / super action space。
2. 调整训练入口到 `third_party/FSRL/examples/customized/collect_dataset.py` 并固定 TRPOlag。
3. 训练与采样共用同一入口，避免模型结构漂移。

验收：

1. 能完成至少 1 个短训练 run。
2. 成功产出 checkpoint。

### P2：采样链路切换到 CTCE

1. 使用 `third_party/FSRL/examples/customized/collect_dataset.py` 作为 CTCE 采样主入口。
2. 使用 `third_party/OSRL/examples/tools/export_ctce_hdf5_to_osrl.py` 导出统一元信息，便于分析与复现。
3. 统一输出目录命名：`data/<N>agents/ctce_<policy_tag>_<date>_<seed>`。

验收：

1. 20~50 episode smoke 采样成功。
2. 200+ episode 采样无崩溃，统计文件完整。

### P3（并行）：DSRL/OSRL 数据规范化

1. 固化字段：observations, next_observations, actions, rewards, costs, terminals, timeouts。
2. 增加数据版本元信息：ctce_schema_version、policy_tag、checkpoint_id、seed、num_agents。
3. 提供最小转换脚本（本项目数据 -> DSRL/OSRL 兼容格式）。

验收：

1. 下游离线训练器可以直接读取数据。
2. 有一份可复现实验命令清单。

### P4（后续主线）：值分解 Offline Safe RL

1. 以 CTCE 采集数据构建值分解训练集（联合状态、局部价值分配所需字段）。
2. 先跑最小值分解基线，验证 reward/cost 分解口径一致性。
3. 输出离线评估对照：random 数据 vs fsrl_ckpt 数据。

验收：

1. 至少 1 个值分解离线算法可稳定训练。
2. 训练日志与评估指标可复现。

## 3. 不做事项（防止跑偏）

1. 本阶段不做 CTDE/多策略池/联合 credit assignment 复杂改造。
2. 本阶段不改 FSRL 算法内核源码。
3. 本阶段优先采集链路闭环与数据质量，不追求算法创新结论。

## 4. 风险与排查

1. 维度错配：训练与采样必须共享同一 CTCE 包装配置。
2. 动作切分错误：必须按固定顺序和切片表切回各 agent。
3. 回报口径漂移：sum/mean 口径必须在文档与代码里保持一致。
4. 历史文档冲突：若出现“单受控 agent”描述，统一以本文件为准。

## 5. 本轮新增进展（2026-03-20）

1. 已打通 DSRL `process_dataset.py` 对 CTCE 轨迹数据的处理流程。
2. 已验证可以直接输出 reward-cost 可视化：
   - `fig/ctce_n3-120_before_filter.png`
   - `fig/ctce_n3-120_after_filter.png`
3. 已验证可输出过滤后的离线数据：
   - `third_party/DSRL/processed/ctce_n3-120-1043.hdf5`

## 6. 下一阶段执行清单（offlineRLMARL 主线）

1. 数据版本冻结：
   - v1: 原始 `offline_dataset_ctce.hdf5`
   - v2: 过滤后 `ctce_n3-120-1043.hdf5`
2. 建立统一离线训练配置模板：
   - 固定随机种子、batch size、学习率、训练步数。
3. 启动最小离线值分解基线训练（offlineRLMARL）：
   - 记录 reward/cost 曲线与约束违反率。
4. 做 v1 vs v2 对比实验：
   - 比较收敛速度、最终 reward、最终 cost、稳定性。
5. 产出可复现实验清单：
   - 命令、配置、输入数据版本、输出路径、关键指标表。
