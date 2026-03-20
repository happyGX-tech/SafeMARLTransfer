# FSRL-DSRL-OSRL 端到端流程

本流程固定使用你们当前的可变智能体 Safety-Gymnasium 环境，并将多智能体按 CTCE 方式拼接成 super-agent。

## 1. 环境验证

```bash
python verify_safety_gym_backend.py --num-agents 6 --steps 20 --render-mode rgb_array
```

## 2. 训练 TRPOlag 行为策略（CTCE）

```bash
python third_party/FSRL/examples/customized/collect_dataset.py --backend safety_gym --num_agents 6 --collect_only False --epoch 10 --step_per_epoch 5000 --training_num 2 --testing_num 1 --device cpu --name ctce_n6_train --logdir ./third_party/FSRL/logs_local --logger_backend tensorboard --vector_env auto --memory_guard auto
```

说明：
- 默认 hidden_sizes 为 `[64*num_agents, 64*num_agents]`。
- 可通过 `--hidden_sizes 512 512` 手工覆盖。

## 3. 采集离线数据（FSRL 原生 collect_dataset）

```bash
python third_party/FSRL/examples/customized/collect_dataset.py --backend safety_gym --num_agents 6 --collect_only True --num_episodes 200 --policy_checkpoint ./third_party/FSRL/logs_local/fast-safe-rl/ctce-n6-cost-5-100/ctce_n6_train/checkpoint/model.pt --output_dir ./data/6agents/ctce_collect --dataset_name offline_dataset_ctce.hdf5 --seed 11
```

输出：
- `offline_dataset_ctce.hdf5`

## 4. 导出 OSRL/DSRL 可读数据

```bash
python third_party/OSRL/examples/tools/export_ctce_hdf5_to_osrl.py --input-hdf5 ./data/6agents/ctce_collect/<run_tag>/offline_dataset_ctce.hdf5 --output-npz ./data/6agents/ctce_collect/<run_tag>/offline_dataset_osrl.npz --num-agents 6 --policy-tag fsrl_ctce --checkpoint-id ctce_n6_train --seed 11
```

输出：
- `offline_dataset_osrl.npz`
- `offline_dataset_osrl.meta.json`

## 5. 字段规范

导出的 NPZ 至少包含以下键：
- `observations`
- `next_observations`
- `actions`
- `rewards`
- `costs`
- `terminals`
- `timeouts`

## 6. 快速自检

1. `observations.shape[0] == actions.shape[0] == rewards.shape[0]`
2. `np.logical_or(terminals, timeouts)` 至少包含若干 episode 终止点
3. `meta.json` 中的 `num_agents`、`checkpoint_id`、`seed` 与命令参数一致

## 7. DSRL trajectory 处理与 reward-cost 可视化

```bash
python third_party/DSRL/scripts/process_dataset.py "./third_party/FSRL/logs_local/fast-safe-rl/ctce-n3-cost-5-100/<run_tag>/offline_dataset_ctce.hdf5" --output ctce_n3 --dir ./third_party/DSRL/processed --cmin 0 --cmax 120 --rmin -200 --rmax 200 --save
```

输出：
- `fig/ctce_n3-120_before_filter.png`
- `fig/ctce_n3-120_after_filter.png`
- `third_party/DSRL/processed/ctce_n3-120-<traj_num>.hdf5`

说明：
- before/after 两张图可直接查看 trajectory 在 reward-cost 空间的覆盖与筛选效果。
- 过滤后的 hdf5 可作为 offlineRLMARL 训练输入候选版本。
