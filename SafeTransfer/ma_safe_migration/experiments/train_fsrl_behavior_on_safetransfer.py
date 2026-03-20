"""在 SafeTransfer 环境上训练 FSRL 行为策略。

用途：生成与当前多智能体环境维度一致的 FSRL checkpoint，供
`third_party/FSRL/examples/customized/collect_dataset.py` 在 collect_only 模式直接采样。

Deprecated: 训练与采样主流程已统一到
`third_party/FSRL/examples/customized/collect_dataset.py`（TRPOlag 单入口）。
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tianshou.env import DummyVectorEnv, ShmemVectorEnv

from envs.fsrl_single_agent_wrapper import FSRLCTCESingleAgentEnv, FSRLSingleAgentEnv


def parse_args():
    parser = argparse.ArgumentParser(description='Train FSRL behavior policy on SafeTransfer env')

    parser.add_argument('--num-agents', type=int, default=3)
    parser.add_argument('--backend', type=str, default='safety_gym', choices=['safety_gym'])
    parser.add_argument('--env-id', type=str, default=None)
    parser.add_argument('--controlled-agent', type=str, default='agent_0')
    parser.add_argument('--other-policy', type=str, default='random', choices=['random', 'zero'])
    parser.add_argument(
        '--fsrl-train-mode',
        type=str,
        default='ctce',
        choices=['ctce', 'single_agent'],
        help='ctce: concat all agents as super-agent; single_agent: train only one controlled agent',
    )
    layout_group = parser.add_mutually_exclusive_group()
    layout_group.add_argument(
        '--randomize-layout',
        dest='randomize_layout',
        action='store_true',
        help='Whether to randomize object/goal layout at reset',
    )
    layout_group.add_argument(
        '--no-randomize-layout',
        dest='randomize_layout',
        action='store_false',
        help='Disable layout randomization at reset',
    )
    parser.set_defaults(randomize_layout=True)
    parser.add_argument('--num-hazards', type=int, default=None, help='Override hazard count')
    parser.add_argument('--world-size', type=float, default=None, help='Override world size')

    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--step-per-epoch', type=int, default=10000)
    parser.add_argument('--episode-per-collect', type=int, default=10)
    parser.add_argument('--repeat-per-collect', type=int, default=4)
    parser.add_argument('--training-num', type=int, default=2)
    parser.add_argument('--testing-num', type=int, default=1)
    parser.add_argument(
        '--vector-env',
        type=str,
        default='auto',
        choices=['auto', 'dummy', 'shmem'],
        help='Vector env backend. auto uses dummy on Windows to avoid MuJoCo subprocess OOM.',
    )
    parser.add_argument(
        '--memory-guard',
        type=str,
        default='auto',
        choices=['auto', 'on', 'off'],
        help='Auto memory protection for heavy MuJoCo env creation. auto: enabled on Windows safety_gym, off on Linux.',
    )
    parser.add_argument(
        '--max-parallel-envs',
        type=int,
        default=1,
        help='Cap training/testing vector env count when memory guard is active.',
    )
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--cost-limit', type=float, default=25.0)

    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--thread', type=int, default=1)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument(
        '--hidden-sizes',
        type=int,
        nargs='+',
        default=None,
        help='MLP hidden sizes. If omitted, use [64*num_agents, 64*num_agents].',
    )

    parser.add_argument('--logdir', type=str, default='./third_party/FSRL/logs_local')
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--save-interval', type=int, default=4)

    return parser.parse_args()


def resolve_hidden_sizes(num_agents: int, hidden_sizes):
    if hidden_sizes:
        return tuple(int(x) for x in hidden_sizes)
    base = 64 * int(num_agents)
    return (base, base)


def _make_single_env(args, seed_offset=0):
    def _thunk():
        env_kwargs = {
            'randomize_layout': args.randomize_layout,
        }
        if args.num_hazards is not None:
            env_kwargs['num_hazards'] = args.num_hazards
        if args.world_size is not None:
            env_kwargs['world_size'] = args.world_size

        if args.fsrl_train_mode == 'ctce':
            env = FSRLCTCESingleAgentEnv(
                num_agents=args.num_agents,
                backend=args.backend,
                env_id=args.env_id,
                **env_kwargs,
            )
        else:
            env = FSRLSingleAgentEnv(
                num_agents=args.num_agents,
                backend=args.backend,
                env_id=args.env_id,
                controlled_agent=args.controlled_agent,
                other_policy=args.other_policy,
                **env_kwargs,
            )
        return env

    return _thunk


def _build_vector_env(args, env_fns):
    backend = args.vector_env
    if backend == 'auto':
        backend = 'dummy' if os.name == 'nt' else 'shmem'
    if backend == 'dummy':
        return DummyVectorEnv(env_fns), 'dummy'
    return ShmemVectorEnv(env_fns), 'shmem'


def main():
    args = parse_args()
    guard_active = False
    if args.memory_guard == 'on':
        guard_active = True
    elif args.memory_guard == 'auto':
        guard_active = (os.name == 'nt' and args.backend == 'safety_gym')

    if guard_active:
        limit = max(1, int(args.max_parallel_envs))
        if args.training_num > limit:
            print(
                f'[Memory Guard] training_num {args.training_num} -> {limit} '
                f'(platform={os.name}, backend={args.backend}, num_agents={args.num_agents})'
            )
            args.training_num = limit
        if args.testing_num > limit:
            print(
                f'[Memory Guard] testing_num {args.testing_num} -> {limit} '
                f'(platform={os.name}, backend={args.backend}, num_agents={args.num_agents})'
            )
            args.testing_num = limit

    hidden_sizes = resolve_hidden_sizes(args.num_agents, args.hidden_sizes)

    try:
        from fsrl.agent import PPOLagAgent
        from fsrl.utils import TensorboardLogger
    except Exception as exc:
        raise ImportError('Please install FSRL in current environment before running this script.') from exc

    run_name = args.run_name or f'fsrl_safetransfer_n{args.num_agents}_{datetime.now().strftime("%m%d_%H%M%S")}'
    run_log_dir = os.path.join(args.logdir, run_name)
    logger = TensorboardLogger(run_log_dir, log_txt=True, name=run_name)

    demo_env_kwargs = {
        'randomize_layout': args.randomize_layout,
    }
    if args.num_hazards is not None:
        demo_env_kwargs['num_hazards'] = args.num_hazards
    if args.world_size is not None:
        demo_env_kwargs['world_size'] = args.world_size

    if args.fsrl_train_mode == 'ctce':
        demo_env = FSRLCTCESingleAgentEnv(
            num_agents=args.num_agents,
            backend=args.backend,
            env_id=args.env_id,
            **demo_env_kwargs,
        )
    else:
        demo_env = FSRLSingleAgentEnv(
            num_agents=args.num_agents,
            backend=args.backend,
            env_id=args.env_id,
            controlled_agent=args.controlled_agent,
            other_policy=args.other_policy,
            **demo_env_kwargs,
        )

    agent = PPOLagAgent(
        env=demo_env,
        logger=logger,
        cost_limit=args.cost_limit,
        device=args.device,
        thread=args.thread,
        seed=args.seed,
        lr=args.lr,
        hidden_sizes=hidden_sizes,
        deterministic_eval=True,
    )
    # PPOLagAgent only needs env spaces here; release the heavy MuJoCo env early.
    demo_env.close()

    train_env_fns = [_make_single_env(args, seed_offset=i) for i in range(max(1, args.training_num))]
    test_env_fns = [_make_single_env(args, seed_offset=1000 + i) for i in range(max(1, args.testing_num))]
    train_envs, train_backend = _build_vector_env(args, train_env_fns)
    test_envs, test_backend = _build_vector_env(args, test_env_fns)

    print('=' * 70)
    print('FSRL Behavior Policy Training on SafeTransfer')
    print('=' * 70)
    print(f'run_name: {run_name}')
    print(f'log_dir: {run_log_dir}')
    print(f'vector_env(train/test): {train_backend}/{test_backend}')
    print(f'memory_guard: {args.memory_guard} (active={guard_active}, max_parallel_envs={args.max_parallel_envs})')
    print(f'fsrl_train_mode: {args.fsrl_train_mode}')
    print(f'num_agents: {args.num_agents}, controlled_agent: {args.controlled_agent}')
    print(f'hidden_sizes: {hidden_sizes}')
    print(f'other_policy: {args.other_policy}')
    print(f'randomize_layout: {args.randomize_layout}')
    print(f'num_hazards: {args.num_hazards}')
    print(f'world_size: {args.world_size}')
    print('=' * 70)

    try:
        agent.learn(
            train_envs=train_envs,
            test_envs=test_envs,
            epoch=args.epoch,
            episode_per_collect=args.episode_per_collect,
            step_per_epoch=args.step_per_epoch,
            repeat_per_collect=args.repeat_per_collect,
            batch_size=args.batch_size,
            testing_num=args.testing_num,
            save_interval=args.save_interval,
            save_ckpt=True,
            verbose=True,
        )
    finally:
        train_envs.close()
        test_envs.close()

    print('\nTraining finished.')
    print(f'Checkpoint directory: {os.path.join(run_log_dir, "checkpoint")}')


if __name__ == '__main__':
    main()
