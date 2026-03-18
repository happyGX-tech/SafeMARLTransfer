"""
验证当前环境是否基于 Safety-Gymnasium 构建。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.env_factory import make_marl_env, get_safety_gym_train_envs


def parse_args():
    parser = argparse.ArgumentParser(description='Verify Safety-Gymnasium backend integration')
    parser.add_argument('--env-id', type=str, default=None, help='Train env name or Safety* env id')
    parser.add_argument('--num-agents', type=int, default=None, help='Use num_agents mapping when env-id is None')
    parser.add_argument('--render-mode', type=str, default='rgb_array', help='none/human/rgb_array')
    parser.add_argument('--steps', type=int, default=5, help='Rollout steps')
    parser.add_argument('--seed', type=int, default=None, help='Reset seed, default None means random each reset')
    layout_group = parser.add_mutually_exclusive_group()
    layout_group.add_argument(
        '--randomize-layout',
        dest='randomize_layout',
        action='store_true',
        help='Whether to randomize layout at reset',
    )
    layout_group.add_argument(
        '--no-randomize-layout',
        dest='randomize_layout',
        action='store_false',
        help='Disable layout randomization at reset',
    )
    parser.set_defaults(randomize_layout=True)
    parser.add_argument('--num-hazards', type=int, default=None, help='Override hazard count for domain randomization')
    parser.add_argument('--world-size', type=float, default=None, help='Override world size for domain randomization')
    return parser.parse_args()


def main():
    args = parse_args()

    render_mode = None if args.render_mode.lower() == 'none' else args.render_mode

    print('=' * 70)
    print('Verify Safety-Gymnasium Backend')
    print('=' * 70)

    print('\nAvailable training env presets:')
    for name, info in get_safety_gym_train_envs().items():
        print(f"  - {name}: {info['description']}")

    env = make_marl_env(
        env_id=args.env_id,
        num_agents=args.num_agents,
        backend='safety_gym',
        render_mode=render_mode,
        randomize_layout=args.randomize_layout,
        num_hazards=args.num_hazards,
        world_size=args.world_size,
    )

    base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    backend_module = type(base_env).__module__
    backend_class = type(base_env).__name__
    env_module = type(env).__module__
    env_class = type(env).__name__
    is_safety_backend = (
        'safety_gymnasium' in backend_module
        or 'safety_gymnasium' in env_module
        or env_class == 'SafeMAEnv'
    )

    print('\nBackend inspection:')
    print(f'  wrapper class: {env_class}')
    print(f'  wrapper module: {env_module}')
    print(f'  base class: {backend_class}')
    print(f'  base module: {backend_module}')
    print(f'  safety-gym backend detected: {is_safety_backend}')

    reset_seed = args.seed if args.randomize_layout else None
    obs, _ = env.reset(seed=reset_seed)
    print('\nRuntime inspection:')
    print(f'  agents: {env.agents}')
    print(f'  observation_space(agent_0): {env.observation_space(env.agents[0])}')
    print(f'  action_space(agent_0): {env.action_space(env.agents[0])}')
    print(f'  render_mode: {render_mode}')
    print(f'  randomize_layout: {args.randomize_layout}')
    print(f'  reset_seed: {reset_seed}')
    print(f'  num_hazards: {args.num_hazards}')
    print(f'  world_size: {args.world_size}')

    for step in range(args.steps):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, costs, terminations, truncations, _ = env.step(actions)
        if render_mode == 'human':
            env.render()
        print(f'  step {step + 1}: rewards={rewards}, costs={costs}')
        if any(terminations.values()) or any(truncations.values()):
            print('  episode finished early')
            break

    env.close()
    print('\nVerification completed.')


if __name__ == '__main__':
    main()
