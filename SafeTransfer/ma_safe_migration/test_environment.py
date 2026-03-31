"""
一键环境测试脚本（Safety-Gymnasium）
用于验证当前项目环境创建、step、渲染和数据兼容性。
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.env_factory import (
    make_marl_env,
    create_safety_gym_training_suite,
    get_safety_gym_train_envs,
)


def sample_actions(env, obs):
    agent_names = list(getattr(env, 'possible_agents', obs.keys()))
    return {agent: env.action_space(agent).sample() for agent in agent_names}


def test_backend_identity():
    print('=' * 60)
    print('Test 1: Backend Identity')
    print('=' * 60)

    env = make_marl_env(env_id='sg_ant_goal_n2', backend='safety_gym')
    base_env = env.unwrapped if hasattr(env, 'unwrapped') else env

    module_name = type(base_env).__module__
    print(f'Base env class: {type(base_env).__name__}')
    print(f'Base env module: {module_name}')

    assert 'safety_gymnasium' in module_name, 'Environment is not built from safety_gymnasium module'

    env.close()
    print('\nTest 1 PASSED!\n')


def test_training_presets():
    print('=' * 60)
    print('Test 2: Training Presets Availability')
    print('=' * 60)

    envs = get_safety_gym_train_envs()
    assert len(envs) >= 3, 'Expected at least baseline presets'
    assert 'sg_ant_goal_n2' in envs
    assert 'sg_ant_goal_n4' in envs
    assert 'sg_ant_goal_n10' in envs

    for env_name, info in envs.items():
        print(f"  - {env_name}: {info['description']}")

    print('\nTest 2 PASSED!\n')


def test_variable_agent_suite():
    print('=' * 60)
    print('Test 3: 2/3/4 Agent Suite Build')
    print('=' * 60)

    suite = create_safety_gym_training_suite()
    expected_counts = {
        'sg_ant_goal_n2': 2,
        'sg_ant_goal_n3': 3,
        'sg_ant_goal_n4': 4,
    }

    for env_name, env in suite.items():
        obs, _ = env.reset(seed=42)
        actions = sample_actions(env, obs)
        obs, rewards, costs, terminations, truncations, infos = env.step(actions)

        print(f'  {env_name}:')
        print(f'    agents={len(env.agents)}, names={env.agents}')
        print(f'    obs_shape(agent_0)={obs[env.agents[0]].shape}')
        print(f'    rewards_keys={list(rewards.keys())}')
        print(f'    costs_keys={list(costs.keys())}')

        assert len(env.agents) == expected_counts[env_name]
        assert len(rewards) == len(env.agents)
        assert len(costs) == len(env.agents)

    for env in suite.values():
        env.close()

    print('\nTest 3 PASSED!\n')


def test_high_n_agent_creation():
    print('=' * 60)
    print('Test 3.1: High-N Agent Creation (N=10)')
    print('=' * 60)

    env = make_marl_env(num_agents=12, backend='safety_gym')
    obs, _ = env.reset(seed=42)
    actions = sample_actions(env, obs)
    obs, rewards, costs, terminations, truncations, infos = env.step(actions)

    print(f'  agents={len(env.agents)}')
    assert len(env.agents) == 12
    assert len(rewards) == 12
    assert len(costs) == 12

    env.close()
    print('\nTest 3.1 PASSED!\n')


def test_rgb_render():
    print('=' * 60)
    print('Test 4: RGB Render')
    print('=' * 60)

    env = make_marl_env(env_id='sg_ant_goal_n2', backend='safety_gym', render_mode='rgb_array')
    obs, _ = env.reset(seed=42)

    for _ in range(3):
        actions = sample_actions(env, obs)
        obs, rewards, costs, terminations, truncations, infos = env.step(actions)

    frame = env.render()
    assert frame is not None, 'Render returned None'
    assert len(frame.shape) == 3, 'Expected image frame'

    print(f'  frame shape: {frame.shape}')
    print(f'  frame dtype: {frame.dtype}')
    print(f'  frame range: [{np.min(frame)}, {np.max(frame)}]')

    env.close()
    print('\nTest 4 PASSED!\n')


def test_human_mode_creation(skip_human: bool):
    print('=' * 60)
    print('Test 5: Human Render Mode Creation')
    print('=' * 60)

    if skip_human:
        print('  Skipped human render by argument')
        print('\nTest 5 SKIPPED!\n')
        return

    env = make_marl_env(env_id='sg_ant_goal_n2', backend='safety_gym', render_mode='human')
    obs, _ = env.reset(seed=42)
    actions = sample_actions(env, obs)
    obs, rewards, costs, terminations, truncations, infos = env.step(actions)

    print('  human mode env created and stepped successfully')
    env.close()
    print('\nTest 5 PASSED!\n')


def run_all_tests(skip_human: bool = False):
    print('\n' + '=' * 60)
    print('Running Safety-Gym Backend Environment Tests')
    print('=' * 60 + '\n')

    tests = [
        test_backend_identity,
        test_training_presets,
        test_variable_agent_suite,
        test_high_n_agent_creation,
        test_rgb_render,
        lambda: test_human_mode_creation(skip_human),
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f'\n❌ Test FAILED: {e}\n')
            failed += 1

    print('=' * 60)
    print(f'Test Results: {passed} passed, {failed} failed')
    print('=' * 60)

    return failed == 0


def parse_args():
    parser = argparse.ArgumentParser(description='Safety-Gym backend environment tests')
    parser.add_argument('--skip-human', action='store_true', help='Skip human render mode test')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    success = run_all_tests(skip_human=args.skip_human)
    sys.exit(0 if success else 1)
