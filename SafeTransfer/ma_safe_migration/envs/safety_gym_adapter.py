"""Safety-Gymnasium 真实环境适配器。"""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import Any, Dict, List, Optional


def _candidate_local_sg_paths() -> List[str]:
    """Return candidate local source paths for safety_gymnasium.

    Priority:
    1) SAFETY_GYM_LOCAL_PATH (explicit override)
    2) workspace_root/SRL/safety-gymnasium-main (historical layout)
    3) ma_safe_migration/third_party/safety-gymnasium-main (optional colocated layout)
    4) ma_safe_migration/external/safety-gymnasium-main (optional renamed vendor dir)
    """
    current_dir = os.path.dirname(__file__)
    workspace_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))

    env_override = os.environ.get('SAFETY_GYM_LOCAL_PATH')
    candidates = [
        env_override,
        os.path.join(workspace_root, 'SRL', 'safety-gymnasium-main'),
        os.path.join(project_root, 'third_party', 'safety-gymnasium-main'),
        os.path.join(project_root, 'external', 'safety-gymnasium-main'),
    ]

    return [path for path in candidates if path]


def _ensure_local_sg_on_syspath() -> None:
    for local_sg_path in _candidate_local_sg_paths():
        if os.path.isdir(local_sg_path):
            if local_sg_path in sys.path:
                sys.path.remove(local_sg_path)
            sys.path.insert(0, local_sg_path)
            break


# Always prioritize workspace safety-gymnasium source over site-packages.
_ensure_local_sg_on_syspath()

try:
    from safety_gymnasium.utils.registration import safe_registry
except ImportError:
    _ensure_local_sg_on_syspath()
    from safety_gymnasium.utils.registration import safe_registry


try:
    import safety_gymnasium

    SAFETY_GYM_AVAILABLE = True
except ImportError:
    safety_gymnasium = None
    SAFETY_GYM_AVAILABLE = False


@dataclass(frozen=True)
class SafetyGymTrainEnvSpec:
    """Safety-Gymnasium 训练环境规格。"""

    env_type: str  # 'registered' | 'make_ma'
    env_id: Optional[str] = None
    scenario: Optional[str] = None
    agent_conf: Optional[str] = None
    description: str = ''


MIN_SUPPORTED_AGENTS = 2
BASELINE_NUM_HAZARDS = 6
BASELINE_WORLD_SIZE = 3.0
BASELINE_MAX_EPISODE_STEPS = 1000


def _build_train_env_specs() -> Dict[str, SafetyGymTrainEnvSpec]:
    specs: Dict[str, SafetyGymTrainEnvSpec] = {}
    for n in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        specs[f'sg_ant_goal_n{n}'] = SafetyGymTrainEnvSpec(
            env_type='registered',
            env_id=f'SafetyAntMultiGoalN{n}-v0',
            description=f'Safety-Gymnasium Variable MultiGoal, {n} agents -> {n} goals',
        )
    return specs


SAFETY_GYM_TRAIN_ENVS: Dict[str, SafetyGymTrainEnvSpec] = _build_train_env_specs()


def _normalize_num_agents(num_agents: int) -> int:
    num_agents = int(num_agents)
    if num_agents < MIN_SUPPORTED_AGENTS:
        raise ValueError(f'num_agents must be >= {MIN_SUPPORTED_AGENTS}, got {num_agents}')
    return num_agents


def _preset_name(num_agents: int) -> str:
    return f'sg_ant_goal_n{num_agents}'


def _registered_env_id(num_agents: int) -> str:
    return f'SafetyAntMultiGoalN{num_agents}-v0'


def _build_spec_for_num_agents(num_agents: int) -> SafetyGymTrainEnvSpec:
    num_agents = _normalize_num_agents(num_agents)
    return SafetyGymTrainEnvSpec(
        env_type='registered',
        env_id=_registered_env_id(num_agents),
        description=f'Safety-Gymnasium Variable MultiGoal, {num_agents} agents -> {num_agents} goals',
    )


def ensure_variable_multigoal_registered(num_agents: int) -> str:
    num_agents = _normalize_num_agents(num_agents)
    env_id = _registered_env_id(num_agents)

    if env_id not in safe_registry:
        safety_gymnasium.register(
            id=env_id,
            entry_point='safety_gymnasium.tasks.safe_multi_agent.builder:Builder',
            kwargs={
                'config': {
                    'agent_name': 'Ant',
                    'task_name': 'MultiGoal1',
                    'num_agents': num_agents,
                },
                'task_id': env_id,
            },
            max_episode_steps=BASELINE_MAX_EPISODE_STEPS,
            disable_env_checker=True,
        )

    return env_id


class SafetyGymAdapter:
    """统一创建 Safety-Gymnasium 训练环境。"""

    @staticmethod
    def ensure_available():
        if not SAFETY_GYM_AVAILABLE:
            raise ImportError(
                'safety-gymnasium is not installed or not importable in current environment.'
            )

    @staticmethod
    def get_train_env_specs() -> Dict[str, SafetyGymTrainEnvSpec]:
        """返回可直接用于 SafeTransfer 的 Safety-Gym 训练环境配置。"""
        return dict(SAFETY_GYM_TRAIN_ENVS)

    @staticmethod
    def get_available_registered_multi_goal_envs() -> List[str]:
        """返回常用 MultiGoal 注册环境 ID。"""
        return [f'SafetyAntMultiGoalN{n}-v0' for n in [2, 3, 4, 5, 6, 7, 8, 9, 10]]

    @staticmethod
    def make_from_spec(
        spec: SafetyGymTrainEnvSpec,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ):
        """根据环境规格创建环境。"""
        SafetyGymAdapter.ensure_available()

        if spec.env_type == 'registered':
            env_kwargs = dict(kwargs)
            if render_mode is not None:
                env_kwargs['render_mode'] = render_mode
            return safety_gymnasium.make(spec.env_id, **env_kwargs)

        if spec.env_type == 'make_ma':
            env_kwargs = dict(kwargs)
            if render_mode is not None:
                env_kwargs['render_mode'] = render_mode
            return safety_gymnasium.make_ma(spec.scenario, spec.agent_conf, **env_kwargs)

        raise ValueError(f'Unsupported spec env_type: {spec.env_type}')

    @staticmethod
    def make_train_env(
        train_env_name: str,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ):
        """按预设名称创建训练环境。"""
        if train_env_name.startswith('sg_ant_goal_n'):
            try:
                num_agents = int(train_env_name.split('sg_ant_goal_n', 1)[1])
                ensure_variable_multigoal_registered(num_agents)
                if train_env_name not in SAFETY_GYM_TRAIN_ENVS:
                    SAFETY_GYM_TRAIN_ENVS[train_env_name] = _build_spec_for_num_agents(num_agents)
            except ValueError:
                pass

        if train_env_name not in SAFETY_GYM_TRAIN_ENVS:
            raise ValueError(
                f'Unknown Safety-Gym train env: {train_env_name}. '
                f'Available: {list(SAFETY_GYM_TRAIN_ENVS.keys())}'
            )

        return SafetyGymAdapter.make_from_spec(
            SAFETY_GYM_TRAIN_ENVS[train_env_name],
            render_mode=render_mode,
            **kwargs,
        )


def test_safety_gym_integration():
    """快速验证 Safety-Gymnasium 适配器。"""
    if not SAFETY_GYM_AVAILABLE:
        print('safety-gymnasium not installed, skipping integration test')
        return

    print('Testing Safety-Gymnasium Integration')
    print('=' * 50)

    for train_env_name in ['sg_ant_goal_n2', 'sg_ant_goal_n3', 'sg_ant_goal_n4']:
        print(f'\nCreating train env: {train_env_name}')
        env = SafetyGymAdapter.make_train_env(train_env_name)
        obs, _ = env.reset(seed=42)
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, costs, terminations, truncations, _ = env.step(actions)
        print(f'  Agents: {len(env.agents)} -> {env.agents}')
        print(f'  Reward keys: {list(rewards.keys())}')
        print(f'  Cost keys: {list(costs.keys())}')
        print(f'  Terminated: {terminations}')
        env.close()

    print('\n✓ Integration test passed')


if __name__ == '__main__':
    test_safety_gym_integration()