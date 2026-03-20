"""
环境工厂和注册模块
用于创建不同智能体数量的环境变体
"""

import re
from typing import Dict, Any, Optional, List
from envs.safety_gym_adapter import (
    SAFETY_GYM_AVAILABLE,
    SafetyGymAdapter,
    ensure_variable_multigoal_registered,
)


MIN_SUPPORTED_AGENTS = 2


def _is_supported_num_agents(num_agents: int) -> bool:
    return MIN_SUPPORTED_AGENTS <= num_agents


def _preset_from_num_agents(num_agents: int) -> str:
    if not _is_supported_num_agents(num_agents):
        raise ValueError(f'num_agents must be >= {MIN_SUPPORTED_AGENTS}, got {num_agents}')
    return f'sg_ant_goal_n{num_agents}'


def _extract_num_agents_from_preset(env_id: str) -> Optional[int]:
    match = re.fullmatch(r'sg_ant_goal_n(\d+)', env_id)
    if not match:
        return None
    return int(match.group(1))


def _extract_num_agents_from_registered_env(env_id: str) -> Optional[int]:
    match = re.fullmatch(r'SafetyAntMultiGoalN(\d+)-v0', env_id)
    if not match:
        return None
    return int(match.group(1))


def _resolve_requested_num_agents(env_id: Optional[str], num_agents: Optional[int]) -> Optional[int]:
    if num_agents is not None:
        return num_agents
    if env_id is None:
        return None
    parsed = _extract_num_agents_from_preset(env_id)
    if parsed is not None:
        return parsed
    return _extract_num_agents_from_registered_env(env_id)

SAFETY_GYM_CUSTOM_ONLY_KWARGS = {
    'randomize_layout',
    'num_hazards',
    'world_size',
    'num_gremlins',
    'agent_radius',
    'hazard_radius',
    'goal_radius',
    'agent_collision_penalty',
    'hazard_penalty',
    'goal_reward',
    'observe_other_agents',
    'observation_radius',
    'strict_mujoco_consistency',
}


def _apply_runtime_task_overrides(
    env: Any,
    randomize_layout: Optional[bool],
    num_hazards: Optional[int],
    world_size: Optional[float],
) -> Any:
    """Apply task-level overrides after env construction.

    These options are not part of Builder.__init__ kwargs in safety_gymnasium,
    so we patch task configs directly before first reset.
    """
    base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    task = getattr(base_env, 'task', None)
    if task is None:
        return env

    if randomize_layout is not None and hasattr(task, 'mechanism_conf'):
        task.mechanism_conf.randomize_layout = bool(randomize_layout)

    if world_size is not None and hasattr(task, 'placements_conf'):
        w = float(world_size)
        task.placements_conf.extents = [-w, -w, w, w]

    if num_hazards is not None and hasattr(task, 'hazards'):
        task.hazards.num = int(num_hazards)

    return env


def make_marl_env(
    env_id: Optional[str] = None,
    num_agents: Optional[int] = None,
    render_mode: Optional[str] = None,
    backend: str = 'safety_gym',
    **kwargs
) -> Any:
    """
    创建多智能体环境
    
    Args:
        env_id: 环境ID（可选）
        num_agents: 智能体数量
        render_mode: 渲染模式
        backend: 环境后端，支持 'safety_gym'
        **kwargs: 额外的环境参数

    Returns:
        Safety-Gymnasium 多智能体环境实例
    """
    if backend != 'safety_gym':
        raise ValueError("backend must be 'safety_gym'")

    return make_safety_gym_env(
        env_id=env_id,
        num_agents=num_agents,
        render_mode=render_mode,
        **kwargs,
    )


def create_env_family(
    base_num_agents: int = 2,
    max_num_agents: int = 12,
    shared_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    创建一个环境家族，包含从 base 到 max 的所有智能体数量配置
    
    用于迁移学习：在较少智能体环境训练，迁移到较多智能体环境
    
    Args:
        base_num_agents: 最小智能体数量
        max_num_agents: 最大智能体数量
        shared_config: 共享的配置参数
    
    Returns:
        环境字典 {env_id: env}
    """
    if shared_config is None:
        shared_config = {}
    
    env_family = {}
    
    for n in range(base_num_agents, max_num_agents + 1):
        env_id = f'Safety{n}AgentMultiGoal-v0'
        if not _is_supported_num_agents(n):
            continue
        env_family[env_id] = make_safety_gym_env(num_agents=n, **shared_config)
    
    return env_family


def get_safety_gym_train_envs() -> Dict[str, Dict[str, Any]]:
    """
    获取 Safety-Gymnasium 训练环境清单。

    Returns:
        字典 {train_env_name: {'description': str, 'env_type': str}}
    """
    if not SAFETY_GYM_AVAILABLE:
        return {}

    specs = SafetyGymAdapter.get_train_env_specs()
    return {
        name: {
            'description': spec.description,
            'env_type': spec.env_type,
        }
        for name, spec in specs.items()
    }


def make_safety_gym_env(
    env_id: Optional[str] = None,
    num_agents: Optional[int] = None,
    render_mode: Optional[str] = None,
    **kwargs,
):
    """
    创建 Safety-Gymnasium 训练环境。

    Args:
        env_id: 训练环境名（如 sg_ant_goal_n2）或真实注册环境 ID（如 SafetyAntMultiGoalN2-v0）
        num_agents: 智能体数量（>=2 时会自动映射到训练环境）
        render_mode: 渲染模式
        **kwargs: 传入 safety_gymnasium.make / make_ma 的参数

    Returns:
        Safety-Gymnasium 环境实例
    """
    if not SAFETY_GYM_AVAILABLE:
        raise ImportError('safety-gymnasium is not available in current environment')

    specs = SafetyGymAdapter.get_train_env_specs()
    randomize_layout = kwargs.get('randomize_layout', None)
    num_hazards = kwargs.get('num_hazards', None)
    world_size = kwargs.get('world_size', None)
    env_kwargs = {k: v for k, v in kwargs.items() if k not in SAFETY_GYM_CUSTOM_ONLY_KWARGS}
    strict_mujoco_consistency = kwargs.get('strict_mujoco_consistency', True)

    requested_num_agents = _resolve_requested_num_agents(env_id=env_id, num_agents=num_agents)

    if strict_mujoco_consistency:
        if requested_num_agents is None:
            requested_num_agents = 2

        import safety_gymnasium

        native_kwargs = dict(env_kwargs)
        native_kwargs.setdefault('camera_name', 'track')
        if render_mode is not None:
            native_kwargs['render_mode'] = render_mode
        native_env_id = f'SafetyAntMultiGoalN{requested_num_agents}-v0'
        ensure_variable_multigoal_registered(requested_num_agents)
        env = safety_gymnasium.make(native_env_id, **native_kwargs)
        return _apply_runtime_task_overrides(env, randomize_layout, num_hazards, world_size)

    if env_id is None and num_agents is not None:
        env_id = _preset_from_num_agents(num_agents)

    if env_id is not None:
        parsed_num_agents = _extract_num_agents_from_preset(env_id)
        if parsed_num_agents is not None:
            if not _is_supported_num_agents(parsed_num_agents):
                raise ValueError(f'Unsupported env_id={env_id}. Only sg_ant_goal_n{{N}} with N>=2 is supported.')
            env_id = f'sg_ant_goal_n{parsed_num_agents}'
            env = SafetyGymAdapter.make_train_env(env_id, render_mode=render_mode, **env_kwargs)
            return _apply_runtime_task_overrides(env, randomize_layout, num_hazards, world_size)

    if env_id in specs:
        env = SafetyGymAdapter.make_train_env(env_id, render_mode=render_mode, **env_kwargs)
        return _apply_runtime_task_overrides(env, randomize_layout, num_hazards, world_size)

    if env_id is not None and env_id.startswith('Safety'):
        direct_registered_n = _extract_num_agents_from_registered_env(env_id)
        if direct_registered_n is not None:
            ensure_variable_multigoal_registered(direct_registered_n)
        if render_mode is not None:
            env_kwargs['render_mode'] = render_mode
        import safety_gymnasium

        env = safety_gymnasium.make(env_id, **env_kwargs)
        return _apply_runtime_task_overrides(env, randomize_layout, num_hazards, world_size)

    if env_id is None:
        env = SafetyGymAdapter.make_train_env('sg_ant_goal_n2', render_mode=render_mode, **env_kwargs)
        return _apply_runtime_task_overrides(env, randomize_layout, num_hazards, world_size)

    raise ValueError(
        f'Unknown Safety-Gym environment: {env_id}. '
        f'Available train envs: {list(specs.keys())}'
    )


def create_safety_gym_training_suite(
    train_env_names: Optional[List[str]] = None,
    render_mode: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    批量创建 Safety-Gymnasium 训练环境。

    默认创建 3 个训练环境：
    - sg_ant_goal_n2 (2 agents)
    - sg_ant_goal_n3 (3 agents)
    - sg_ant_goal_n4 (4 agents)

    如需 N-agent(<=10)，可传入 train_env_names，
    例如 ['sg_ant_goal_n7', 'sg_ant_goal_n10']。
    """
    if train_env_names is None:
        train_env_names = [
            'sg_ant_goal_n2',
            'sg_ant_goal_n3',
            'sg_ant_goal_n4',
        ]

    suite = {}
    for train_env_name in train_env_names:
        suite[train_env_name] = SafetyGymAdapter.make_train_env(
            train_env_name,
            render_mode=render_mode,
            **kwargs,
        )
    return suite


class EnvironmentCurriculum:
    """
    环境课程学习管理器
    逐步增加智能体数量，实现渐进式学习
    """
    
    def __init__(
        self,
        base_num_agents: int = 2,
        max_num_agents: int = 10,
        success_threshold: float = 0.8,
        **env_kwargs
    ):
        """
        初始化课程学习管理器
        
        Args:
            base_num_agents: 起始智能体数量
            max_num_agents: 最大智能体数量
            success_threshold: 成功率阈值，达到后增加难度
            **env_kwargs: 环境参数
        """
        self.base_num_agents = base_num_agents
        self.max_num_agents = max_num_agents
        self.current_num_agents = base_num_agents
        self.success_threshold = success_threshold
        self.env_kwargs = env_kwargs
        
        self.current_env = None
        self.reset()
    
    def reset(self):
        """重置当前环境"""
        self.current_env = make_safety_gym_env(
            num_agents=self.current_num_agents,
            **self.env_kwargs,
        )
        return self.current_env
    
    def should_advance(self, success_rate: float) -> bool:
        """
        判断是否应该增加难度（智能体数量）
        
        Args:
            success_rate: 当前成功率
        
        Returns:
            是否应该增加难度
        """
        return (
            success_rate >= self.success_threshold and 
            self.current_num_agents < self.max_num_agents
        )
    
    def advance(self) -> Any:
        """
        增加难度，创建更多智能体的环境
        
        Returns:
            新的环境实例
        """
        if self.current_num_agents < self.max_num_agents:
            self.current_num_agents += 1
            print(f"Advancing curriculum to {self.current_num_agents} agents")
        
        return self.reset()
    
    def get_current_env(self) -> Any:
        """获取当前环境"""
        if self.current_env is None:
            self.reset()
        return self.current_env
    
    @property
    def is_max_difficulty(self) -> bool:
        """是否已达到最大难度"""
        return self.current_num_agents >= self.max_num_agents