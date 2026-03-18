"""
Offline 数据收集主脚本
用于收集多智能体环境的 offline 数据
"""

import argparse
import os
import sys
from typing import Any, Dict, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import gymnasium as gym

from envs.env_factory import make_marl_env
from data_collection import RandomPolicy, collect_offline_data
from migration.variable_agent_policy import VariableAgentPolicy


class VariableAgentCheckpointPolicy:
    """从 checkpoint 加载的可变智能体策略（用于在线策略 rollout 采集）。"""

    def __init__(
        self,
        env,
        checkpoint_path: str,
        device: str = 'cpu',
        hidden_dim: int = 128,
        max_agents: int = 10,
        noise_std: float = 0.05,
    ):
        if not checkpoint_path:
            raise ValueError('policy_type=variable_ckpt 时必须提供 --policy-checkpoint')

        first_agent = env.agents[0]
        obs_dim = int(env.observation_space(first_agent).shape[0])
        action_dim = int(env.action_space(first_agent).shape[0])

        self.device = device
        self.noise_std = noise_std
        self.model = VariableAgentPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            max_agents=max_agents,
        ).to(device)
        self.model.eval()

        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'policy' in checkpoint:
            state_dict = checkpoint['policy']
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict, strict=False)

    def get_actions(
        self,
        observations: Dict[str, 'np.ndarray'],
        deterministic: bool = False,
    ) -> Dict[str, 'np.ndarray']:
        import numpy as np

        actions = self.model.get_action(observations, deterministic=True)
        if not deterministic and self.noise_std > 0.0:
            for agent, action in actions.items():
                noise = np.random.normal(loc=0.0, scale=self.noise_std, size=action.shape)
                actions[agent] = np.clip(action + noise, -1.0, 1.0)
        return actions


class _SingleAgentSpaceEnv:
    """仅用于构建 FSRL Agent 的最小单智能体 env 包装。"""

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space


class FSRLCheckpointPolicy:
    """从 FSRL checkpoint 加载策略，并用于多智能体逐 agent rollout。"""

    def __init__(
        self,
        env,
        checkpoint_path: str,
        device: str = 'cpu',
        hidden_dim: int = 128,
        noise_std: float = 0.05,
        deterministic_default: bool = True,
        inference_mode: str = 'ctce',
    ):
        if not checkpoint_path:
            raise ValueError('policy_type=fsrl_ckpt 时必须提供 --policy-checkpoint')

        try:
            from fsrl.agent import PPOLagAgent
            from fsrl.utils import BaseLogger
            from fsrl.utils.exp_util import load_config_and_model
            from tianshou.data import Batch
        except Exception as exc:
            raise ImportError(
                '无法导入 FSRL 运行时依赖，请确认已在当前环境安装 fsrl/tianshou。'
            ) from exc

        import numpy as np

        self.device = device
        self.noise_std = float(noise_std)
        self.deterministic_default = bool(deterministic_default)
        self._Batch = Batch
        self._checkpoint_path = checkpoint_path
        self.inference_mode = inference_mode
        if self.inference_mode not in ('ctce', 'per_agent'):
            raise ValueError('inference_mode must be one of: ctce, per_agent')

        self.agent_names = list(getattr(env, 'possible_agents', env.agents))
        self._obs_shapes = {agent: env.observation_space(agent).shape for agent in self.agent_names}
        self._obs_dims = {agent: int(np.prod(env.observation_space(agent).shape)) for agent in self.agent_names}
        self._act_shapes = {agent: env.action_space(agent).shape for agent in self.agent_names}
        self._act_dims = {agent: int(np.prod(env.action_space(agent).shape)) for agent in self.agent_names}

        if self.inference_mode == 'ctce':
            obs_space, act_space = self._build_ctce_spaces(env)
        else:
            first_agent = env.agents[0]
            obs_space = env.observation_space(first_agent)
            act_space = env.action_space(first_agent)

        obs_dim = int(np.prod(obs_space.shape))
        act_dim = int(np.prod(act_space.shape))

        dummy_env = _SingleAgentSpaceEnv(obs_space, act_space)

        config: Dict[str, Any] = {}
        model_obj: Optional[Any] = None

        if os.path.isdir(checkpoint_path):
            config, model_obj = load_config_and_model(checkpoint_path, best=False)
        else:
            model_obj = torch.load(checkpoint_path, map_location=device)
            # 支持传入 logs/.../checkpoint/model.pt，自动回溯 config
            parent_dir = os.path.dirname(os.path.dirname(checkpoint_path))
            config_path = os.path.join(parent_dir, 'config.yaml')
            if os.path.isfile(config_path):
                import yaml

                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.load(f.read(), Loader=yaml.FullLoader) or {}

        hidden_sizes = tuple(config.get('hidden_sizes', (hidden_dim, hidden_dim)))

        self.agent = PPOLagAgent(
            env=dummy_env,
            logger=BaseLogger(),
            device=device,
            thread=int(config.get('thread', 1)),
            seed=int(config.get('seed', 10)),
            hidden_sizes=hidden_sizes,
            unbounded=bool(config.get('unbounded', False)),
            last_layer_scale=bool(config.get('last_layer_scale', False)),
            use_lagrangian=bool(config.get('use_lagrangian', True)),
            deterministic_eval=True,
        )

        if isinstance(model_obj, dict) and 'model' in model_obj:
            state_dict = model_obj['model']
        else:
            state_dict = model_obj
        try:
            self.agent.policy.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            sigma_shape = None
            obs_proj_shape = None
            if isinstance(state_dict, dict):
                if 'actor.sigma_param' in state_dict:
                    sigma_shape = tuple(state_dict['actor.sigma_param'].shape)
                if 'actor.preprocess.model.model.0.weight' in state_dict:
                    obs_proj_shape = tuple(state_dict['actor.preprocess.model.model.0.weight'].shape)

            raise ValueError(
                'FSRL checkpoint 与当前采样环境维度不兼容，无法直接加载。\n'
                f'- 当前环境: obs_dim={obs_dim}, act_dim={act_dim}\n'
                f'- checkpoint(actor.sigma_param): {sigma_shape}\n'
                f'- checkpoint(actor.preprocess.model.model.0.weight): {obs_proj_shape}\n\n'
                '建议：\n'
                '1) 使用与当前环境观测/动作维度一致的 checkpoint；\n'
                '2) 或改用 --policy-type variable_ckpt / random 先采样；\n'
                '3) 若坚持迁移，需要额外训练一个观测映射/动作头适配器。\n'
                f'checkpoint: {self._checkpoint_path}'
            ) from exc
        self.agent.policy.eval()

        # 使用动作空间映射参数
        self.agent.policy.action_scaling = bool(config.get('action_scaling', True))
        self.agent.policy.action_bound_method = str(config.get('action_bound_method', 'clip'))

        self._np = np

    def _get_single_action(self, observation, deterministic: bool = True):
        obs_batch = self._Batch(obs=self._np.expand_dims(observation, axis=0))

        self.agent.policy._deterministic_eval = bool(deterministic)
        with torch.no_grad():
            result = self.agent.policy.forward(obs_batch)

        act = result.act
        if hasattr(act, 'detach'):
            act = act.detach().cpu().numpy()
        if isinstance(act, self._np.ndarray) and act.ndim > 1:
            act = act[0]

        mapped = self.agent.policy.map_action(self._np.asarray(act, dtype=self._np.float32))
        return mapped

    def _build_ctce_spaces(self, env):
        obs_spaces = [env.observation_space(agent) for agent in self.agent_names]
        act_spaces = [env.action_space(agent) for agent in self.agent_names]

        obs_low = np.concatenate([np.asarray(space.low).reshape(-1) for space in obs_spaces], axis=0)
        obs_high = np.concatenate([np.asarray(space.high).reshape(-1) for space in obs_spaces], axis=0)
        act_low = np.concatenate([np.asarray(space.low).reshape(-1) for space in act_spaces], axis=0)
        act_high = np.concatenate([np.asarray(space.high).reshape(-1) for space in act_spaces], axis=0)

        obs_space = gym.spaces.Box(low=obs_low.astype(np.float32), high=obs_high.astype(np.float32), dtype=np.float32)
        act_space = gym.spaces.Box(low=act_low.astype(np.float32), high=act_high.astype(np.float32), dtype=np.float32)
        return obs_space, act_space

    def _aggregate_observation(self, observations: Dict[str, 'np.ndarray']) -> 'np.ndarray':
        obs_list = []
        for agent in self.agent_names:
            obs = observations.get(agent, None)
            if obs is None:
                obs = np.zeros((self._obs_dims[agent],), dtype=np.float32)
            obs_list.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        return np.concatenate(obs_list, axis=0)

    def _disaggregate_action(self, joint_action: 'np.ndarray') -> Dict[str, 'np.ndarray']:
        action_vec = np.asarray(joint_action, dtype=np.float32).reshape(-1)
        expected_dim = int(np.sum([self._act_dims[agent] for agent in self.agent_names]))
        if action_vec.shape[0] != expected_dim:
            raise ValueError(
                f'CTCE action dim mismatch, expected {expected_dim}, got {action_vec.shape[0]}. '
                f'Checkpoint: {self._checkpoint_path}'
            )

        actions: Dict[str, 'np.ndarray'] = {}
        offset = 0
        for agent in self.agent_names:
            dim = self._act_dims[agent]
            chunk = action_vec[offset: offset + dim]
            offset += dim
            actions[agent] = chunk.reshape(self._act_shapes[agent])
        return actions

    def get_actions(
        self,
        observations: Dict[str, 'np.ndarray'],
        deterministic: bool = False,
    ) -> Dict[str, 'np.ndarray']:
        deterministic_flag = self.deterministic_default if deterministic is False else True

        if self.inference_mode == 'ctce':
            joint_obs = self._aggregate_observation(observations)
            joint_action = self._get_single_action(joint_obs, deterministic=deterministic_flag)
            if (not deterministic_flag) and self.noise_std > 0.0:
                noise = self._np.random.normal(0.0, self.noise_std, size=joint_action.shape)
                joint_action = self._np.clip(joint_action + noise, -1.0, 1.0)
            return self._disaggregate_action(joint_action)

        actions: Dict[str, 'np.ndarray'] = {}
        for agent, obs in observations.items():
            action = self._get_single_action(obs, deterministic=deterministic_flag)
            if (not deterministic_flag) and self.noise_std > 0.0:
                noise = self._np.random.normal(0.0, self.noise_std, size=action.shape)
                action = self._np.clip(action + noise, -1.0, 1.0)
            actions[agent] = action
        return actions


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Collect offline data for MARL')
    
    # 环境参数
    parser.add_argument('--backend', type=str, default='safety_gym', choices=['safety_gym'],
                        help='Environment backend')
    parser.add_argument('--num-agents', type=int, default=2,
                        help='Number of agents in the environment')
    parser.add_argument('--env-id', type=str, default=None,
                        help='Environment ID (if not specified, will use num_agents)')
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
    parser.add_argument('--world-size', type=float, default=None,
                        help='World size')
    parser.add_argument('--num-hazards', type=int, default=None,
                        help='Number of hazards')
    
    # 数据收集参数
    parser.add_argument('--num-episodes', type=int, default=1000,
                        help='Number of episodes to collect')
    parser.add_argument('--max-episode-steps', type=int, default=None,
                        help='Maximum steps per episode (default: use env setting)')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic policy')
    parser.add_argument('--render', action='store_true',
                        help='Render environment')

    # 策略参数
    parser.add_argument('--policy-type', type=str, default='random',
                        choices=['random', 'variable_ckpt', 'fsrl_ckpt'],
                        help='Policy used for rollout collection')
    parser.add_argument('--policy-checkpoint', type=str, default=None,
                        help='Path to online policy checkpoint (.pt/.pth)')
    parser.add_argument('--policy-device', type=str, default='cpu',
                        help='Device for policy inference (cpu/cuda)')
    parser.add_argument('--policy-hidden-dim', type=int, default=128,
                        help='Hidden dim for VariableAgentPolicy when loading checkpoint')
    parser.add_argument('--policy-max-agents', type=int, default=10,
                        help='Max agents supported by loaded policy')
    parser.add_argument('--policy-noise-std', type=float, default=0.05,
                        help='Exploration noise std for stochastic rollout')
    parser.add_argument('--fsrl-hidden-dim', type=int, default=128,
                        help='Fallback hidden dim when FSRL config is unavailable')
    parser.add_argument('--fsrl-deterministic-default', action='store_true',
                        help='Use deterministic action by default for fsrl_ckpt')
    parser.add_argument(
        '--fsrl-inference-mode',
        type=str,
        default='ctce',
        choices=['ctce', 'per_agent'],
        help='ctce: concatenate all agents as super-agent for one forward pass; per_agent: run policy per agent',
    )
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='./offline_data',
                        help='Output directory for collected data')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='Save checkpoint every N episodes')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for computation')
    parser.add_argument('--torch-num-threads', type=int, default=1,
                        help='Torch CPU thread count for rollout process')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.torch_num_threads))
    
    # 创建环境
    if args.env_id:
        env_kwargs = {'randomize_layout': args.randomize_layout}
        if args.world_size is not None:
            env_kwargs['world_size'] = args.world_size
        if args.num_hazards is not None:
            env_kwargs['num_hazards'] = args.num_hazards
        env = make_marl_env(env_id=args.env_id, backend=args.backend, **env_kwargs)
    else:
        env_kwargs = {'randomize_layout': args.randomize_layout}
        if args.world_size is not None:
            env_kwargs['world_size'] = args.world_size
        if args.num_hazards is not None:
            env_kwargs['num_hazards'] = args.num_hazards
        
        env = make_marl_env(
            num_agents=args.num_agents,
            backend=args.backend,
            **env_kwargs
        )
    
    print("=" * 60)
    print("Environment Information")
    print("=" * 60)
    print(f"Backend: {args.backend}")
    print(f"Number of Agents: {len(env.agents)}")
    print(f"Agent Names: {env.agents}")
    print(f"World Size: {getattr(env, 'world_size', 'N/A')}")
    print(f"Number of Hazards: {getattr(env, 'num_hazards', 'N/A')}")
    print(f"Randomize Layout: {args.randomize_layout}")
    print(f"Max Episode Steps: {getattr(env, 'max_episode_steps', getattr(env, '_max_episode_steps', 'N/A'))}")
    print(f"Observation Space (per agent): {env.observation_space(env.agents[0])}")
    print(f"Action Space (per agent): {env.action_space(env.agents[0])}")
    print("=" * 60)
    
    # 创建策略（默认随机；可加载在线策略 checkpoint）
    if args.policy_type == 'random':
        policy = RandomPolicy(env.action_space(env.agents[0]))
    elif args.policy_type == 'variable_ckpt':
        policy = VariableAgentCheckpointPolicy(
            env=env,
            checkpoint_path=args.policy_checkpoint,
            device=args.policy_device,
            hidden_dim=args.policy_hidden_dim,
            max_agents=args.policy_max_agents,
            noise_std=args.policy_noise_std,
        )
    else:
        policy = FSRLCheckpointPolicy(
            env=env,
            checkpoint_path=args.policy_checkpoint,
            device=args.policy_device,
            hidden_dim=args.fsrl_hidden_dim,
            noise_std=args.policy_noise_std,
            deterministic_default=args.fsrl_deterministic_default,
            inference_mode=args.fsrl_inference_mode,
        )
    
    # 收集数据
    print("\nStarting data collection...")
    print(f"Policy Type: {args.policy_type}")
    if args.policy_type in ('variable_ckpt', 'fsrl_ckpt'):
        print(f"Policy Checkpoint: {args.policy_checkpoint}")
        print(f"Policy Device: {args.policy_device}")
    if args.policy_type == 'fsrl_ckpt':
        print(f"FSRL Inference Mode: {args.fsrl_inference_mode}")
    dataset = collect_offline_data(
        env=env,
        policy=policy,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
        deterministic=args.deterministic,
        render=args.render,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
        device=args.device,
    )
    
    # 打印统计信息
    print("\n")
    dataset.print_statistics()
    
    # 关闭环境
    env.close()
    
    print(f"\nData collection completed! Data saved to {args.output_dir}")


if __name__ == '__main__':
    main()