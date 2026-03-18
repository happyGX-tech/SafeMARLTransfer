"""FSRL 单智能体训练包装器。

将多智能体环境包装为单智能体 Gymnasium 接口，便于使用 FSRL 的单智能体算法
训练一个可迁移到多智能体 rollout 的共享策略。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:
    import gymnasium as gym
except Exception as exc:
    raise ImportError('gymnasium is required for FSRLSingleAgentEnv') from exc

from envs.env_factory import make_marl_env


def _concat_box_spaces(spaces: List[gym.spaces.Box]) -> gym.spaces.Box:
    """Concatenate a list of Box spaces into one flattened Box space."""
    low = np.concatenate([np.asarray(space.low).reshape(-1) for space in spaces], axis=0)
    high = np.concatenate([np.asarray(space.high).reshape(-1) for space in spaces], axis=0)
    return gym.spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32)


class FSRLSingleAgentEnv(gym.Env):
    """把多智能体环境包装成单智能体接口。"""

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(
        self,
        num_agents: int,
        backend: str = 'safety_gym',
        env_id: Optional[str] = None,
        controlled_agent: str = 'agent_0',
        other_policy: str = 'random',
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.num_agents = int(num_agents)
        self.backend = backend
        self.env_id = env_id
        self.controlled_agent = controlled_agent
        self.other_policy = other_policy
        self.render_mode = render_mode
        self.env_kwargs = dict(kwargs)

        if self.env_id is not None:
            self.ma_env = make_marl_env(
                env_id=self.env_id,
                backend=self.backend,
                render_mode=self.render_mode,
                **self.env_kwargs,
            )
        else:
            self.ma_env = make_marl_env(
                num_agents=self.num_agents,
                backend=self.backend,
                render_mode=self.render_mode,
                **self.env_kwargs,
            )

        if self.controlled_agent not in self.ma_env.possible_agents:
            raise ValueError(
                f'controlled_agent={self.controlled_agent} not in possible_agents={self.ma_env.possible_agents}'
            )

        self.observation_space = self.ma_env.observation_space(self.controlled_agent)
        self.action_space = self.ma_env.action_space(self.controlled_agent)
        self._last_obs_dict: Dict[str, np.ndarray] = {}

    def _other_action(self, agent: str, obs: np.ndarray) -> np.ndarray:
        if self.other_policy == 'zero':
            return np.zeros(self.action_space.shape, dtype=np.float32)
        return self.ma_env.action_space(agent).sample()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs_dict, info = self.ma_env.reset(seed=seed)
        self._last_obs_dict = obs_dict

        obs = np.asarray(obs_dict[self.controlled_agent], dtype=np.float32)
        return obs, {}

    def step(self, action):
        actions = {}
        for agent in self.ma_env.agents:
            if agent == self.controlled_agent:
                actions[agent] = np.asarray(action, dtype=np.float32)
            else:
                obs = self._last_obs_dict.get(agent, None)
                if obs is None:
                    obs = np.zeros(self.ma_env.observation_space(agent).shape, dtype=np.float32)
                actions[agent] = self._other_action(agent, obs)

        next_obs, rewards, costs, terminations, truncations, infos = self.ma_env.step(actions)
        self._last_obs_dict = next_obs

        reward = float(rewards.get(self.controlled_agent, 0.0))
        cost = float(costs.get(self.controlled_agent, 0.0))

        terminated = bool(terminations.get(self.controlled_agent, False) or any(terminations.values()))
        truncated = bool(truncations.get(self.controlled_agent, False) or any(truncations.values()))

        if self.controlled_agent in next_obs:
            obs = np.asarray(next_obs[self.controlled_agent], dtype=np.float32)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {'cost': cost}
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.ma_env.render()

    def close(self):
        self.ma_env.close()


class FSRLCTCESingleAgentEnv(gym.Env):
    """CTCE 包装：将多智能体观测/动作线性拼接成一个 super-agent 接口。"""

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(
        self,
        num_agents: int,
        backend: str = 'safety_gym',
        env_id: Optional[str] = None,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.num_agents = int(num_agents)
        self.backend = backend
        self.env_id = env_id
        self.render_mode = render_mode
        self.env_kwargs = dict(kwargs)

        if self.env_id is not None:
            self.ma_env = make_marl_env(
                env_id=self.env_id,
                backend=self.backend,
                render_mode=self.render_mode,
                **self.env_kwargs,
            )
        else:
            self.ma_env = make_marl_env(
                num_agents=self.num_agents,
                backend=self.backend,
                render_mode=self.render_mode,
                **self.env_kwargs,
            )

        self.agent_names = list(getattr(self.ma_env, 'possible_agents', self.ma_env.agents))
        self._obs_dims = [
            int(np.prod(self.ma_env.observation_space(agent).shape))
            for agent in self.agent_names
        ]
        self._act_dims = [
            int(np.prod(self.ma_env.action_space(agent).shape))
            for agent in self.agent_names
        ]

        obs_spaces = [self.ma_env.observation_space(agent) for agent in self.agent_names]
        act_spaces = [self.ma_env.action_space(agent) for agent in self.agent_names]
        self.observation_space = _concat_box_spaces(obs_spaces)
        self.action_space = _concat_box_spaces(act_spaces)

    def _pack_observations(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        obs_list = []
        for agent in self.agent_names:
            if agent in obs_dict:
                obs = np.asarray(obs_dict[agent], dtype=np.float32).reshape(-1)
            else:
                obs = np.zeros((self._obs_dims[self.agent_names.index(agent)],), dtype=np.float32)
            obs_list.append(obs)
        return np.concatenate(obs_list, axis=0)

    def _unpack_actions(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        joint_action = np.asarray(action, dtype=np.float32).reshape(-1)
        expected_dim = int(np.sum(self._act_dims))
        if joint_action.shape[0] != expected_dim:
            raise ValueError(f'CTCE action dim mismatch: expected {expected_dim}, got {joint_action.shape[0]}')

        actions: Dict[str, np.ndarray] = {}
        offset = 0
        for idx, agent in enumerate(self.agent_names):
            dim = self._act_dims[idx]
            chunk = joint_action[offset: offset + dim]
            offset += dim
            actions[agent] = chunk.reshape(self.ma_env.action_space(agent).shape)
        return actions

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs_dict, info = self.ma_env.reset(seed=seed)
        joint_obs = self._pack_observations(obs_dict)
        return joint_obs, {}

    def step(self, action):
        actions = self._unpack_actions(action)
        next_obs, rewards, costs, terminations, truncations, infos = self.ma_env.step(actions)

        reward = float(np.sum([float(rewards.get(agent, 0.0)) for agent in self.agent_names]))
        cost = float(np.sum([float(costs.get(agent, 0.0)) for agent in self.agent_names]))
        terminated = bool(any(terminations.values()))
        truncated = bool(any(truncations.values()))

        joint_obs = self._pack_observations(next_obs)
        info = {
            'cost': cost,
            'reward_per_agent': rewards,
            'cost_per_agent': costs,
        }
        return joint_obs, reward, terminated, truncated, info

    def render(self):
        return self.ma_env.render()

    def close(self):
        self.ma_env.close()
