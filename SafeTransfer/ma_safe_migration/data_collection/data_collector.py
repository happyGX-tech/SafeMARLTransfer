"""
Offline 数据收集器
参考 OSRL 的数据收集方式，使用 online 算法收集 offline 数据
支持 centralized 数据收集
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any, Tuple
from pathlib import Path
import time

from data_collection.offline_dataset import MAOfflineDataset, MATrajectory


class RandomPolicy:
    """随机策略，用于基线数据收集"""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def get_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """获取动作"""
        return self.action_space.sample()
    
    def get_actions(
        self,
        observations: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> Dict[str, np.ndarray]:
        """批量获取动作"""
        return {
            agent: self.action_space.sample()
            for agent, obs in observations.items()
        }


class CentralizedPolicy:
    """
    Centralized 策略包装器
    将 decentralized 观测聚合为 centralized 状态，输出联合动作
    """
    
    def __init__(
        self,
        policy_network: nn.Module,
        agent_names: List[str],
        device: str = 'cpu'
    ):
        """
        初始化 centralized 策略
        
        Args:
            policy_network: 策略网络，输入 centralized 状态，输出联合动作
            agent_names: 智能体名称列表
            device: 计算设备
        """
        self.policy_network = policy_network.to(device)
        self.agent_names = agent_names
        self.device = device
    
    def get_actions(
        self, 
        observations: Dict[str, np.ndarray],
        deterministic: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        获取联合动作
        
        Args:
            observations: 每个智能体的观测
            deterministic: 是否确定性采样
        
        Returns:
            每个智能体的动作
        """
        # 将 decentralized 观测聚合为 centralized 状态
        centralized_obs = self._aggregate_observations(observations)
        
        # 转换为 tensor
        obs_tensor = torch.FloatTensor(centralized_obs).unsqueeze(0).to(self.device)
        
        # 策略网络前向传播
        with torch.no_grad():
            action_dist = self.policy_network(obs_tensor)
            
            if deterministic:
                actions = action_dist.mean.cpu().numpy()[0]
            else:
                actions = action_dist.sample().cpu().numpy()[0]
        
        # 将联合动作分解为每个智能体的动作
        return self._disaggregate_actions(actions)
    
    def _aggregate_observations(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        """
        将多个智能体的观测聚合成 centralized 状态
        
        Args:
            observations: {agent: obs}
        
        Returns:
            拼接后的 centralized 观测
        """
        # 按固定顺序拼接观测
        obs_list = [observations[agent] for agent in self.agent_names]
        return np.concatenate(obs_list)
    
    def _disaggregate_actions(self, actions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        将联合动作分解为每个智能体的动作
        
        Args:
            actions: 联合动作数组
        
        Returns:
            {agent: action}
        """
        action_dim = actions.shape[0] // len(self.agent_names)
        return {
            agent: actions[i * action_dim:(i + 1) * action_dim]
            for i, agent in enumerate(self.agent_names)
        }


class OfflineDataCollector:
    """
    Offline 数据收集器
    使用 online 策略收集 offline 数据
    """
    
    def __init__(
        self,
        env,
        policy: Optional[Any] = None,
        num_episodes: int = 1000,
        max_episode_steps: Optional[int] = None,
        deterministic: bool = False,
        render: bool = False,
        save_interval: int = 100,
        output_dir: str = './offline_data',
        centralized: bool = True,
        device: str = 'cpu',
    ):
        """
        初始化数据收集器
        
        Args:
            env: 多智能体环境
            policy: 数据收集策略（如果为 None 则使用随机策略）
            num_episodes: 收集的轨迹数量
            max_episode_steps: 每回合最大步数
            deterministic: 是否确定性采样
            render: 是否渲染
            save_interval: 保存间隔（每多少回合保存一次）
            output_dir: 输出目录
            centralized: 是否使用 centralized 数据收集
            device: 计算设备
        """
        self.env = env
        self.policy = policy or RandomPolicy(env.action_space(env.agents[0]))
        self.agent_names = list(getattr(env, 'possible_agents', env.agents))
        self.num_episodes = num_episodes
        inferred_max_steps = getattr(env, 'max_episode_steps', None)
        if inferred_max_steps is None:
            inferred_max_steps = getattr(env, '_max_episode_steps', None)
        if inferred_max_steps is None and hasattr(env, 'spec') and env.spec is not None:
            inferred_max_steps = getattr(env.spec, 'max_episode_steps', None)
        self.max_episode_steps = max_episode_steps or inferred_max_steps or 1000
        self.deterministic = deterministic
        self.render = render
        self.save_interval = save_interval
        self.output_dir = Path(output_dir)
        self.centralized = centralized
        self.device = device
        
        # 创建数据集
        self.dataset = MAOfflineDataset(num_agents=len(self.agent_names))
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        self.collected_episodes = 0
        self.collected_steps = 0
        self.start_time = None
    
    def collect(self) -> MAOfflineDataset:
        """
        执行数据收集
        
        Returns:
            收集的数据集
        """
        self.start_time = time.time()
        
        print(f"Starting data collection: {self.num_episodes} episodes")
        print(f"Centralized mode: {self.centralized}")
        print(f"Max episode steps: {self.max_episode_steps}")
        print("=" * 50)
        
        for episode in range(self.num_episodes):
            self._collect_episode(episode)
            
            # 定期保存
            if (episode + 1) % self.save_interval == 0:
                self._save_checkpoint(episode + 1)
                self._print_progress(episode + 1)
        
        # 最终保存
        self._save_final_dataset()
        
        elapsed_time = time.time() - self.start_time
        print(f"\nData collection completed!")
        print(f"Total time: {elapsed_time:.2f}s")
        print(f"Episodes: {self.collected_episodes}")
        print(f"Steps: {self.collected_steps}")
        print(f"Avg steps/episode: {self.collected_steps / max(1, self.collected_episodes):.2f}")
        
        return self.dataset
    
    def _collect_episode(self, episode_id: int):
        """收集一条轨迹"""
        obs, info = self.env.reset(seed=episode_id)
        
        trajectory = MATrajectory(trajectory_id=episode_id, num_agents=len(self.agent_names))
        
        done = False
        step_count = 0
        episode_reward = {agent: 0.0 for agent in self.agent_names}
        episode_cost = {agent: 0.0 for agent in self.agent_names}
        
        while not done and step_count < self.max_episode_steps:
            # 获取动作
            current_agents = list(obs.keys())

            if hasattr(self.policy, 'get_actions'):
                actions = self.policy.get_actions(obs, deterministic=self.deterministic)
            else:
                # 逐个获取动作
                actions = {
                    agent: self.policy.get_action(obs[agent], deterministic=self.deterministic)
                    for agent in current_agents
                }

            # 补齐可能缺失的动作键（部分环境 step 期望 possible_agents）
            for agent in self.agent_names:
                if agent not in actions:
                    actions[agent] = self.env.action_space(agent).sample()
            
            # 执行动作
            next_obs, rewards, costs, terminations, truncations, infos = self.env.step(actions)
            
            # 记录数据
            terminal = any(terminations.values())
            timeout = any(truncations.values())
            
            trajectory.add_step(
                obs=obs,
                action=actions,
                reward=rewards,
                cost=costs,
                terminal=terminal,
                timeout=timeout,
                info=infos
            )
            
            # 累计奖励和成本
            for agent in self.agent_names:
                if agent in rewards:
                    episode_reward[agent] += rewards[agent]
                if agent in costs:
                    episode_cost[agent] += costs[agent]
            
            # 渲染
            if self.render:
                self.env.render()
            
            # 更新状态
            obs = next_obs
            step_count += 1
            done = terminal or timeout
        
        # 完成轨迹
        trajectory.finalize()
        self.dataset.add_trajectory(trajectory)
        
        self.collected_episodes += 1
        self.collected_steps += step_count
        
        # 打印轨迹信息
        if (episode_id + 1) % 10 == 0:
            avg_reward = np.mean([episode_reward[agent] for agent in self.agent_names])
            avg_cost = np.mean([episode_cost[agent] for agent in self.agent_names])
            print(f"Episode {episode_id + 1}: Steps={step_count}, "
                  f"Avg Reward={avg_reward:.4f}, Avg Cost={avg_cost:.4f}")
    
    def _save_checkpoint(self, episode: int):
        """保存检查点"""
        checkpoint_path = self.output_dir / f'checkpoint_{episode}_episodes.hdf5'
        self.dataset.save_to_hdf5(str(checkpoint_path))
    
    def _save_final_dataset(self):
        """保存最终数据集"""
        final_path = self.output_dir / 'offline_dataset_final.hdf5'
        self.dataset.save_to_hdf5(str(final_path))
        
        # 同时保存为 pickle（便于快速加载）
        pickle_path = self.output_dir / 'offline_dataset_final.pkl'
        self.dataset.save_to_pickle(str(pickle_path))
        
        # 保存统计信息
        stats_path = self.output_dir / 'dataset_stats.txt'
        with open(stats_path, 'w') as f:
            stats = self.dataset.get_statistics()
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
    
    def _print_progress(self, episode: int):
        """打印进度"""
        elapsed = time.time() - self.start_time
        eps_per_sec = episode / elapsed
        remaining_eps = self.num_episodes - episode
        eta = remaining_eps / eps_per_sec if eps_per_sec > 0 else 0
        
        print(f"\nProgress: {episode}/{self.num_episodes} episodes")
        print(f"Speed: {eps_per_sec:.2f} eps/s")
        print(f"ETA: {eta/60:.2f} minutes")
        print(f"Total steps collected: {self.collected_steps}")


class ParallelDataCollector:
    """
    并行数据收集器（使用多进程加速数据收集）
    注意：由于环境限制，这里提供接口设计，实际实现需要根据具体环境调整
    """
    
    def __init__(
        self,
        env_fn: Callable,
        policy_fn: Callable,
        num_workers: int = 4,
        episodes_per_worker: int = 250,
        **collector_kwargs
    ):
        """
        初始化并行数据收集器
        
        Args:
            env_fn: 环境创建函数
            policy_fn: 策略创建函数
            num_workers: 并行工作进程数
            episodes_per_worker: 每个工作进程收集的轨迹数
            **collector_kwargs: 其他收集器参数
        """
        self.env_fn = env_fn
        self.policy_fn = policy_fn
        self.num_workers = num_workers
        self.episodes_per_worker = episodes_per_worker
        self.collector_kwargs = collector_kwargs
    
    def collect(self) -> MAOfflineDataset:
        """
        并行收集数据
        
        Returns:
            合并后的数据集
        """
        # 注意：这里提供接口设计
        # 实际实现需要使用 multiprocessing 或 ray 等并行框架
        
        print(f"Parallel data collection with {self.num_workers} workers")
        print("Note: This is a placeholder implementation")
        
        # 串行收集作为回退
        env = self.env_fn()
        policy = self.policy_fn()
        
        collector = OfflineDataCollector(
            env=env,
            policy=policy,
            num_episodes=self.episodes_per_worker * self.num_workers,
            **self.collector_kwargs
        )
        
        return collector.collect()


def collect_offline_data(
    env,
    policy: Optional[Any] = None,
    num_episodes: int = 1000,
    output_dir: str = './offline_data',
    save_interval: int = 100,
    render: bool = False,
    **kwargs
) -> MAOfflineDataset:
    """
    便捷的 offline 数据收集函数
    
    Args:
        env: 多智能体环境
        policy: 收集策略（None 则使用随机策略）
        num_episodes: 轨迹数量
        output_dir: 输出目录
        save_interval: 保存间隔
        render: 是否渲染
        **kwargs: 其他参数
    
    Returns:
        收集的数据集
    """
    collector = OfflineDataCollector(
        env=env,
        policy=policy,
        num_episodes=num_episodes,
        output_dir=output_dir,
        save_interval=save_interval,
        render=render,
        **kwargs
    )
    
    return collector.collect()


def convert_to_osrl_format(
    dataset: MAOfflineDataset,
    output_path: str,
    normalize: bool = True
):
    """
    将数据集转换为 OSRL 兼容格式
    
    Args:
        dataset: MAOfflineDataset 实例
        output_path: 输出路径
        normalize: 是否归一化
    """
    data = dataset.get_centralized_data()
    
    if normalize:
        # 计算归一化参数
        obs_mean = np.mean(data['observations'], axis=0)
        obs_std = np.std(data['observations'], axis=0) + 1e-8
        
        act_mean = np.mean(data['actions'], axis=0)
        act_std = np.std(data['actions'], axis=0) + 1e-8
        
        # 归一化
        data['observations'] = (data['observations'] - obs_mean) / obs_std
        data['actions'] = (data['actions'] - act_mean) / act_std
        
        # 保存归一化参数
        data['obs_mean'] = obs_mean
        data['obs_std'] = obs_std
        data['act_mean'] = act_mean
        data['act_std'] = act_std
    
    # 保存为 npz 格式（OSRL 常用格式）
    np.savez(output_path, **data)
    print(f"Data saved to {output_path} in OSRL format")
    
    return data