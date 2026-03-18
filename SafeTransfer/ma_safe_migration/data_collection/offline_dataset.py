"""
Offline 数据集管理模块
用于存储和管理多智能体的 offline 数据
参考 OSRL 的数据格式设计
"""

import numpy as np
import torch
import h5py
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import pickle


@dataclass
class MATrajectory:
    """
    多智能体轨迹数据类
    存储一条完整轨迹的所有信息
    """
    observations: Dict[str, np.ndarray] = field(default_factory=dict)  # {agent: obs_array}
    actions: Dict[str, np.ndarray] = field(default_factory=dict)       # {agent: action_array}
    rewards: Dict[str, np.ndarray] = field(default_factory=dict)       # {agent: reward_array}
    costs: Dict[str, np.ndarray] = field(default_factory=dict)         # {agent: cost_array}
    terminals: np.ndarray = field(default_factory=lambda: np.array([]))  # 终止标志
    timeouts: np.ndarray = field(default_factory=lambda: np.array([]))   # 超时标志
    
    # 额外信息
    infos: Dict[str, List[Dict]] = field(default_factory=dict)  # {agent: [info_dict]}
    
    # 轨迹元数据
    trajectory_id: int = 0
    num_agents: int = 0
    length: int = 0
    total_reward: Dict[str, float] = field(default_factory=dict)
    total_cost: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        if self.observations:
            self.num_agents = len(self.observations)
            agent = list(self.observations.keys())[0]
            self.length = len(self.observations[agent])
    
    def add_step(
        self,
        obs: Dict[str, np.ndarray],
        action: Dict[str, np.ndarray],
        reward: Dict[str, float],
        cost: Dict[str, float],
        terminal: bool,
        timeout: bool,
        info: Optional[Dict[str, Dict]] = None
    ):
        """添加一步数据到轨迹"""
        # 初始化存储
        if self.length == 0:
            for agent in obs.keys():
                self.observations[agent] = []
                self.actions[agent] = []
                self.rewards[agent] = []
                self.costs[agent] = []
                self.infos[agent] = []
            self.terminals = []
            self.timeouts = []
        
        # 添加数据
        for agent in obs.keys():
            self.observations[agent].append(obs[agent])
            self.actions[agent].append(action[agent])
            self.rewards[agent].append(reward[agent])
            self.costs[agent].append(cost[agent])
            if info:
                self.infos[agent].append(info.get(agent, {}))
        
        self.terminals.append(terminal)
        self.timeouts.append(timeout)
        self.length += 1
    
    def finalize(self):
        """将列表转换为 numpy 数组"""
        for agent in self.observations.keys():
            self.observations[agent] = np.array(self.observations[agent])
            self.actions[agent] = np.array(self.actions[agent])
            self.rewards[agent] = np.array(self.rewards[agent])
            self.costs[agent] = np.array(self.costs[agent])
            
            # 计算总奖励和成本
            self.total_reward[agent] = float(np.sum(self.rewards[agent]))
            self.total_cost[agent] = float(np.sum(self.costs[agent]))
        
        self.terminals = np.array(self.terminals)
        self.timeouts = np.array(self.timeouts)
    
    def get_agent_trajectory(self, agent: str) -> Dict[str, np.ndarray]:
        """获取单个智能体的轨迹数据"""
        return {
            'observations': self.observations[agent],
            'actions': self.actions[agent],
            'rewards': self.rewards[agent],
            'costs': self.costs[agent],
            'terminals': self.terminals,
            'timeouts': self.timeouts,
        }


class MAOfflineDataset:
    """
    多智能体 Offline 数据集
    支持 centralized 数据收集和 decentralized 存储
    """
    
    def __init__(self, num_agents: Optional[int] = None):
        """
        初始化数据集
        
        Args:
            num_agents: 智能体数量（可选，用于验证）
        """
        self.num_agents = num_agents
        self.trajectories: List[MATrajectory] = []
        
        # 统计数据
        self.total_steps = 0
        self.total_episodes = 0
        self.agent_names: List[str] = []
        
        # 数据缓冲区（用于正在收集的轨迹）
        self._current_trajectory: Optional[MATrajectory] = None
    
    def start_trajectory(self, trajectory_id: Optional[int] = None):
        """开始一条新轨迹"""
        self._current_trajectory = MATrajectory(trajectory_id=trajectory_id or self.total_episodes)
    
    def add_step(
        self,
        obs: Dict[str, np.ndarray],
        action: Dict[str, np.ndarray],
        reward: Dict[str, float],
        cost: Dict[str, float],
        terminal: bool,
        timeout: bool,
        info: Optional[Dict[str, Dict]] = None
    ):
        """添加一步数据"""
        if self._current_trajectory is None:
            self.start_trajectory()
        
        self._current_trajectory.add_step(obs, action, reward, cost, terminal, timeout, info)
        
        # 记录智能体名称
        if not self.agent_names:
            self.agent_names = list(obs.keys())
    
    def end_trajectory(self):
        """结束当前轨迹并保存"""
        if self._current_trajectory is not None and self._current_trajectory.length > 0:
            self._current_trajectory.finalize()
            self.trajectories.append(self._current_trajectory)
            self.total_steps += self._current_trajectory.length
            self.total_episodes += 1
            self._current_trajectory = None
    
    def add_trajectory(self, trajectory: MATrajectory):
        """添加完整轨迹"""
        self.trajectories.append(trajectory)
        self.total_steps += trajectory.length
        self.total_episodes += 1
        
        if not self.agent_names and trajectory.observations:
            self.agent_names = list(trajectory.observations.keys())
    
    def get_all_data(self) -> Dict[str, np.ndarray]:
        """
        获取所有数据（拼接所有轨迹）
        
        Returns:
            包含所有轨迹数据的字典
        """
        if not self.trajectories:
            return {}
        
        all_data = {
            'observations': {},
            'actions': {},
            'rewards': {},
            'costs': {},
            'terminals': [],
            'timeouts': [],
        }
        
        for agent in self.agent_names:
            all_data['observations'][agent] = []
            all_data['actions'][agent] = []
            all_data['rewards'][agent] = []
            all_data['costs'][agent] = []
        
        for traj in self.trajectories:
            for agent in self.agent_names:
                all_data['observations'][agent].append(traj.observations[agent])
                all_data['actions'][agent].append(traj.actions[agent])
                all_data['rewards'][agent].append(traj.rewards[agent])
                all_data['costs'][agent].append(traj.costs[agent])
            all_data['terminals'].append(traj.terminals)
            all_data['timeouts'].append(traj.timeouts)
        
        # 拼接数组
        for agent in self.agent_names:
            all_data['observations'][agent] = np.concatenate(all_data['observations'][agent])
            all_data['actions'][agent] = np.concatenate(all_data['actions'][agent])
            all_data['rewards'][agent] = np.concatenate(all_data['rewards'][agent])
            all_data['costs'][agent] = np.concatenate(all_data['costs'][agent])
        all_data['terminals'] = np.concatenate(all_data['terminals'])
        all_data['timeouts'] = np.concatenate(all_data['timeouts'])
        
        return all_data
    
    def get_centralized_data(self) -> Dict[str, np.ndarray]:
        """
        获取 centralized 格式的数据
        将所有智能体的观测和动作拼接在一起
        
        Returns:
            centralized 数据字典
        """
        data = self.get_all_data()
        if not data:
            return {}
        
        # 获取智能体数量
        num_agents = len(self.agent_names)
        
        # 计算观测和动作维度（允许不同智能体维度不同，统一 padding 到最大维度）
        obs_dim = max(data['observations'][agent].shape[1] for agent in self.agent_names)
        act_dim = max(data['actions'][agent].shape[1] for agent in self.agent_names)
        
        # 拼接所有智能体的数据
        centralized_obs = np.zeros((len(data['terminals']), num_agents, obs_dim))
        centralized_act = np.zeros((len(data['terminals']), num_agents, act_dim))
        centralized_rew = np.zeros((len(data['terminals']), num_agents))
        centralized_cost = np.zeros((len(data['terminals']), num_agents))
        
        for i, agent in enumerate(self.agent_names):
            agent_obs = data['observations'][agent]
            agent_act = data['actions'][agent]
            centralized_obs[:, i, :agent_obs.shape[1]] = agent_obs
            centralized_act[:, i, :agent_act.shape[1]] = agent_act
            centralized_rew[:, i] = data['rewards'][agent]
            centralized_cost[:, i] = data['costs'][agent]
        
        return {
            'observations': centralized_obs,
            'actions': centralized_act,
            'rewards': centralized_rew,
            'costs': centralized_cost,
            'terminals': data['terminals'],
            'timeouts': data['timeouts'],
            'num_agents': num_agents,
        }
    
    def save_to_hdf5(self, filepath: str):
        """
        保存数据集到 HDF5 文件
        
        Args:
            filepath: 保存路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            # 保存元数据
            f.attrs['num_agents'] = self.num_agents if self.num_agents else len(self.agent_names)
            f.attrs['total_episodes'] = self.total_episodes
            f.attrs['total_steps'] = self.total_steps
            f.attrs['agent_names'] = ','.join(self.agent_names)
            
            # 保存每条轨迹
            for i, traj in enumerate(self.trajectories):
                traj_group = f.create_group(f'trajectory_{i}')
                traj_group.attrs['length'] = traj.length
                traj_group.attrs['num_agents'] = traj.num_agents
                traj_group.attrs['trajectory_id'] = traj.trajectory_id
                
                # 保存每个智能体的数据
                for agent in self.agent_names:
                    agent_group = traj_group.create_group(agent)
                    agent_group.create_dataset('observations', data=traj.observations[agent])
                    agent_group.create_dataset('actions', data=traj.actions[agent])
                    agent_group.create_dataset('rewards', data=traj.rewards[agent])
                    agent_group.create_dataset('costs', data=traj.costs[agent])
                
                # 保存共享数据
                traj_group.create_dataset('terminals', data=traj.terminals)
                traj_group.create_dataset('timeouts', data=traj.timeouts)
        
        print(f"Dataset saved to {filepath}")
    
    @classmethod
    def load_from_hdf5(cls, filepath: str) -> 'MAOfflineDataset':
        """
        从 HDF5 文件加载数据集
        
        Args:
            filepath: 文件路径
        
        Returns:
            MAOfflineDataset 实例
        """
        dataset = cls()
        
        with h5py.File(filepath, 'r') as f:
            # 加载元数据
            dataset.num_agents = f.attrs['num_agents']
            dataset.total_episodes = f.attrs['total_episodes']
            dataset.total_steps = f.attrs['total_steps']
            dataset.agent_names = f.attrs['agent_names'].split(',')
            
            # 加载每条轨迹
            for i in range(dataset.total_episodes):
                traj_group = f[f'trajectory_{i}']
                traj = MATrajectory(
                    trajectory_id=traj_group.attrs['trajectory_id'],
                    num_agents=traj_group.attrs['num_agents'],
                    length=traj_group.attrs['length']
                )
                
                # 加载每个智能体的数据
                for agent in dataset.agent_names:
                    agent_group = traj_group[agent]
                    traj.observations[agent] = agent_group['observations'][:]
                    traj.actions[agent] = agent_group['actions'][:]
                    traj.rewards[agent] = agent_group['rewards'][:]
                    traj.costs[agent] = agent_group['costs'][:]
                
                traj.terminals = traj_group['terminals'][:]
                traj.timeouts = traj_group['timeouts'][:]
                
                # 计算统计信息
                for agent in dataset.agent_names:
                    traj.total_reward[agent] = float(np.sum(traj.rewards[agent]))
                    traj.total_cost[agent] = float(np.sum(traj.costs[agent]))
                
                dataset.trajectories.append(traj)
        
        print(f"Dataset loaded from {filepath}")
        print(f"Total episodes: {dataset.total_episodes}, Total steps: {dataset.total_steps}")
        
        return dataset
    
    def save_to_pickle(self, filepath: str):
        """保存到 pickle 文件"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'num_agents': self.num_agents,
                'total_episodes': self.total_episodes,
                'total_steps': self.total_steps,
                'agent_names': self.agent_names,
                'trajectories': self.trajectories,
            }, f)
        
        print(f"Dataset saved to {filepath}")
    
    @classmethod
    def load_from_pickle(cls, filepath: str) -> 'MAOfflineDataset':
        """从 pickle 文件加载"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        dataset = cls()
        dataset.num_agents = data['num_agents']
        dataset.total_episodes = data['total_episodes']
        dataset.total_steps = data['total_steps']
        dataset.agent_names = data['agent_names']
        dataset.trajectories = data['trajectories']
        
        return dataset
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        if not self.trajectories:
            return {}
        
        stats = {
            'num_agents': len(self.agent_names),
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'avg_episode_length': self.total_steps / max(1, self.total_episodes),
        }
        
        # 每个智能体的统计
        for agent in self.agent_names:
            rewards = [traj.total_reward[agent] for traj in self.trajectories]
            costs = [traj.total_cost[agent] for traj in self.trajectories]
            
            stats[f'{agent}_avg_reward'] = float(np.mean(rewards))
            stats[f'{agent}_avg_cost'] = float(np.mean(costs))
            stats[f'{agent}_max_reward'] = float(np.max(rewards))
            stats[f'{agent}_min_reward'] = float(np.min(rewards))
        
        return stats
    
    def print_statistics(self):
        """打印数据集统计信息"""
        stats = self.get_statistics()
        
        print("=" * 50)
        print("Dataset Statistics")
        print("=" * 50)
        print(f"Number of Agents: {stats.get('num_agents', 'N/A')}")
        print(f"Total Episodes: {stats.get('total_episodes', 'N/A')}")
        print(f"Total Steps: {stats.get('total_steps', 'N/A')}")
        print(f"Avg Episode Length: {stats.get('avg_episode_length', 'N/A'):.2f}")
        print()
        
        for agent in self.agent_names:
            print(f"Agent: {agent}")
            print(f"  Avg Reward: {stats.get(f'{agent}_avg_reward', 'N/A'):.4f}")
            print(f"  Avg Cost: {stats.get(f'{agent}_avg_cost', 'N/A'):.4f}")
            print(f"  Max Reward: {stats.get(f'{agent}_max_reward', 'N/A'):.4f}")
            print(f"  Min Reward: {stats.get(f'{agent}_min_reward', 'N/A'):.4f}")
        print("=" * 50)