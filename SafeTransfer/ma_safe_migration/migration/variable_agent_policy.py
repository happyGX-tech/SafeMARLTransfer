"""
可变智能体数量的策略网络
支持处理不同数量的智能体输入/输出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class AgentEncoder(nn.Module):
    """
    单个智能体的编码器
    将智能体的观测编码为特征向量
    """
    
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.hidden_dim = hidden_dim
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch_size, obs_dim)
        Returns:
            features: (batch_size, hidden_dim)
        """
        return self.net(obs)


class AgentDecoder(nn.Module):
    """
    单个智能体的解码器
    将特征向量解码为动作
    """
    
    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # 输出范围 [-1, 1]
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch_size, hidden_dim)
        Returns:
            actions: (batch_size, action_dim)
        """
        return self.net(features)


class MultiHeadAttentionPooling(nn.Module):
    """
    多头注意力池化
    用于聚合多个智能体的信息
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (batch_size, 1, hidden_dim)
            key_value: (batch_size, num_agents, hidden_dim)
            mask: (batch_size, num_agents) 可选的掩码
        Returns:
            output: (batch_size, 1, hidden_dim)
        """
        output, _ = self.attention(query, key_value, key_value, key_padding_mask=mask)
        return output


class VariableAgentPolicy(nn.Module):
    """
    可变智能体数量的策略网络
    
    特点：
    1. 使用共享的编码器处理每个智能体的观测
    2. 使用注意力机制聚合其他智能体的信息
    3. 使用共享的解码器生成动作
    4. 支持可变数量的智能体
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        max_agents: int = 10,
    ):
        """
        初始化策略网络
        
        Args:
            obs_dim: 每个智能体的观测维度
            action_dim: 每个智能体的动作维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            max_agents: 支持的最大智能体数量
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_agents = max_agents
        
        # 共享的编码器和解码器
        self.encoder = AgentEncoder(obs_dim, hidden_dim)
        self.decoder = AgentDecoder(hidden_dim, action_dim)
        
        # 注意力池化（用于聚合其他智能体信息）
        self.attention = MultiHeadAttentionPooling(hidden_dim, num_heads)
        
        # 用于生成查询向量的网络
        self.query_net = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        observations: torch.Tensor,
        num_agents: Optional[int] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            observations: (batch_size, max_agents, obs_dim) 或 (batch_size, num_agents, obs_dim)
            num_agents: 实际的智能体数量（如果 observations 包含填充）
            masks: (batch_size, max_agents) 用于标记哪些智能体是有效的
        
        Returns:
            actions: (batch_size, num_agents, action_dim)
        """
        batch_size = observations.shape[0]
        
        if num_agents is None:
            num_agents = observations.shape[1]
        
        # 编码所有智能体的观测
        # (batch_size * num_agents, obs_dim) -> (batch_size * num_agents, hidden_dim)
        flat_obs = observations.view(-1, self.obs_dim)
        flat_features = self.encoder(flat_obs)
        features = flat_features.view(batch_size, num_agents, self.hidden_dim)
        
        # 为每个智能体生成动作
        actions_list = []
        
        for i in range(num_agents):
            # 当前智能体的特征
            agent_feature = features[:, i:i+1, :]  # (batch_size, 1, hidden_dim)
            
            # 生成查询向量
            query = self.query_net(agent_feature)  # (batch_size, 1, hidden_dim)
            
            # 聚合其他智能体的信息
            if num_agents > 1:
                # 创建掩码，屏蔽当前智能体
                other_mask = torch.ones(batch_size, num_agents, device=observations.device)
                other_mask[:, i] = 0
                
                if masks is not None:
                    other_mask = other_mask * masks
                
                # 注意力池化
                attended = self.attention(
                    query,
                    features,
                    mask=(1 - other_mask).bool()
                )
                
                # 结合自身特征和注意力特征
                combined = agent_feature + attended
            else:
                combined = agent_feature
            
            # 解码动作
            agent_action = self.decoder(combined.squeeze(1))  # (batch_size, action_dim)
            actions_list.append(agent_action)
        
        # 拼接所有智能体的动作
        actions = torch.stack(actions_list, dim=1)  # (batch_size, num_agents, action_dim)
        
        return actions
    
    def get_action(
        self,
        observations: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        获取动作（用于推理）
        
        Args:
            observations: {agent_id: obs}
            deterministic: 是否确定性采样
        
        Returns:
            actions: {agent_id: action}
        """
        agent_ids = sorted(observations.keys())
        num_agents = len(agent_ids)
        
        # 转换为 tensor
        obs_array = np.stack([observations[aid] for aid in agent_ids])
        obs_tensor = torch.FloatTensor(obs_array).unsqueeze(0)  # (1, num_agents, obs_dim)
        
        with torch.no_grad():
            actions_tensor = self.forward(obs_tensor, num_agents=num_agents)
            
            if deterministic:
                actions_array = actions_tensor.squeeze(0).cpu().numpy()
            else:
                # 添加高斯噪声
                noise = torch.randn_like(actions_tensor) * 0.1
                actions_array = (actions_tensor + noise).squeeze(0).cpu().numpy()
        
        # 转换回字典格式
        actions = {aid: actions_array[i] for i, aid in enumerate(agent_ids)}
        
        return actions


class VariableAgentCritic(nn.Module):
    """
    可变智能体数量的 Critic 网络
    估计联合价值函数
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 编码器
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.act_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 注意力聚合
        self.attention = MultiHeadAttentionPooling(hidden_dim * 2, num_heads)
        
        # 价值头
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            observations: (batch_size, num_agents, obs_dim)
            actions: (batch_size, num_agents, action_dim)
        
        Returns:
            values: (batch_size, 1)
        """
        batch_size, num_agents, _ = observations.shape
        
        # 编码观测和动作
        obs_features = self.obs_encoder(observations)  # (batch_size, num_agents, hidden_dim)
        act_features = self.act_encoder(actions)       # (batch_size, num_agents, hidden_dim)
        
        # 拼接特征
        combined = torch.cat([obs_features, act_features], dim=-1)
        
        # 使用第一个智能体作为查询（或使用全局池化）
        query = combined.mean(dim=1, keepdim=True)  # (batch_size, 1, hidden_dim*2)
        
        # 注意力聚合
        attended = self.attention(query, combined)
        
        # 计算价值
        values = self.value_net(attended.squeeze(1))
        
        return values


class MigrationTrainer:
    """
    迁移学习训练器
    支持从 N 智能体迁移到 M 智能体
    """
    
    def __init__(
        self,
        policy: VariableAgentPolicy,
        critic: VariableAgentCritic,
        source_num_agents: int = 2,
        target_num_agents: int = 4,
        lr: float = 3e-4,
        device: str = 'cpu',
    ):
        """
        初始化训练器
        
        Args:
            policy: 策略网络
            critic: Critic 网络
            source_num_agents: 源环境智能体数量
            target_num_agents: 目标环境智能体数量
            lr: 学习率
            device: 计算设备
        """
        self.policy = policy.to(device)
        self.critic = critic.to(device)
        self.source_num_agents = source_num_agents
        self.target_num_agents = target_num_agents
        self.device = device
        
        self.policy_optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
    
    def train_on_source(self, dataset, num_epochs: int = 100):
        """
        在源环境数据上训练
        
        Args:
            dataset: 源环境数据集
            num_epochs: 训练轮数
        """
        print(f"Training on source environment ({self.source_num_agents} agents)")
        
        for epoch in range(num_epochs):
            # 这里应该实现具体的训练逻辑
            # 例如：BC (Behavior Cloning), CQL (Conservative Q-Learning) 等
            
            pass
    
    def fine_tune_on_target(self, dataset, num_epochs: int = 50):
        """
        在目标环境数据上微调
        
        Args:
            dataset: 目标环境数据集
            num_epochs: 微调轮数
        """
        print(f"Fine-tuning on target environment ({self.target_num_agents} agents)")
        
        for epoch in range(num_epochs):
            # 微调逻辑
            pass
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            'source_num_agents': self.source_num_agents,
            'target_num_agents': self.target_num_agents,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.critic.load_state_dict(checkpoint['critic'])
        print(f"Model loaded from {path}")


def create_variable_agent_policy(
    obs_dim: int,
    action_dim: int,
    hidden_dim: int = 128,
    max_agents: int = 10,
) -> VariableAgentPolicy:
    """
    创建可变智能体策略的便捷函数
    
    Args:
        obs_dim: 观测维度
        action_dim: 动作维度
        hidden_dim: 隐藏层维度
        max_agents: 最大智能体数量
    
    Returns:
        VariableAgentPolicy 实例
    """
    return VariableAgentPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        max_agents=max_agents,
    )


if __name__ == '__main__':
    # 测试策略网络
    print("Testing VariableAgentPolicy")
    
    obs_dim = 17  # 示例观测维度
    action_dim = 2  # 示例动作维度
    
    policy = VariableAgentPolicy(obs_dim, action_dim, hidden_dim=64)
    
    # 测试不同智能体数量
    for num_agents in [2, 3, 4, 5]:
        batch_size = 8
        obs = torch.randn(batch_size, num_agents, obs_dim)
        
        actions = policy(obs, num_agents=num_agents)
        
        print(f"  {num_agents} agents: input {obs.shape} -> output {actions.shape}")
        assert actions.shape == (batch_size, num_agents, action_dim)
    
    print("✓ All tests passed!")