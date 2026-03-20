from envs.env_factory import make_marl_env

env = make_marl_env(num_agents=8, backend='safety_gym', render_mode='rgb_array')
obs, info = env.reset(seed=42)
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
obs, rewards, costs, terminations, truncations, infos = env.step(actions)
env.close()
print(obs['agent_0'].shape)#打印agent_0的观测维度
print(costs['agent_0'],actions['agent_0'])
