"""Variable-agent MultiGoal environment for Safety-Gymnasium integration.

This environment is designed for transfer-learning studies where N agents must
reach N goals while avoiding hazards and inter-agent collisions.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class VariableMultiGoalEnv(gym.Env):
    """N-agent to N-goal safety environment.

    Notes:
        - API follows Safety-Gym multi-agent convention: dict observations/actions.
        - Cost includes hazard contact and inter-agent collision penalties.
    """

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30,
    }

    MIN_AGENTS = 2

    def __init__(
        self,
        num_agents: int = 2,
        num_hazards: int = 6,
        world_size: float = 3.0,
        max_episode_steps: int = 1000,
        render_mode: Optional[str] = None,
        agent_radius: float = 0.25,
        goal_radius: float = 0.25,
        hazard_radius: float = 0.25,
        goal_reward: float = 1.0,
        hazard_penalty: float = 1.0,
        collision_penalty: float = 1.0,
        observe_other_agents: bool = True,
        **kwargs: Any,
    ):
        super().__init__()

        if num_agents < self.MIN_AGENTS:
            raise ValueError(f'num_agents must be >= {self.MIN_AGENTS}')

        self.num_agents = num_agents
        self.num_hazards = num_hazards
        self.world_size = world_size
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        self.agent_radius = agent_radius
        self.goal_radius = goal_radius
        self.hazard_radius = hazard_radius

        self.goal_reward = goal_reward
        self.hazard_penalty = hazard_penalty
        self.collision_penalty = collision_penalty
        self.observe_other_agents = observe_other_agents

        self.possible_agents = [f'agent_{i}' for i in range(self.num_agents)]
        self._agents = self.possible_agents.copy()

        self._np_random = np.random.default_rng()
        self.current_step = 0

        self.agent_positions: Dict[str, np.ndarray] = {}
        self.agent_velocities: Dict[str, np.ndarray] = {}
        self.goal_positions: Dict[str, np.ndarray] = {}
        self.hazard_positions: List[np.ndarray] = []
        self.agent_reached_goal: Dict[str, bool] = {}
        self._human_figure = None
        self._human_axes = None
        self._human_render_disabled = False
        self._human_render_error = None

        self.action_dim = 2
        self._action_spaces = {
            name: spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
            for name in self.possible_agents
        }

        base_obs_dim = 2 + 2 + 2 + 3
        if self.observe_other_agents:
            base_obs_dim += (self.num_agents - 1) * 3

        self._observation_spaces = {
            name: spaces.Box(low=-np.inf, high=np.inf, shape=(base_obs_dim,), dtype=np.float32)
            for name in self.possible_agents
        }

    @property
    def agents(self) -> List[str]:
        return self._agents.copy()

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_spaces[agent]

    def observation_space(self, agent: str) -> spaces.Space:
        return self._observation_spaces[agent]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        self.current_step = 0
        self._agents = self.possible_agents.copy()

        self.hazard_positions = self._generate_hazard_positions()
        self.agent_positions = self._generate_agent_positions()
        self.agent_velocities = {name: np.zeros(2, dtype=np.float32) for name in self.possible_agents}
        self.goal_positions = self._assign_goals()
        self.agent_reached_goal = {name: False for name in self.possible_agents}

        observations = {name: self._get_obs(name) for name in self.possible_agents}
        infos = {name: {} for name in self.possible_agents}
        return observations, infos

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        self.current_step += 1

        for name in self.possible_agents:
            action = np.clip(np.asarray(actions[name], dtype=np.float32), -1.0, 1.0)
            self.agent_velocities[name] = action * 0.1
            self.agent_positions[name] = np.clip(
                self.agent_positions[name] + self.agent_velocities[name],
                -self.world_size,
                self.world_size,
            )

        agent_collisions = self._detect_agent_collisions()

        rewards: Dict[str, float] = {}
        costs: Dict[str, float] = {}
        terminations: Dict[str, bool] = {}
        truncations: Dict[str, bool] = {}
        infos: Dict[str, Dict[str, Any]] = {}

        all_reached = all(self.agent_reached_goal.values())

        for name in self.possible_agents:
            dist_to_goal = float(np.linalg.norm(self.agent_positions[name] - self.goal_positions[name]))
            reward = -0.01 * dist_to_goal

            if dist_to_goal < self.goal_radius and not self.agent_reached_goal[name]:
                reward += self.goal_reward
                self.agent_reached_goal[name] = True

            cost = 0.0
            hazard_dist = self._min_distance_to_hazards(self.agent_positions[name])
            if hazard_dist < self.hazard_radius + self.agent_radius:
                cost += self.hazard_penalty
            cost += self.collision_penalty * agent_collisions.get(name, 0)

            rewards[name] = reward
            costs[name] = cost

            terminations[name] = all_reached
            truncations[name] = self.current_step >= self.max_episode_steps
            infos[name] = {
                'distance_to_goal': dist_to_goal,
                'reached_goal': self.agent_reached_goal[name],
                'hazard_distance': hazard_dist,
                'step': self.current_step,
            }

        observations = {name: self._get_obs(name) for name in self.possible_agents}

        if self.render_mode == 'human':
            self.render()

        return observations, rewards, costs, terminations, truncations, infos

    def render(self):
        if self.render_mode is None:
            return None

        if self.render_mode == 'human' and self._human_render_disabled:
            return None

        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        if self.render_mode == 'human':
            try:
                if self._human_figure is None or not plt.fignum_exists(self._human_figure.number):
                    plt.ion()
                    self._human_figure, self._human_axes = plt.subplots(figsize=(6, 6))

                ax = self._human_axes
                ax.clear()
                self._draw_scene(ax, plt, patches)
                self._human_figure.canvas.draw_idle()
                self._human_figure.canvas.flush_events()
                plt.pause(1.0 / max(1, int(self.metadata.get('render_fps', 30))))
                return None
            except Exception as exc:
                self._human_render_disabled = True
                self._human_render_error = str(exc)
                print(
                    '[Warning] Human rendering disabled for VariableMultiGoalEnv '
                    f'due to backend/display error: {exc}'
                )
                return None

        fig, ax = plt.subplots(figsize=(6, 6))
        self._draw_scene(ax, plt, patches)

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        frame = plt.imread(buf)
        plt.close(fig)
        return (frame[:, :, :3] * 255).astype(np.uint8)

    def close(self):
        if self._human_figure is not None:
            import matplotlib.pyplot as plt

            if plt.fignum_exists(self._human_figure.number):
                plt.close(self._human_figure)
        self._human_figure = None
        self._human_axes = None
        return None

    def _draw_scene(self, ax, plt, patches):
        use_camera_follow = (
            self.render_mode == 'human'
            and self.num_agents == 2
            and 'agent_0' in self.agent_positions
        )

        if use_camera_follow:
            camera_target = self.agent_positions['agent_0']
            half_view = min(self.world_size, 2.75)
            ax.set_xlim(camera_target[0] - half_view, camera_target[0] + half_view)
            ax.set_ylim(camera_target[1] - half_view, camera_target[1] + half_view)
        else:
            ax.set_xlim(-self.world_size - 0.5, self.world_size + 0.5)
            ax.set_ylim(-self.world_size - 0.5, self.world_size + 0.5)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25)

        for hazard_pos in self.hazard_positions:
            ax.add_patch(patches.Circle(hazard_pos, self.hazard_radius, color='red', alpha=0.45))

        colors = plt.cm.tab10(np.linspace(0, 1, max(self.num_agents, 3)))
        for i, name in enumerate(self.possible_agents):
            g = self.goal_positions[name]
            a = self.agent_positions[name]
            ax.add_patch(patches.Circle(g, self.goal_radius, color=colors[i], alpha=0.30))
            ax.add_patch(patches.Circle(a, self.agent_radius, color=colors[i], alpha=0.85))
            ax.text(g[0], g[1], f'G{i}', ha='center', va='center', fontsize=8)
            ax.text(a[0], a[1], f'A{i}', ha='center', va='center', fontsize=8, color='white')

        mode_tag = 'camera-follow' if use_camera_follow else 'global-overview'
        ax.set_title(f'VariableMultiGoal | step={self.current_step} | agents={self.num_agents} | {mode_tag}')

    def _get_obs(self, agent_name: str) -> np.ndarray:
        pos = self.agent_positions[agent_name]
        vel = self.agent_velocities[agent_name]
        goal_rel = self.goal_positions[agent_name] - pos

        nearest_hazard = self._nearest_hazard(pos)
        if nearest_hazard is None:
            hazard_vec = np.zeros(2, dtype=np.float32)
            hazard_dist = np.array([self.world_size * 2], dtype=np.float32)
        else:
            hazard_vec = nearest_hazard - pos
            hazard_dist = np.array([np.linalg.norm(hazard_vec)], dtype=np.float32)

        obs_items = [pos.astype(np.float32), vel.astype(np.float32), goal_rel.astype(np.float32), hazard_vec.astype(np.float32), hazard_dist.astype(np.float32)]

        if self.observe_other_agents:
            for other_name in self.possible_agents:
                if other_name == agent_name:
                    continue
                rel = self.agent_positions[other_name] - pos
                rel_dist = np.array([np.linalg.norm(rel)], dtype=np.float32)
                obs_items.extend([rel.astype(np.float32), rel_dist.astype(np.float32)])

        return np.concatenate(obs_items, axis=0).astype(np.float32)

    def _generate_agent_positions(self) -> Dict[str, np.ndarray]:
        positions: Dict[str, np.ndarray] = {}
        for name in self.possible_agents:
            while True:
                pos = self._np_random.uniform(-self.world_size * 0.8, self.world_size * 0.8, size=2)
                if self._is_position_safe(pos, positions.values(), self.hazard_positions):
                    positions[name] = pos.astype(np.float32)
                    break
        return positions

    def _assign_goals(self) -> Dict[str, np.ndarray]:
        goals: Dict[str, np.ndarray] = {}
        for name in self.possible_agents:
            while True:
                goal = self._np_random.uniform(-self.world_size * 0.8, self.world_size * 0.8, size=2)
                if self._is_position_safe(goal, goals.values(), self.hazard_positions, min_dist=2 * self.goal_radius):
                    goals[name] = goal.astype(np.float32)
                    break
        return goals

    def _generate_hazard_positions(self) -> List[np.ndarray]:
        hazards: List[np.ndarray] = []
        for _ in range(self.num_hazards):
            tries = 0
            while True:
                h = self._np_random.uniform(-self.world_size * 0.9, self.world_size * 0.9, size=2)
                if self._is_position_safe(h, hazards, [], min_dist=2 * self.hazard_radius):
                    hazards.append(h.astype(np.float32))
                    break
                tries += 1
                if tries > 200:
                    break
        return hazards

    def _is_position_safe(
        self,
        pos: np.ndarray,
        existing_positions,
        hazard_positions,
        min_dist: Optional[float] = None,
    ) -> bool:
        if min_dist is None:
            min_dist = 2 * self.agent_radius

        for e in existing_positions:
            if np.linalg.norm(pos - np.asarray(e)) < min_dist:
                return False
        for h in hazard_positions:
            if np.linalg.norm(pos - np.asarray(h)) < (self.agent_radius + self.hazard_radius):
                return False
        return True

    def _detect_agent_collisions(self) -> Dict[str, int]:
        collisions = {name: 0 for name in self.possible_agents}
        names = self.possible_agents
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if np.linalg.norm(self.agent_positions[a] - self.agent_positions[b]) < (2 * self.agent_radius):
                    collisions[a] += 1
                    collisions[b] += 1
        return collisions

    def _min_distance_to_hazards(self, pos: np.ndarray) -> float:
        if not self.hazard_positions:
            return float('inf')
        return min(float(np.linalg.norm(pos - h)) for h in self.hazard_positions)

    def _nearest_hazard(self, pos: np.ndarray) -> Optional[np.ndarray]:
        if not self.hazard_positions:
            return None
        nearest = min(self.hazard_positions, key=lambda h: np.linalg.norm(pos - h))
        return np.asarray(nearest, dtype=np.float32)
