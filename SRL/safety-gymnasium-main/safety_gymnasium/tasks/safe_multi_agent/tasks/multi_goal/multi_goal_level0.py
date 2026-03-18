# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Multi Goal level 0."""

import numpy as np

from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.goal import Goal
from safety_gymnasium.tasks.safe_multi_agent.bases.base_task import BaseTask


class MultiGoalLevel0(BaseTask):
    """An agent must navigate to a goal."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        arena_half_extent = 1.0 + 0.35 * max(0, self.agent.nums - 3)
        self.placements_conf.extents = [
            -arena_half_extent,
            -arena_half_extent,
            arena_half_extent,
            arena_half_extent,
        ]

        self.goal_names = [f'goal_{idx}' for idx in range(self.agent.nums)]
        self._goal_objects = []
        self.last_dist_to_goals = [0.0 for _ in range(self.agent.nums)]

        color_cycle = [
            np.array([0.7412, 0.0431, 0.1843, 1.0]),
            np.array([0.0039, 0.1529, 0.3961, 1.0]),
            np.array([0.1686, 0.5137, 0.2588, 1.0]),
            np.array([0.9725, 0.6745, 0.1098, 1.0]),
            np.array([0.5804, 0.4039, 0.7412, 1.0]),
            np.array([0.2275, 0.5255, 0.7686, 1.0]),
        ]

        goal_keepout = max(0.18, 0.305 - 0.03 * max(0, self.agent.nums - 2))

        for idx, goal_name in enumerate(self.goal_names):
            goal_obj = Goal(
                name=goal_name,
                keepout=goal_keepout,
                color=color_cycle[idx % len(color_cycle)],
            )
            self._goal_objects.append(goal_obj)
            setattr(self, goal_name, goal_obj)

        self._add_geoms(*self._goal_objects)

    def dist_goal(self, agent_idx: int) -> float:
        """Return distance from one agent to its paired goal."""
        goal_name = self.goal_names[agent_idx]
        assert hasattr(self, goal_name), f'Please make sure you have added {goal_name} into env.'
        goal_obj = getattr(self, goal_name)
        return self.agent.dist_xy(agent_idx, goal_obj.pos)

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        reward = {agent_name: 0.0 for agent_name in self.agent.possible_agents}

        for idx, agent_name in enumerate(self.agent.possible_agents):
            goal_obj = getattr(self, self.goal_names[idx])
            dist_to_goal = self.dist_goal(idx)
            reward[agent_name] += (
                self.last_dist_to_goals[idx] - dist_to_goal
            ) * goal_obj.reward_distance
            self.last_dist_to_goals[idx] = dist_to_goal

            if self.goal_achieved[idx]:
                reward[agent_name] += goal_obj.reward_goal

        return reward

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        self.build_goal_position()
        self.last_dist_to_goals = [self.dist_goal(idx) for idx in range(self.agent.nums)]

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        return tuple(
            self.dist_goal(idx) <= getattr(self, self.goal_names[idx]).size
            for idx in range(self.agent.nums)
        )
