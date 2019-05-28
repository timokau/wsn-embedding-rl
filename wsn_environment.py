"""Coach environment wrapper for a Wireless Sensor Network"""

import random
from typing import Union

import numpy as np

from rl_coach.base_parameters import VisualizationParameters
from rl_coach.spaces import StateSpace, DiscreteActionSpace
from rl_coach.environments.environment import (
    Environment,
    EnvironmentParameters,
)
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter
from generator_visualization import random_embedding


class WSNEnvironment(Environment):
    # That is just how coach environments work
    # pylint: disable=too-many-instance-attributes
    """Wireless Sensor Network Environment"""

    def __init__(
        self,
        frame_skip: int,
        visualization_parameters: VisualizationParameters,
        seed: Union[None, int] = None,
        human_control: bool = False,
        custom_reward_threshold: Union[int, float] = None,
        **kwargs,
    ):
        super().__init__(
            level=None,
            seed=seed,
            frame_skip=frame_skip,
            human_control=human_control,
            custom_reward_threshold=custom_reward_threshold,
            visualization_parameters=visualization_parameters,
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        self.env = None
        self.last_result = None

        self.state_space = StateSpace({})
        self.state_space["observation"] = None
        self.actions = []
        self.action_space = DiscreteActionSpace(1)

        self.reset_internal_state(True)

    def _query_actions(self):
        self.actions = self.env.possibilities()
        self.action_space.high = max(len(self.actions) - 1, 0)

    def _update_state(self):
        self.state = {}

        self.state["graph"] = self.last_result["graph"]

        self.reward = self.last_result.get("reward", 0)

        self.done = self.last_result["done"]
        self._query_actions()

    def _take_action(self, action_idx):
        print(action_idx)
        (source, sink, timeslot) = self.actions[action_idx]
        ts_before = self.env.used_timeslots
        self.env.take_action(source, sink, timeslot)
        self.last_result = {
            "graph": self.env.graph.copy(),
            "reward": ts_before - self.env.used_timeslots,
            "done": self.env.is_complete(),
        }

    def _restart_environment_episode(self, force_environment_reset=False):
        self.env = random_embedding(50, np.random)
        self.last_result = {
            "graph": self.env.graph.copy(),
            "reward": 0,
            "done": False,
        }
        self._query_actions()

    def get_rendered_image(self):
        raise NotImplementedError()


class WSNEnvironmentParameters(EnvironmentParameters):
    """Parameters for the Wireless Sensor Network Environment"""

    def __init__(self):
        super().__init__()
        self.default_input_filter = NoInputFilter()
        self.default_output_filter = NoOutputFilter()

    @property
    def path(self):
        return "wsn_environment:WSNEnvironment"
