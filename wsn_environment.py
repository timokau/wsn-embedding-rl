"""Coach environment wrapper for a Wireless Sensor Network"""

import random
from typing import Union

import numpy as np
import networkx as nx

from rl_coach.base_parameters import VisualizationParameters
from rl_coach.spaces import (
    StateSpace,
    DiscreteActionSpace,
    VectorObservationSpace,
)
from rl_coach.environments.environment import (
    Environment,
    EnvironmentParameters,
)
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter
from generator import random_embedding


class DynamicDiscreteActionSpace(DiscreteActionSpace):
    """
    A discrete action space that ignores deepcopies.
    """

    def __deepcopy__(self, memo):
        return self


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
        self.last_translation_dict = None

        self.state_space = StateSpace({})
        # this is a lie, there is no proper observation space yet and
        # the precise definition isn't important in my implementation
        self.state_space["observation"] = VectorObservationSpace(1)
        self.state_space["graph"] = VectorObservationSpace(1)
        self.state["observation"] = np.array([])
        self.actions = []
        self.action_space = DynamicDiscreteActionSpace(1)

        self.reset_internal_state(True)

    def _query_actions(self):
        self.actions = self.env.possibilities()
        self._get_observation()

        def key(a):
            (u, v, t) = a
            return (
                self.last_translation_dict[u],
                self.last_translation_dict[v],
                t,
            )

        self.actions.sort(key=key)
        self.action_space.high = max(len(self.actions) - 1, 0)

    def _get_observation(self):
        # build graphs from scratch, since we need to change the node
        # indexing (graph_nets can only deal with integer indexed nodes)
        input_graph = nx.MultiDiGraph()
        embedding = self.env
        infra_graph = embedding.infra.graph
        node_to_index = dict()

        # add the nodes
        for (i, enode) in enumerate(embedding.graph.nodes()):
            inode = infra_graph.node[enode.node]
            node_to_index[enode] = i
            input_graph.add_node(
                i, features=np.array([inode["pos"][0], inode["pos"][1]])
            )

        # add the edges
        num_possbile = 0
        for (u, v, k, d) in embedding.graph.edges(data=True, keys=True):
            source_chosen = embedding.graph.nodes[u]["chosen"]
            u = node_to_index[u]
            v = node_to_index[v]
            chosen = d["chosen"]
            timeslot = d["timeslot"]
            possible = not chosen and source_chosen
            if possible:
                num_possbile += 1
            input_graph.add_edge(
                u,
                v,
                k,
                features=np.array(
                    [float(chosen), float(possible), float(timeslot)]
                ),
            )

        # no globals in input
        input_graph.graph["features"] = np.array([0.0])
        # return infra_graph
        from graph_nets import utils_np

        gt = utils_np.networkxs_to_graphs_tuple([input_graph])
        self.last_translation_dict = node_to_index
        return gt

    def _update_state(self):
        self.state = self._get_observation()._asdict()
        self.reward = self.last_result.get("reward", 0)
        self.done = self.last_result["done"]

    def _take_action(self, action_idx):
        (source, sink, timeslot) = self.actions[action_idx]
        ts_before = self.env.used_timeslots
        self.env.take_action(source, sink, timeslot)
        self._query_actions()

        complete = self.env.is_complete()
        failed = not complete and len(self.actions) == 0
        if failed:
            reward = -100
        else:
            reward = ts_before - self.env.used_timeslots
        self.last_result = {
            "reward": reward,
            "done": self.env.is_complete() or failed,
        }

    def _restart_environment_episode(self, force_environment_reset=False):
        self.env = random_embedding(20, np.random)
        self.last_result = {"reward": 0, "done": False}
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
