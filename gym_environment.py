"""Gym environment wrapper for a Wireless Sensor Network"""

import tensorflow as tf
import gym

from graph_nets import utils_np, utils_tf
from graph_nets.graphs import GraphsTuple

from observation import ObservationBuilder, TIMESLOT_IDX, POSSIBLE_IDX


class GraphSpace(gym.spaces.Space):
    """Graph space for usage with graph_nets"""

    def __init__(self, global_dim, node_dim, edge_dim):
        super().__init__()
        self.global_dim = global_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim

    def contains(self, x):
        raise NotImplementedError()

    def to_placeholders(self, batch_size=None):
        """Creates a placeholder to be fed into a graph_net"""
        # pylint: disable=protected-access
        placeholders = utils_tf._build_placeholders_from_specs(
            dtypes=GraphsTuple(
                nodes=tf.float64,
                edges=tf.float64,
                receivers=tf.int32,
                senders=tf.int32,
                globals=tf.float64,
                n_node=tf.int32,
                n_edge=tf.int32,
            ),
            shapes=GraphsTuple(
                nodes=[batch_size, self.node_dim],
                edges=[batch_size, self.edge_dim],
                receivers=[batch_size],
                senders=[batch_size],
                globals=[batch_size, self.global_dim],
                n_node=[batch_size],
                n_edge=[batch_size],
            ),
        )

        def make_feed_dict(val):
            if isinstance(val, GraphsTuple):
                graphs_tuple = val
            else:
                dicts = []
                for graphs_tuple in val:
                    dicts.append(
                        utils_np.graphs_tuple_to_data_dicts(graphs_tuple)[0]
                    )
                graphs_tuple = utils_np.data_dicts_to_graphs_tuple(dicts)
            return utils_tf.get_feed_dict(placeholders, graphs_tuple)

        placeholders.make_feed_dict = make_feed_dict
        placeholders.name = "Graph observation placeholder"
        return placeholders


class WSNEnvironment(gym.Env):
    """Wireless Sensor Network Environment"""

    # That is what reset ist for.
    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, problem_generator, features, early_exit_factor, seedgen
    ):
        self.problem_generator = problem_generator
        self._features = features

        node_dim = sum([feature.node_dim for feature in self._features])
        # always has to include "possible" bit
        edge_dim = 1 + sum([feature.edge_dim for feature in self._features])
        self.observation_space = GraphSpace(
            global_dim=1, node_dim=node_dim, edge_dim=edge_dim
        )

        self.seedgen = seedgen
        self.early_exit_factor = early_exit_factor

    def _get_observation(self):
        graph = ObservationBuilder(features=self._features).get_observation(
            self.env
        )

        gt = utils_np.networkxs_to_graphs_tuple([graph])

        # build action indices here to make sure the indices matches the
        # one the network is seeing
        self.actions = []
        edges = gt.edges if gt.edges is not None else []  # may be None
        for (u, v, d) in zip(gt.senders, gt.receivers, edges):
            possible = d[POSSIBLE_IDX] == 1
            if not possible:
                continue
            else:
                source = graph.node[u]["represents"]
                target = graph.node[v]["represents"]
                timeslot = d[TIMESLOT_IDX]
                self.actions.append((source, target, timeslot))

        return gt

    def step(self, action):
        (source, sink, timeslot) = self.actions[action]
        ts_before = self.env.used_timeslots
        assert self.env.take_action(source, sink, timeslot)

        reward = ts_before - self.env.used_timeslots
        self.total_reward += reward
        done = self.env.is_complete()

        blocks = len(self.env.overlay.blocks())
        links = len(self.env.overlay.links())
        nodes = len(self.env.infra.nodes())
        bl = self.baseline

        if not done and len(self.env.possibilities()) == 0:
            # Avoid getting stuck on difficult/impossible problems,
            # especially in the beginning. It is important not to do
            # this too early, since otherwise the agent could learn to
            # strategically fail. It should be possible to solve every
            # solvable problem in (node*links) timesteps though, so
            # anything bigger than that should be fine.
            min_reward = -self.early_exit_factor * nodes * links
            if self.total_reward < min_reward:
                print("Early exit")
                done = True
        if not done and len(self.env.possibilities()) == 0:
            # Failed to solve the problem, retry without ending the
            # episode (thus penalizing the failed attempt).
            embedded_links = len(self.env.finished_embeddings)
            ts_used = self.env.used_timeslots

            if bl is not None:
                print(
                    "RESET; "
                    f"n{nodes}b{blocks}, "
                    f"{embedded_links}/{links} done "
                    f"in {ts_used}ts ({bl})"
                )

            self.env = self.env.reset()
            self.restarts += 1
            # make it easier for the network to figure out that resets
            # are bad
            reward -= 10

        self._last_ob = self._get_observation()

        return self._last_ob, reward, done, {}

    @property
    def action_space(self):
        """Return the dynamic action space, may change with each step"""
        result = gym.spaces.Discrete(len(self.actions))
        return result

    # optional argument is fine
    # pylint: disable=arguments-differ
    def reset(self, embedding=None):
        self.baseline = None
        if embedding is None:
            (embedding, baseline) = self.problem_generator()
            self.baseline = baseline
        self.env = embedding
        self.restarts = 0
        self.total_reward = 0
        self._last_ob = self._get_observation()
        return self._last_ob

    def render(self, mode="human"):
        raise NotImplementedError()
