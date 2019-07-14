"""Gym environment wrapper for a Wireless Sensor Network"""

import numpy as np
import networkx as nx
import tensorflow as tf
import gym

from graph_nets import utils_np, utils_tf
from graph_nets.graphs import GraphsTuple

POSSIBLE_IDX = 0
TIMESLOT_IDX = 1


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


SUPPORTED_NODE_FEATURES = frozenset(
    (
        "posx",
        "posy",
        "relay",
        "sink",
        "remaining_capacity",
        "requirement",
        "compute_fraction",
        "unembedded_blocks_embeddable_after",
    )
)
SUPPORTED_EDGE_FEATURES = frozenset(
    (
        "timeslot",
        "chosen",
        "capacity",
        "datarate_requirement",
        "datarate_fraction",
        "additional_timeslot",
    )
)


def frac(a, b):
    """Regular fraction, but result is 0 if a = b = 0"""
    if a == 0 and b == 0:
        return 0
    return a / b


class WSNEnvironment(gym.Env):
    """Wireless Sensor Network Environment"""

    # That is what reset ist for.
    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        problem_generator,
        node_features=SUPPORTED_NODE_FEATURES,
        edge_features=SUPPORTED_EDGE_FEATURES,
        early_exit_factor=np.infty,
        seedgen=lambda: np.random.randint(0, 2 ** 32),
    ):
        self.problem_generator = problem_generator
        self._node_features = node_features
        self._edge_features = edge_features
        assert set(self._node_features).issubset(SUPPORTED_NODE_FEATURES)
        assert set(self._edge_features).issubset(SUPPORTED_EDGE_FEATURES)

        node_dim = len(self._node_features)
        # always has to include "possible" bit
        edge_dim = 1 + len(self._edge_features)
        self.observation_space = GraphSpace(
            global_dim=1, node_dim=node_dim, edge_dim=edge_dim
        )

        self.seedgen = seedgen
        self.early_exit_factor = early_exit_factor

    def _get_observation(self):
        # This is a complex function, but I see no use in splitting it
        # up.
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements
        # pylint: disable=too-many-branches
        # build graphs from scratch, since we need to change the node
        # indexing (graph_nets can only deal with integer indexed nodes)
        input_graph = nx.MultiDiGraph()
        embedding = self.env
        infra_graph = embedding.infra.graph
        node_to_index = dict()
        index_to_node = dict()

        # add the nodes
        for (i, enode) in enumerate(embedding.graph.nodes()):
            inode = infra_graph.node[enode.node]
            node_to_index[enode] = i
            index_to_node[i] = enode
            is_sink = enode.node == self.env.infra.sink
            requirement = embedding.overlay.requirement(enode.block)
            remaining = embedding.remaining_capacity(enode.node)

            features = []
            if "posx" in self._node_features:
                features += [inode["pos"][0]]
            if "posy" in self._node_features:
                features += [inode["pos"][1]]
            if "relay" in self._node_features:
                features += [float(enode.relay)]
            if "sink" in self._node_features:
                features += [float(is_sink)]
            if "remaining_capacity" in self._node_features:
                features += [remaining]
            if "requirement" in self._node_features:
                features += [requirement]
            if "compute_fraction" in self._node_features:
                features += [frac(requirement, remaining)]
            if "unembedded_blocks_embeddable_after" in self._node_features:
                embedded = set(embedding.taken_embeddings.keys()).union(
                    [enode.block]
                )
                options = set(embedding.overlay.blocks()).difference(embedded)
                remaining_after = remaining - requirement
                embeddable = {
                    option
                    for option in options
                    if embedding.overlay.requirement(option) < remaining_after
                }
                nropt = len(options)
                fraction = len(embeddable) / nropt if nropt > 0 else 1
                features += [fraction]

            input_graph.add_node(i, features=np.array(features))

        # add the edges
        for (u, v, k, d) in embedding.graph.edges(data=True, keys=True):
            source_chosen = embedding.graph.nodes[u]["chosen"]
            chosen = d["chosen"]
            timeslot = d["timeslot"]
            capacity = embedding.known_capacity(u.node, v.node, timeslot)
            datarate_requirement = embedding.overlay.datarate(u.block)
            possible = not chosen and source_chosen
            features = [float(possible)]
            additional_timeslot = timeslot >= self.env.used_timeslots

            if "timeslot" in self._edge_features:
                features += [float(timeslot)]
            if "chosen" in self._edge_features:
                features += [float(chosen)]
            if "capacity" in self._edge_features:
                features += [capacity]
            if "additional_timeslot" in self._edge_features:
                features += [float(additional_timeslot)]
            if "datarate_requirement" in self._edge_features:
                features += [datarate_requirement]
            if "datarate_fraction" in self._edge_features:
                features += [frac(datarate_requirement, capacity)]

            assert features[POSSIBLE_IDX] == float(possible)
            assert features[TIMESLOT_IDX] == float(timeslot)

            input_graph.add_edge(
                node_to_index[u],
                node_to_index[v],
                k,
                features=np.array(features),
            )

        # no globals in input
        input_graph.graph["features"] = np.array([0.0])

        gt = utils_np.networkxs_to_graphs_tuple([input_graph])

        # build action indices here to make sure the indices matches the
        # one the network is seeing
        self.actions = []
        edges = gt.edges if gt.edges is not None else []  # may be None
        for (u, v, d) in zip(gt.senders, gt.receivers, edges):
            possible = d[POSSIBLE_IDX] == 1
            if not possible:
                continue
            else:
                source = index_to_node[u]
                target = index_to_node[v]
                timeslot = int(d[TIMESLOT_IDX])
                self.actions.append((source, target, timeslot))

        self.last_translation_dict = node_to_index
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
        self.last_translation_dict = dict()
        self._last_ob = self._get_observation()
        return self._last_ob

    def render(self, mode="human"):
        raise NotImplementedError()
