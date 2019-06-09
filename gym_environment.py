"""Gym environment wrapper for a Wireless Sensor Network"""

import numpy as np
import networkx as nx
import tensorflow as tf
import gym

from graph_nets import utils_np, utils_tf
from graph_nets.graphs import GraphsTuple

from generator import random_embedding


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
        result = utils_tf._build_placeholders_from_specs(
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

        def make_result_feed_dict(val):
            if isinstance(val, GraphsTuple):
                graphs_tuple = val
            else:
                dicts = []
                for graphs_tuple in val:
                    dicts.append(
                        utils_np.graphs_tuple_to_data_dicts(graphs_tuple)[0]
                    )
                graphs_tuple = utils_np.data_dicts_to_graphs_tuple(dicts)
            return utils_tf.get_feed_dict(result, graphs_tuple)

        result.make_feed_dict = make_result_feed_dict
        result.name = "TEST_NAME"
        return result


class WSNEnvironment(gym.Env):
    """Wireless Sensor Network Environment"""

    # That is what reset ist for.
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, max_embedding_size=50):
        self.max_embedding_size = max_embedding_size
        self.observation_space = GraphSpace(
            global_dim=1, node_dim=2, edge_dim=3
        )
        self.reset()

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

        gt = utils_np.networkxs_to_graphs_tuple([input_graph])

        self.last_translation_dict = node_to_index
        return gt

    def step(self, action):
        (source, sink, timeslot) = self.actions[action]
        ts_before = self.env.used_timeslots
        self.env.take_action(source, sink, timeslot)
        self._query_actions()

        reward = ts_before - self.env.used_timeslots
        ob = self._get_observation()
        done = self.env.is_complete()

        return ob, reward, done, {}

    @property
    def action_space(self):
        """Return the dynamic action space, may change with each step"""
        result = gym.spaces.Discrete(len(self.actions))
        return result

    def reset(self):
        self.env = random_embedding(self.max_embedding_size, np.random)
        self.last_translation_dict = dict()
        self._query_actions()
        return self._get_observation()

    def render(self, mode="human"):
        raise NotImplementedError()