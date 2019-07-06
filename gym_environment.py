"""Gym environment wrapper for a Wireless Sensor Network"""

import multiprocessing
from queue import Queue

import numpy as np
import networkx as nx
import tensorflow as tf
import gym

from graph_nets import utils_np, utils_tf
from graph_nets.graphs import GraphsTuple

import generator

QUEUE_SIZE = 128


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

    def __init__(self):
        self.observation_space = GraphSpace(
            global_dim=1, node_dim=3, edge_dim=4
        )

        # optimize reset
        self._instance_queue = Queue(QUEUE_SIZE)
        self._pool = None

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
        index_to_node = dict()

        # add the nodes
        for (i, enode) in enumerate(embedding.graph.nodes()):
            inode = infra_graph.node[enode.node]
            node_to_index[enode] = i
            index_to_node[i] = enode
            input_graph.add_node(
                i,
                features=np.array(
                    [inode["pos"][0], inode["pos"][1], float(enode.relay)]
                ),
            )

        # add the edges
        for (u, v, k, d) in embedding.graph.edges(data=True, keys=True):
            source_chosen = embedding.graph.nodes[u]["chosen"]
            chosen = d["chosen"]
            timeslot = d["timeslot"]
            capacity = embedding.known_capacity(u.node, v.node, timeslot)
            possible = not chosen and source_chosen

            input_graph.add_edge(
                node_to_index[u],
                node_to_index[v],
                k,
                features=np.array(
                    [float(chosen), float(possible), float(timeslot), capacity]
                ),
            )

        # no globals in input
        input_graph.graph["features"] = np.array([0.0])
        # return infra_graph

        gt = utils_np.networkxs_to_graphs_tuple([input_graph])

        # build action indices here to make sure the indices matches the
        # one the network is seeing
        self.actions = []
        for (u, v, d) in zip(gt.senders, gt.receivers, gt.edges):
            possible = d[1] == 1
            if not possible:
                continue
            else:
                source = index_to_node[u]
                target = index_to_node[v]
                timeslot = int(d[2])
                self.actions.append((source, target, timeslot))

        self.last_translation_dict = node_to_index
        return gt

    def step(self, action):
        (source, sink, timeslot) = self.actions[action]
        ts_before = self.env.used_timeslots
        self.env.take_action(source, sink, timeslot)

        reward = ts_before - self.env.used_timeslots
        done = self.env.is_complete()

        if not done and len(self.env.possibilities()) == 0:
            # Failed to solve the problem, retry without ending the
            # episode (thus penalizing the failed attempt).
            self.env = self.env.reset()
            self.restarts += 1
            # make it easier for the network to figure out that resets
            # are bad
            reward -= 10

        ob = self._get_observation()

        return ob, reward, done, {}

    @property
    def action_space(self):
        """Return the dynamic action space, may change with each step"""
        result = gym.spaces.Discrete(len(self.actions))
        return result

    def _new_instance(self):
        """Transparently uses multiprocessing"""
        if self._pool is None:
            # reserver one cpu for the actual training
            cpus = max(1, multiprocessing.cpu_count() - 1)
            cpus = min(8, cpus)
            self._pool = multiprocessing.Pool(cpus)

        # similar to a lazy infinite imap; preserves the order of the
        # generated elements to prevent under-representation of long
        # running ones.
        while not self._instance_queue.full():
            rand = np.random.RandomState(np.random.randint(0, 2 ** 32))
            job = self._pool.map_async(generator.validated_random, [rand])
            self._instance_queue.put_nowait(job)

        next_job = self._instance_queue.get()
        if not next_job.ready():
            print("Blocked on queue")
        return next_job.get()[0]

    # optional argument is fine
    # pylint: disable=arguments-differ
    def reset(self, embedding=None):
        self.baseline = None
        if embedding is None:
            (embedding, baseline) = self._new_instance()
            self.baseline = baseline
        self.env = embedding
        self.restarts = 0
        self.last_translation_dict = dict()
        return self._get_observation()

    def render(self, mode="human"):
        raise NotImplementedError()
