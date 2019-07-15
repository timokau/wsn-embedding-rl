"""Extract observations from embedding state"""

import networkx as nx
import numpy as np

import gym_environment
from embedding import PartialEmbedding


def frac(a, b):
    """Regular fraction, but result is 0 if a = b = 0"""
    if a == 0 and b == 0:
        return 0
    return a / b


class ObservationBuilder:
    """Build a feature graph from a partial embedding"""

    # pylint: disable=too-few-public-methods
    def __init__(self, node_features, edge_features):
        self._node_features = node_features
        self._edge_features = edge_features

    def get_observation(self, embedding: PartialEmbedding):
        """Extracts features from an embedding and returns a graph-nets
        compatible graph"""
        # This is a complex function, but I see no use in splitting it
        # up.
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements
        # pylint: disable=too-many-branches
        # build graphs from scratch, since we need to change the node
        # indexing (graph_nets can only deal with integer indexed nodes)
        infra = embedding.infra
        overlay = embedding.overlay
        infra_graph = infra.graph

        input_graph = nx.MultiDiGraph()
        node_to_index = dict()
        index_to_node = dict()

        # add the nodes
        for (i, enode) in enumerate(embedding.graph.nodes()):
            inode = infra_graph.node[enode.node]
            node_to_index[enode] = i
            index_to_node[i] = enode
            is_sink = enode.node == infra.sink
            requirement = overlay.requirement(enode.block)
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
                options = set(overlay.blocks()).difference(embedded)
                remaining_after = remaining - requirement
                embeddable = {
                    option
                    for option in options
                    if overlay.requirement(option) < remaining_after
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
            datarate_requirement = overlay.datarate(u.block)
            possible = not chosen and source_chosen
            features = [float(possible)]
            additional_timeslot = timeslot >= embedding.used_timeslots

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
            if "is_broadcast" in self._edge_features:
                is_broadcast = False
                for (source, _target) in embedding.taken_edges_in[timeslot]:
                    if source.block == u.block and source.node == u.node:
                        is_broadcast = True
                features += [float(is_broadcast)]

            assert features[gym_environment.POSSIBLE_IDX] == float(possible)
            assert features[gym_environment.TIMESLOT_IDX] == float(timeslot)

            input_graph.add_edge(
                node_to_index[u],
                node_to_index[v],
                k,
                features=np.array(features),
            )

        # no globals in input
        input_graph.graph["features"] = np.array([0.0])
        return (input_graph, node_to_index, index_to_node)
