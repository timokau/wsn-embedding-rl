"""Extract observations from embedding state"""

import networkx as nx
import numpy as np

from embedding import PartialEmbedding, ENode

POSSIBLE_IDX = 0
TIMESLOT_IDX = 1


def frac(a, b):
    """Regular fraction, but result is 0 if a = b = 0"""
    if a == 0 and b == 0:
        return 0
    return a / b


class ObservationBuilder:
    """Build a feature graph from a partial embedding"""

    def __init__(self, node_features, edge_features):
        self._node_features = node_features
        self._edge_features = edge_features

    def extract_node_features(self, embedding: PartialEmbedding, enode: ENode):
        """Build feature array for a single enode"""
        features = []
        for node_feature in self._node_features:
            features.extend(node_feature.compute(embedding, enode))
        return features

    def extract_edge_features(
        self,
        embedding: PartialEmbedding,
        source: ENode,
        target: ENode,
        timeslot: int,
        edge_data,
    ):
        """Build feature array for a single edge"""
        source_chosen = embedding.graph.nodes[source]["chosen"]
        chosen = edge_data["chosen"]
        capacity = embedding.known_capacity(source.node, target.node, timeslot)
        datarate_requirement = embedding.overlay.datarate(source.block)
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
            for (other_so, _other_ta) in embedding.taken_edges_in[timeslot]:
                if (
                    other_so.block == source.block
                    and other_so.node == source.node
                ):
                    is_broadcast = True
            features += [float(is_broadcast)]

        assert features[POSSIBLE_IDX] == float(possible)
        assert features[TIMESLOT_IDX] == float(timeslot)
        return features

    def get_observation(self, embedding: PartialEmbedding):
        """Extracts features from an embedding and returns a graph-nets
        compatible graph"""
        # This is a complex function, but I see no use in splitting it
        # up.
        # build graphs from scratch, since we need to change the node
        # indexing (graph_nets can only deal with integer indexed nodes)
        input_graph = nx.MultiDiGraph()
        node_to_index = dict()

        # add the nodes
        for (i, enode) in enumerate(embedding.graph.nodes()):
            node_to_index[enode] = i
            input_graph.add_node(
                i,
                features=self.extract_node_features(embedding, enode),
                represents=enode,
            )

        # add the edges
        for (u, v, k, d) in embedding.graph.edges(data=True, keys=True):
            input_graph.add_edge(
                node_to_index[u],
                node_to_index[v],
                k,
                features=self.extract_edge_features(embedding, u, v, k, d),
            )

        # no globals in input
        input_graph.graph["features"] = np.array([0.0])
        return input_graph
