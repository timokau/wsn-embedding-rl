"""Modelling the physical network"""

from enum import Enum
from math import inf
import numpy as np
import networkx as nx
import wsignal


class NodeKind(Enum):
    """Types of infrastructure nodes"""

    source = 1
    sink = 2
    intermediate = 3


class InfrastructureNetwork:
    """Model of the physical network"""

    def __init__(self, bandwidth=1):
        self._last_id = 0
        # Link capacity is influenced by the SINR and the bandwidth.
        # Leaving the bandwidth set to 1 will result in a link capacity
        # that is relative to the actual bandwidth (so capacity
        # requirements are in the format of
        # "b bits per second per bandwidth"
        self.bandwidth = bandwidth

        self.graph = nx.Graph()

        self.power_received_cache = dict()
        self.sink = None
        self.sources = set()
        self.intermediates = set()

    def nodes(self):
        """Returns all infrastructure nodes"""
        return self.graph.nodes()

    def add_intermediate(
        self,
        pos: (float, float),
        transmit_power_dbm: float,
        capacity: float = inf,
        name: str = None,
    ):
        """Adds an intermediate node to the infrastructure graph"""
        node = self._add_node(
            pos, transmit_power_dbm, NodeKind.intermediate, capacity, name
        )
        self.intermediates.add(node)
        return node

    def add_source(
        self,
        pos: (float, float),
        transmit_power_dbm: float,
        capacity: float = inf,
        name: str = None,
    ):
        """Adds a source node to the infrastructure graph"""
        node = self._add_node(
            pos, transmit_power_dbm, NodeKind.source, capacity, name
        )
        self.sources.add(node)
        return node

    def set_sink(
        self,
        pos: (float, float),
        transmit_power_dbm: float,
        capacity: float = inf,
        name=None,
    ):
        """Sets the node to the infrastructure graph"""
        node = self._add_node(
            pos, transmit_power_dbm, NodeKind.sink, capacity, name
        )
        self.sink = node
        return node

    def _add_node(
        self,
        pos: (float, float),
        transmit_power_dbm: float,
        kind: NodeKind,
        capacity: float,
        name: str = None,
    ):
        if name is None:
            name = self._generate_name()

        self.graph.add_node(
            name,
            kind=kind,
            pos=pos,
            capacity=capacity,
            transmit_power_dbm=transmit_power_dbm,
        )
        return name

    def capacity(self, node):
        """Returns the capacity of a given node"""
        return self.graph.node[node]["capacity"]

    def position(self, node):
        """Returns the position of a given node"""
        return self.graph.node[node]["pos"]

    def power(self, node):
        """Returns the transmit power of a given node"""
        return self.graph.node[node]["transmit_power_dbm"]

    def min_node_distance(self):
        """Calculates the distance between the closest nodes"""
        min_distance = inf
        for a in self.nodes():
            x1, y1 = self.position(a)
            for b in self.nodes():
                if b == a:
                    continue
                x2, y2 = self.position(b)
                dist = wsignal.distance(x1, y1, x2, y2)
                if dist < min_distance:
                    min_distance = dist
        return min_distance

    def power_received_dbm(self, source, target):
        """Power received at sink if source sends at full power"""
        cached = self.power_received_cache.get((source, target))
        if cached is None:
            source_node = self.graph.nodes[source]
            target_node = self.graph.nodes[target]
            src_x, src_y = source_node["pos"]
            trg_x, trg_y = target_node["pos"]
            distance = wsignal.distance(src_x, src_y, trg_x, trg_y)
            transmit_power_dbm = source_node["transmit_power_dbm"]
            cached = wsignal.power_received(distance, transmit_power_dbm)
            self.power_received_cache[(source, target)] = cached
        return cached

    def _generate_name(self):
        self._last_id += 1
        return f"N{self._last_id}"

    def __str__(self):
        result = "Infrastructure with:\n"
        result += f"- {len(self.sources)} sources:\n"
        for source in self.sources:
            s = self._node_to_verbose_str(source)
            result += f"  - add_source{s}\n"
        result += f"- {len(self.intermediates)} intermediates:\n"
        for intermediate in self.intermediates:
            i = self._node_to_verbose_str(intermediate)
            result += f"  - add_intermediate{i}\n"
        result += "- one sink:\n"
        s = self._node_to_verbose_str(self.sink)
        result += f"  - set_sink{s}\n"
        return result

    def _node_to_verbose_str(self, node):
        pos = self.graph.nodes[node]["pos"]
        pos = f"({pos[0]}, {pos[1]})"
        tp = self.graph.nodes[node]["transmit_power_dbm"]
        return f'(name="{node}", pos={pos}, transmit_power_dbm={tp})'


def draw_infra(
    infra: InfrastructureNetwork,
    sources_color="red",
    sink_color="yellow",
    intermediates_color="green",
):
    """Draws a given InfrastructureNetwork"""
    shared_args = {
        "G": infra.graph,
        "pos": nx.get_node_attributes(infra.graph, "pos"),
        "node_size": 450,
    }
    nx.draw_networkx_nodes(
        nodelist=list(infra.sources), node_color=sources_color, **shared_args
    )
    nx.draw_networkx_nodes(
        nodelist=list(infra.intermediates),
        node_color=intermediates_color,
        **shared_args,
    )
    nx.draw_networkx_nodes(
        nodelist=[infra.sink], node_color=sink_color, **shared_args
    )
    nx.draw_networkx_labels(**shared_args)


if __name__ == "__main__":
    from generator import random_infrastructure

    draw_infra(random_infrastructure(2, rand=np.random))
    from matplotlib import pyplot as plt

    plt.show()
