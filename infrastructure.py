"""Modelling the physical network"""

from enum import Enum
from math import inf
import networkx as nx
import numpy as np
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
            result += f"  - {s}\n"
        result += f"- {len(self.intermediates)} intermediates:\n"
        for intermediate in self.intermediates:
            i = self._node_to_verbose_str(intermediate)
            result += f"  - {i}\n"
        result += "- one sink:\n"
        s = self._node_to_verbose_str(self.sink)
        result += f"  - {s}\n"
        return result

    def _node_to_verbose_str(self, node):
        pos = self.graph.nodes[node]["pos"]
        transmit_power_dbm = self.graph.nodes[node]["transmit_power_dbm"]
        return f"{node} at ({pos[0]}, {pos[1]}), {transmit_power_dbm}dBm"


def random_infrastructure(
    rand,
    min_nodes=2,
    max_nodes=10,
    num_sources=1,
    width=10,
    height=10,
    mean_capacity=10,
):
    """
    Generates a randomized infrastructure with uniformly distributed
    nodes in 2d space.
    """
    assert num_sources < min_nodes

    def rand_power():
        # FCC limit for a wifi router is 36dBm
        mean_transmit_power_dbm = 20
        return rand.normal(mean_transmit_power_dbm, 10)

    def rand_capacity():
        return rand.exponential(mean_capacity)

    # select a node count uniformly distributed over the given interval
    num_nodes = rand.randint(min_nodes, max_nodes + 1)

    # place nodes uniformly at random
    node_positions = rand.uniform(
        low=(0, 0), high=(width, height), size=(num_nodes, 2)
    )

    infra = InfrastructureNetwork()

    infra.set_sink(
        pos=node_positions[0],
        transmit_power_dbm=rand_power(),
        capacity=rand_capacity(),
    )

    for source_pos in node_positions[1 : num_sources + 1]:
        infra.add_source(
            pos=source_pos,
            transmit_power_dbm=rand_power(),
            capacity=rand_capacity(),
        )

    for node_pos in node_positions[num_sources + 2 :]:
        infra.add_intermediate(
            pos=node_pos,
            transmit_power_dbm=rand_power(),
            capacity=rand_capacity(),
        )

    return infra


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
    draw_infra(random_infrastructure(np.random))
    from matplotlib import pyplot as plt

    plt.show()
