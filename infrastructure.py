"""Modelling the physical network"""

from enum import Enum
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

    def __init__(self):
        self._last_id = 0

        self.graph = nx.Graph()

        self.power_received_cache = dict()
        self.sink = None
        self.sources = set()
        self.intermediates = set()

    def nodes(self):
        """Returns all infrastructure nodes"""
        return self.graph.nodes()

    def add_intermediate(
        self, pos: (float, float), transmit_power_dbm: float, name: str = None
    ):
        """Adds an intermediate node to the infrastructure graph"""
        node = self._add_node(
            pos, transmit_power_dbm, NodeKind.intermediate, name
        )
        self.intermediates.add(node)
        return node

    def add_source(
        self, pos: (float, float), transmit_power_dbm: float, name: str = None
    ):
        """Adds a source node to the infrastructure graph"""
        node = self._add_node(pos, transmit_power_dbm, NodeKind.source, name)
        self.sources.add(node)
        return node

    def set_sink(
        self, pos: (float, float), transmit_power_dbm: float, name=None
    ):
        """Sets the node to the infrastructure graph"""
        node = self._add_node(pos, transmit_power_dbm, NodeKind.sink, name)
        self.sink = node
        return node

    def _add_node(
        self,
        pos: (float, float),
        transmit_power_dbm: float,
        kind: NodeKind,
        name: str = None,
    ):
        if name is None:
            name = self._generate_name()

        self.graph.add_node(
            name, kind=kind, pos=pos, transmit_power_dbm=transmit_power_dbm
        )
        return name

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
    rand, min_nodes=2, max_nodes=10, num_sources=1, width=10, height=10
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

    # select a node count uniformly distributed over the given interval
    num_nodes = rand.randint(min_nodes, max_nodes + 1)

    # place nodes uniformly at random
    node_positions = rand.uniform(
        low=(0, 0), high=(width, height), size=(num_nodes, 2)
    )

    infra = InfrastructureNetwork()

    infra.set_sink(pos=node_positions[0], transmit_power_dbm=rand_power())

    for source_pos in node_positions[1 : num_sources + 1]:
        infra.add_source(pos=source_pos, transmit_power_dbm=rand_power())

    for node_pos in node_positions[num_sources + 2 :]:
        infra.add_intermediate(pos=node_pos, transmit_power_dbm=rand_power())

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
