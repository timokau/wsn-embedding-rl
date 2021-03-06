"""Modelling the physical network"""

from typing import FrozenSet
from functools import lru_cache
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

    # pylint: disable=too-many-instance-attributes
    # Instance attributes needed for caching, I think private instance
    # attributes are fine.

    def __init__(self, bandwidth=1, noise_floor_dbm: float = -30):
        self._last_id = 0
        # Link capacity is influenced by the SINR and the bandwidth.
        # Leaving the bandwidth set to 1 will result in a link capacity
        # that is relative to the actual bandwidth (so capacity
        # requirements are in the format of
        # "b bits per second per bandwidth"
        self.bandwidth = bandwidth
        # https://www.quora.com/How-high-is-the-ambient-RF-noise-floor-in-the-2-4-GHz-spectrum-in-downtown-San-Francisco
        self.noise_floor_dbm = noise_floor_dbm

        self.graph = nx.Graph()

        self.sink = None
        self.sources = set()
        self.intermediates = set()

        # transparent caching per instance
        self.power_at_node = self._power_at_node
        self.sinr = lru_cache(1)(self._sinr)
        self.power_at_node = lru_cache(1)(self._power_at_node)
        self.power_received_dbm = lru_cache(None)(self._power_received_dbm)

    def _reset_caches(self):
        nodes = len(self.nodes())

        if hasattr(self, "sinr"):
            self.sinr.cache_clear()
        # Enough space for all pairwise SINRs for 20 different
        # configurations of sending nodes. Most relevant is the "no
        # sending nodes" case, which will happen all the time.
        sinr_maxsize = min(20 * nodes ** 2, 10 * 1024)  # upper bound
        self.sinr = lru_cache(maxsize=sinr_maxsize)(self._sinr)

        if hasattr(self, "power_at_node"):
            self.power_at_node.cache_clear()
        self.power_at_node = lru_cache(maxsize=100 * nodes)(
            self._power_at_node
        )

        if hasattr(self, "power_received_dbm"):
            self.power_received_dbm.cache_clear()
        self.power_received_dbm = lru_cache(maxsize=nodes)(
            self._power_received_dbm
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # don't pickle caches
        del state["sinr"]
        del state["power_at_node"]
        del state["power_received_dbm"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._reset_caches()

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
        self._reset_caches()
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

    def _power_received_dbm(self, source, target):
        """Power received at sink if source sends at full power"""
        source_node = self.graph.nodes[source]
        target_node = self.graph.nodes[target]
        src_x, src_y = source_node["pos"]
        trg_x, trg_y = target_node["pos"]
        distance = wsignal.distance(src_x, src_y, trg_x, trg_y)
        transmit_power_dbm = source_node["transmit_power_dbm"]
        return wsignal.power_received(distance, transmit_power_dbm)

    def _power_at_node(self, node: str, senders: FrozenSet[str]):
        """Calculates the amount of power a node receives (signal+noise)
        assuming only `senders` sends"""
        # We need to convert to watts for addition (log scale can only
        # multiply)
        received_power_watt = 0
        for sender in senders:
            p_r = self.power_received_dbm(sender, node)
            received_power_watt += wsignal.dbm_to_watt(p_r)

        return wsignal.watt_to_dbm(received_power_watt)

    def _sinr(self, source: str, target: str, senders: FrozenSet[str]):
        """
        SINR assuming only `senders` are sending.
        """
        received_signal_dbm = self.power_received_dbm(source, target)

        # everything already sending is assumed to be interference
        received_interference_dbm = self.power_at_node(target, senders=senders)

        return wsignal.sinr(
            received_signal_dbm,
            received_interference_dbm,
            self.noise_floor_dbm,
        )

    def _generate_name(self):
        self._last_id += 1
        return f"N{self._last_id}"

    def __str__(self):
        result = "infra = InfrastructureNetwork():\n"
        for source in self.sources:
            s = self._node_to_verbose_str(source)
            result += f"{source} = infra.add_source{s}\n"
        for intermediate in self.intermediates:
            i = self._node_to_verbose_str(intermediate)
            result += f"{intermediate} = infra.add_intermediate{i}\n"
        s = self._node_to_verbose_str(self.sink)
        result += f"{self.sink} = infra.set_sink{s}\n"
        return result

    def _node_to_verbose_str(self, node):
        pos = self.graph.nodes[node]["pos"]
        pos = f"({round(pos[0], 1)}, {round(pos[1], 1)})"
        tp = round(self.graph.nodes[node]["transmit_power_dbm"], 1)
        cap = round(self.capacity(node), 1)
        return (
            f'(name="{node}", '
            f"pos={pos}, "
            f"transmit_power_dbm={tp}, "
            f"capacity={cap})"
        )


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
    from generator import DefaultGenerator

    draw_infra(DefaultGenerator().random_infrastructure(2, rand=np.random))
    from matplotlib import pyplot as plt

    plt.show()
