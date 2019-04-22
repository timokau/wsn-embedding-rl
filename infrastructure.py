import networkx as nx
import numpy as np
from enum import Enum

class NodeKind(Enum):
    source = 1
    sink = 2
    intermediate = 3


class InfrastructureNetwork():
    """Model of the physical network"""
    def __init__(self):
        self._last_id = 0

        self.graph = nx.Graph()

        self.sink = None
        self.sources = set()
        self.intermediates = set()

    def add_intermediate(
            self,
            pos: (float, float),
            name: str = None,
            *kwds,
    ):
        """Adds an intermediate node to the infrastructure graph"""
        node = self._add_node(pos, name, kind=NodeKind.intermediate)
        self.intermediates.add(node)

    def add_source(
            self,
            pos: (float, float),
            name: str = None,
    ):
        """Adds a source node to the infrastructure graph"""
        node = self._add_node(pos, name, kind=NodeKind.source)
        self.sources.add(node)

    def set_sink(
            self,
            pos: (float, float),
            name=None,
            *kwds,
    ):
        """Sets the node to the infrastructure graph"""
        node = self._add_node(pos, name, kind=NodeKind.sink)
        self.sink = node

    def _add_node(
        self,
        pos: (float, float),
        name: str,
        kind: NodeKind,
    ):
        if name is None:
            name = self._generate_name()

        self.graph.add_node(
            name,
            kind=kind,
            pos=pos,
        )
        return name

    def _generate_name(self):
        self._last_id += 1
        return f'N{self._last_id}'

def random_infrastructure(
        rand,
        min_nodes=2,
        max_nodes=10,
        num_sources=1,
):
    """
    Generates a randomized infrastructure with uniformly distributed
    nodes in 2d space.
    """
    assert num_sources < min_nodes

    # select a node count uniformly distributed over the given interval
    num_nodes = rand.randint(min_nodes, max_nodes)

    # place nodes uniformly at random
    node_positions = rand.uniform(
        size=(num_nodes, 2)
    )

    infra = InfrastructureNetwork()

    infra.set_sink(pos=node_positions[0])

    for source_pos in node_positions[1:num_sources+1]:
        infra.add_source(pos=source_pos)

    for node_pos in node_positions[num_sources+2:]:
        infra.add_intermediate(pos=node_pos)

    return infra


def draw_infra(
        infra: InfrastructureNetwork
):
    """Draws a given InfrastructureNetwork"""
    shared_args = {
        'G': infra.graph,
        'pos': nx.get_node_attributes(infra.graph, 'pos'),
        'node_size': 450,
    }
    nx.draw_networkx_nodes(
        nodelist=list(infra.sources),
        node_color='r',
        **shared_args,
    )
    nx.draw_networkx_nodes(
        nodelist=list(infra.intermediates),
        node_color='g',
        **shared_args,
    )
    nx.draw_networkx_nodes(
        nodelist=[infra.sink],
        node_color='y',
        **shared_args,
    )
    nx.draw_networkx_labels(
        **shared_args,
    )

if __name__ == "__main__":
    draw_infra(random_infrastructure(np.random))
    from matplotlib import pyplot as plt
    plt.show()
