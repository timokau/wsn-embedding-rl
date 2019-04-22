import networkx as nx
import numpy as np
from enum import Enum

class BlockKind(Enum):
    source = 1
    sink = 2
    intermediate = 3

class OverlayNetwork():
    """Model of the overlay network"""
    def __init__(self):
        self.graph = nx.DiGraph()
        self._last_id = 0

        self.sources = set()
        self.intermediates = set()
        self.sink = None

    def add_source(
            self,
            name=None
    ):
        block = self._add_block(
            name,
            BlockKind.source
        )
        self.sources.add(block)

    def add_intermediate(
            self,
            name=None
    ):
        block = self._add_block(
            name,
            BlockKind.intermediate
        )
        self.intermediates.add(block)

    def set_sink(
            self,
            name=None
    ):
        block = self._add_block(
            name,
            BlockKind.sink
        )
        self.sink = block

    def _add_block(
            self,
            name,
            kind: BlockKind,
    ):
        """Adds a block to the overlay network"""

        if name is None:
            name = self._generate_name()
        self.graph.add_node(
            name,
            kind=kind,
        )
        return name

    def _generate_name(self):
        self._last_id += 1
        return f'B{self._last_id}'

    def add_link(
            self,
            source: str,
            sink: str,
    ):
        """Adds a link between two blocks in the overlay network"""
        self.graph.add_edge(source, sink)

def random_overlay(
        rand,
        pairwise_connection_prob=0.02,
        min_blocks=2,
        max_blocks=10,
        num_sources=1,
):
    """Generates a randomized overlay graph."""
    assert num_sources < min_blocks

    # select a block count uniformly distributed over the given interval
    num_blocks = rand.randint(min_blocks, max_blocks)

    overlay = OverlayNetwork()

    overlay.set_sink()

    for _ in range(num_sources):
        overlay.add_source()

    for _ in range(num_blocks - num_sources - 1):
        overlay.add_intermediate()

    for source in overlay.graph.nodes():
        for sink in overlay.graph.nodes():
            if rand.random() < pairwise_connection_prob:
                overlay.add_link(source, sink)

    accessible_from_source = set()
    not_accessible_from_source = set()
    has_path_to_sink = set()
    no_path_to_sink = set()
    # TODO optimize
    for node in overlay.graph.nodes():
        if nx.has_path(overlay.graph, node, overlay.sink):
            has_path_to_sink.add(node)
        else:
            no_path_to_sink.add(node)

        source_path_found = False
        for source in overlay.sources:
            if nx.has_path(overlay.graph, source, node):
                source_path_found = True
                break
        if source_path_found:
            accessible_from_source.add(node)
        else:
            not_accessible_from_source.add(node)

    for node in not_accessible_from_source:
        connection = rand.choice(list(accessible_from_source))
        overlay.add_link(connection, node)
        accessible_from_source.add(node)

    for node in no_path_to_sink:
        connection = rand.choice(list(has_path_to_sink))
        overlay.add_link(node, connection)
        has_path_to_sink.add(node)

    return overlay

def draw_overlay(
        overlay: OverlayNetwork,
):
    """Draws a given OverlayNetwork"""
    shared_args = {
        'G': overlay.graph,
        'pos': nx.spring_layout(overlay.graph),
        'node_size': 450,
        'node_shape': 's',
    }
    nx.draw_networkx_nodes(
        nodelist=list(overlay.sources),
        node_color='red',
        **shared_args,
    )
    nx.draw_networkx_nodes(
        nodelist=list(overlay.intermediates),
        node_color='green',
        **shared_args,
    )
    nx.draw_networkx_nodes(
        nodelist=[overlay.sink],
        node_color='yellow',
        **shared_args,
    )
    nx.draw_networkx_labels(
        **shared_args,
    )
    nx.draw_networkx_edges(
        **shared_args,
    )

    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    red_patch = mpatches.Patch(color='red', label='sources')
    yellow_patch = mpatches.Patch(color='yellow', label='sink')
    green_patch = mpatches.Patch(color='green', label='intermediates')
    plt.legend(handles=[red_patch, yellow_patch, green_patch])

if __name__ == "__main__":
    draw_overlay(random_overlay(np.random))
    from matplotlib import pyplot as plt
    plt.show()
