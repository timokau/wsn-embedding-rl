"""Modelling the overlay network"""

from enum import Enum
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np


class BlockKind(Enum):
    """Types of overlay nodes"""

    source = 1
    sink = 2
    intermediate = 3


class OverlayNetwork:
    """Model of the overlay network"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._last_id = 0

        self.sources = set()
        self.intermediates = set()
        self.sink = None

    def blocks(self):
        """Returns all blocks of the overlay"""
        return self.graph.nodes()

    def links(self):
        """Returns all links of the overlay"""
        return self.graph.edges()

    def add_source(self, requirement=0, name=None):
        """Adds a new source node to the overlay and returns it"""
        block = self._add_block(name, BlockKind.source, requirement)
        self.sources.add(block)
        return block

    def add_intermediate(self, requirement=0, name=None):
        """Adds a new intermediate node to the overlay and returns it"""
        block = self._add_block(name, BlockKind.intermediate, requirement)
        self.intermediates.add(block)
        return block

    def set_sink(self, requirement=0, name=None):
        """Creates a new node, sets it as the sink node and returns it"""
        block = self._add_block(name, BlockKind.sink, requirement)
        self.sink = block
        return block

    def _add_block(self, name, kind: BlockKind, requirement):
        """Adds a block to the overlay network"""

        if name is None:
            name = self._generate_name()
        self.graph.add_node(name, kind=kind, requirement=requirement)
        return name

    def requirement(self, block):
        """Returns the resource requirement of a given block"""
        return self.graph.node[block]["requirement"]

    def _generate_name(self):
        self._last_id += 1
        return f"B{self._last_id}"

    def add_link(self, source: str, sink: str, sinrth=2.0):
        """Adds a link between two blocks in the overlay network"""
        self.graph.add_edge(source, sink, sinrth=sinrth)

    def _block_to_verbose_str(self, block):
        out_edge_strings = []
        for (_, v) in self.graph.out_edges(nbunch=[block]):
            out_edge_strings.append(f"-> {v}")
        oe = ", ".join(out_edge_strings)
        return f"{block} ({oe})"

    def __str__(self):
        result = "Overlay with:\n"
        result += f"- {len(self.sources)} sources:\n"
        for source in self.sources:
            s = self._block_to_verbose_str(source)
            result += f"  - {s}\n"
        result += f"- {len(self.intermediates)} intermediates:\n"
        for intermediate in self.intermediates:
            i = self._block_to_verbose_str(intermediate)
            result += f"  - {i}\n"
        result += f"- one sink:\n"
        s = self._block_to_verbose_str(self.sink)
        result += f"  - {s}\n"
        return result


def random_overlay(
    rand,
    pairwise_connection_prob=0.02,
    min_blocks=2,
    max_blocks=10,
    num_sources=1,
    mean_requirement=5,
):
    """Generates a randomized overlay graph."""
    # This is a complicated function, but it would only get harder to
    # understand when split up into multiple single-use functions.
    # pylint: disable=too-many-branches
    assert num_sources < min_blocks

    def rand_requirement():
        return rand.exponential(mean_requirement)

    # select a block count uniformly distributed over the given interval
    num_blocks = rand.randint(min_blocks, max_blocks + 1)

    overlay = OverlayNetwork()

    overlay.set_sink(requirement=rand_requirement())

    for _ in range(num_sources):
        overlay.add_source(requirement=rand_requirement())

    for _ in range(num_blocks - num_sources - 1):
        overlay.add_intermediate(requirement=rand_requirement())

    # randomly add links
    for source in overlay.graph.nodes():
        for sink in overlay.graph.nodes():
            if sink != source and rand.random() < pairwise_connection_prob:
                overlay.add_link(source, sink)

    # add links necessary to have each block on a path from a source to
    # the sink
    accessible_from_source = set()
    not_accessible_from_source = set()
    has_path_to_sink = set()
    no_path_to_sink = set()
    for node in overlay.graph.nodes():
        # check if the node can already reach the sink
        if nx.has_path(overlay.graph, node, overlay.sink):
            has_path_to_sink.add(node)
        else:
            no_path_to_sink.add(node)

        # check if the node is already reachable from the source
        source_path_found = False
        for source in overlay.sources:
            if nx.has_path(overlay.graph, source, node):
                source_path_found = True
                break
        if source_path_found:
            accessible_from_source.add(node)
        else:
            not_accessible_from_source.add(node)

    # make sure all nodes are reachable from a source
    for node in not_accessible_from_source:
        connection = rand.choice(list(accessible_from_source))
        overlay.add_link(connection, node)
        accessible_from_source.add(node)

    # make sure all nodes can reach the sink
    for node in no_path_to_sink:
        connection = rand.choice(list(has_path_to_sink))
        overlay.add_link(node, connection)
        has_path_to_sink.add(node)

    return overlay


def draw_overlay(
    overlay: OverlayNetwork,
    sources_color="red",
    sink_color="yellow",
    intermediates_color="green",
):
    """Draws a given OverlayNetwork"""
    shared_args = {
        "G": overlay.graph,
        "pos": nx.spring_layout(overlay.graph),
        "node_size": 450,
        "node_shape": "s",
    }
    nx.draw_networkx_nodes(
        nodelist=list(overlay.sources), node_color=sources_color, **shared_args
    )
    nx.draw_networkx_nodes(
        nodelist=list(overlay.intermediates),
        node_color=intermediates_color,
        **shared_args,
    )
    nx.draw_networkx_nodes(
        nodelist=[overlay.sink], node_color=sink_color, **shared_args
    )
    nx.draw_networkx_labels(**shared_args)
    nx.draw_networkx_edges(**shared_args)

    # positions are arbitrary for an overlay
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)


if __name__ == "__main__":
    draw_overlay(random_overlay(np.random))
    plt.show()
