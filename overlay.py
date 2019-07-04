"""Modelling the overlay network"""

from enum import Enum
from matplotlib import pyplot as plt
import networkx as nx


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
        if block is None:
            return 0
        return self.graph.node[block]["requirement"]

    def _generate_name(self):
        self._last_id += 1
        return f"B{self._last_id}"

    def add_link(self, source: str, sink: str, datarate=2.0):
        """Adds a link between two blocks in the overlay network"""
        self.graph.add_edge(source, sink, datarate=datarate)

    def _block_to_verbose_str(self, block):
        requirement = self.requirement(block)
        return f'(name="{block}", requirement={requirement})'

    def __str__(self):
        result = "Overlay with:\n"
        result += f"- {len(self.sources)} sources:\n"
        for source in self.sources:
            s = self._block_to_verbose_str(source)
            result += f"  - add_source{s}\n"
        result += f"- {len(self.intermediates)} intermediates:\n"
        for intermediate in self.intermediates:
            i = self._block_to_verbose_str(intermediate)
            result += f"  - add_intermediate{i}\n"
        result += f"- one sink:\n"
        s = self._block_to_verbose_str(self.sink)
        result += f"  - set_sink{s}\n"

        links = self.graph.edges(data=True)
        result += f"- {len(links)} links:\n"
        for (u, v, d) in links:
            result += f"  - add_link({u}, {v}, {d['datarate']})\n"
        return result


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
    from generator import random_overlay

    draw_overlay(random_overlay(2))
    plt.show()
