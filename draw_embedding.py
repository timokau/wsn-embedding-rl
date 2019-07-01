"""Functions for drawing embeddings, mostly for debugging"""

import networkx as nx
from matplotlib import pyplot as plt

from embedding import PartialEmbedding, ENode
from overlay import OverlayNetwork
from infrastructure import InfrastructureNetwork


def draw_embedding(
    embedding: PartialEmbedding,
    sources_color="red",
    sink_color="yellow",
    intermediates_color="green",
):
    """Draws a given PartialEmbedding"""
    g = embedding.graph
    shared_args = {
        "G": g,
        "node_size": 1000,
        "pos": nx.shell_layout(embedding.graph),
    }

    node_list = g.nodes()
    chosen = [node for node in node_list if g.nodes[node]["chosen"]]
    not_chosen = [node for node in node_list if not g.nodes[node]["chosen"]]

    def kind_color(node):
        kind = g.nodes[node]["kind"]
        color = intermediates_color
        if kind == "source":
            color = sources_color
        elif kind == "sink":
            color = sink_color
        return color

    nx.draw_networkx_nodes(
        nodelist=not_chosen,
        node_color=list(map(kind_color, not_chosen)),
        node_shape="o",
        **shared_args,
    )
    nx.draw_networkx_nodes(
        nodelist=chosen,
        node_color=list(map(kind_color, chosen)),
        node_shape="s",
        **shared_args,
    )
    nx.draw_networkx_labels(**shared_args)

    possibilities = embedding.possibilities()

    def chosen_color(edge):
        data = g.edges[edge]
        chosen = data["chosen"]
        (source, target, _) = edge
        if (source, target, data["timeslot"]) in possibilities:
            return "blue"
        if chosen:
            return "black"
        return "gray"

    def chosen_width(edge):
        data = g.edges[edge]
        (source, target, _) = edge
        chosen = data["chosen"]
        possible = (source, target, data["timeslot"]) in possibilities

        if chosen:
            return 2
        if possible:
            return 1
        return 0.1

    edgelist = g.edges(keys=True)
    nx.draw_networkx_edges(
        **shared_args,
        edgelist=edgelist,
        edge_color=list(map(chosen_color, edgelist)),
        width=list(map(chosen_width, edgelist)),
    )

    chosen_edges = [edge for edge in edgelist if g.edges[edge]["chosen"]]
    # Networkx doesn't really deal with drawing multigraphs very well.
    # Luckily for our presentation purposes its enough to pretend the
    # graph isn't a multigraph, so throw away the edge keys.
    labels = {
        (u, v): g.edges[(u, v, k)]["timeslot"] for (u, v, k) in chosen_edges
    }
    nx.draw_networkx_edge_labels(
        **shared_args, edgelist=chosen_edges, edge_labels=labels
    )

    timeslots = embedding.used_timeslots
    complete = embedding.is_complete()
    complete_str = " (complete)" if complete else ""
    plt.gca().text(
        -1,
        -1,
        f"{timeslots} timeslots{complete_str}",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )


def _build_example():
    # for quick testing
    infra = InfrastructureNetwork()
    n1 = infra.add_source(
        pos=(0, 3), transmit_power_dbm=14, capacity=5, name="N1"
    )
    n2 = infra.add_source(
        pos=(0, 1), transmit_power_dbm=8, capacity=8, name="N2"
    )
    n3 = infra.add_intermediate(
        pos=(2, 2), transmit_power_dbm=32, capacity=20, name="N3"
    )
    n4 = infra.set_sink(
        pos=(3, 0), transmit_power_dbm=10, capacity=10, name="N4"
    )
    n5 = infra.add_intermediate(
        pos=(1, 2), transmit_power_dbm=20, capacity=42, name="N5"
    )

    overlay = OverlayNetwork()
    b1 = overlay.add_source(requirement=5, name="B1")
    b2 = overlay.add_source(requirement=5, name="B2")
    b3 = overlay.add_intermediate(requirement=5, name="B3")
    b4 = overlay.set_sink(requirement=5, name="B4")

    overlay.add_link(b1, b3)
    overlay.add_link(b2, b3)
    overlay.add_link(b3, b4)
    overlay.add_link(b2, b4)

    embedding = PartialEmbedding(
        infra, overlay, source_mapping=[(b1, n1), (b2, n2)]
    )

    assert embedding.take_action(ENode(b1, n1), ENode(None, n5), 0)
    assert embedding.take_action(
        ENode(None, n5, ENode(b1, n1)), ENode(b3, n3), 1
    )
    assert embedding.take_action(ENode(b2, n2), ENode(None, n5), 2)
    assert embedding.take_action(
        ENode(None, n5, ENode(b2, n2)), ENode(b3, n3), 3
    )
    assert embedding.take_action(ENode(b2, n2), ENode(b4, n4), 2)
    assert embedding.take_action(ENode(b3, n3), ENode(b4, n4), 4)
    return embedding


if __name__ == "__main__":
    # pylint: disable=ungrouped-imports
    from networkx.drawing.nx_pydot import write_dot

    write_dot(_build_example().succinct_representation(), "result.dot")
