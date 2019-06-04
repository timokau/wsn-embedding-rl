"""Model of wireless overlay networks"""

from typing import List, Tuple, Iterable

import networkx as nx
from networkx.algorithms.shortest_paths.generic import has_path
from matplotlib import pyplot as plt

import wsignal
from infrastructure import InfrastructureNetwork
from overlay import OverlayNetwork


class ENode:
    """A node representing a possible or actual embedding"""

    def __init__(
        self,
        block,
        node,
        # type has to be a string in recursive type definitions
        predecessor: "ENode" = None,
    ):
        self.block = block
        self.node = node
        self.relay = self.block is None
        self.predecessor = predecessor
        self._hash = None
        if self.block is not None:
            self.acting_as = block
        elif self.predecessor is not None:
            self.acting_as = predecessor.acting_as
        else:
            self.acting_as = None

    def __repr__(self):
        result = ""
        if not self.relay:
            result += str(self.block) + "-"
        elif self.acting_as is not None:
            result += f"({self.acting_as})-"
        result += str(self.node)
        return result

    def __eq__(self, other):
        if not isinstance(other, ENode):
            return False

        return (
            self.block == other.block
            and self.node == other.node
            and (not self.relay or self.predecessor == other.predecessor)
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        if self._hash is None:
            if self.relay:
                self._hash = hash((self.block, self.node, self.predecessor))
            else:
                self._hash = hash((self.block, self.node))
        return self._hash


class PartialEmbedding:
    # pylint: disable=too-many-instance-attributes
    # Instance attributes needed for caching, I think private instance
    # attributes are fine.
    """A graph representing a partial embedding and possible actions"""

    def __init__(
        self,
        infra: InfrastructureNetwork,
        overlay: OverlayNetwork,
        # map block to node
        source_mapping: List[Tuple[str, str]],
        sinrth: float = 2.0,
    ):
        self.infra = infra
        self.overlay = overlay
        self.used_timeslots = -1
        self.sinrth = sinrth

        self.graph = nx.MultiDiGraph()

        # just for ease of access
        self._relays = set()
        self._by_block = dict()
        self._transmissions_at = dict()
        self._known_sinr_cache = dict()
        self.embedded_links = []

        self._build_possibilities_graph(source_mapping)

    def possibilities(self):
        """Returns a list of possible actions (edges)"""
        is_chosen = lambda node: self.graph.nodes[node]["chosen"]
        chosen_nodes = [node for node in self.nodes() if is_chosen(node)]
        out_edges = self.graph.out_edges(nbunch=chosen_nodes, data=True)
        possibilities = [
            (source, target, data["timeslot"])
            for (source, target, data) in out_edges
            if not data["chosen"]
        ]
        return possibilities

    def nodes(self):
        """Shortcut to get all ENodes of the underlying graph"""
        return self.graph.nodes()

    def _add_possible_intermediate_embeddings(self):
        for block in self.overlay.intermediates:
            for node in self.infra.graph.nodes():
                self.add_node(ENode(block, node))

    def _embed_sources(self, source_mapping: List[Tuple[str, str]]):
        for (block, node) in source_mapping:
            embedding = ENode(block, node)
            self.add_node(embedding, chosen=True)

    def _embed_sink(self):
        embedding = ENode(self.overlay.sink, self.infra.sink)
        self.add_node(embedding, chosen=True)

    def _add_link_edges(self, timeslot: int):
        for (source_block, sink_block) in self.overlay.links():
            for source_embedding in self._by_block.get(source_block, set()):
                for sink_embedding in self._by_block.get(sink_block, set()):
                    self.add_edge(source_embedding, sink_embedding, timeslot)

    def _add_relay_nodes(self):
        for node in self.infra.nodes():
            embedding = ENode(None, node)
            self._relays.add(embedding)
            self.add_node(embedding, relay=True)

    def add_node(self, node: ENode, chosen=False, relay=False):
        """Adds a given ENode to the graph"""
        bb = self._by_block.get(node.block, set())
        bb.add(node)
        self._by_block[node.block] = bb

        kind = "intermediate"
        if node.block in self.overlay.sources:
            kind = "source"
        elif node.block == self.overlay.sink:
            kind = "sink"

        self.graph.add_node(node, chosen=chosen, relay=relay, kind=kind)

    def add_edge(self, source: ENode, sink: ENode, timeslot: int):
        """Adds a possible connection to the graph if it is feasible"""
        self.graph.add_edge(
            source,
            sink,
            # a new edge can never be chosen
            chosen=False,
            timeslot=timeslot,
            # edges are uniquely identified by (source, target, timeslot)
            key=timeslot,
        )
        if not self._link_feasible(source, sink, timeslot):
            self.remove_link(source, sink, timeslot)

    def remove_edges_from(self, source: ENode):
        """Removes all edges originating from source"""
        to_remove = list(self.graph.out_edges(nbunch=[source], keys=True))
        self.graph.remove_edges_from(to_remove)

    def remove_edges_between(self, source: ENode, sink: ENode):
        """Removes all edges between two nodes in the multigraph"""
        while True:
            # remove edges for all timeslots
            try:
                # remove one arbitrary edge between source and sink
                self.graph.remove_edge(source, sink)
            except nx.NetworkXError:
                break

    def _add_relay_edges(self, timeslot):
        # relay nodes are fully connected to allow for arbitrary paths
        for source_node in self.infra.nodes():
            for sink_node in self.infra.nodes():
                if source_node != sink_node:
                    source_embedding = ENode(None, source_node)
                    sink_embedding = ENode(None, sink_node)
                    self.add_edge(source_embedding, sink_embedding, timeslot)

        # incoming + outgoing relay
        for block in self.overlay.graph.nodes():
            # if a block expects an incoming connection, it may come
            # from any relay node
            add_incoming = self.overlay.graph.in_degree[block] > 0

            # if a block had an outgoing connection, it may go to any
            # relay node
            add_outgoing = self.overlay.graph.out_degree[block] > 0

            for relay in self._relays:
                for embedding in self._by_block.get(block, set()):
                    if embedding.node == relay.node:
                        continue
                    if add_incoming:
                        self.add_edge(relay, embedding, timeslot)
                    if add_outgoing:
                        self.add_edge(embedding, relay, timeslot)

    def _build_possibilities_graph(
        self, source_mapping: List[Tuple[str, str]]
    ):
        self._add_possible_intermediate_embeddings()
        self._embed_sink()
        self._embed_sources(source_mapping)
        self._add_relay_nodes()

        self.add_timeslot()

    def remove_link(self, source: ENode, sink: ENode, timeslot: int):
        """Removes a link given its source, sink and timeslot"""
        self.graph.remove_edge(source, sink, timeslot)

    def _invalidates_chosen(self, source, timeslot):
        """Checks if node sending would invalidate sinr of chosen action"""
        for (u, v) in self._all_known_transmissions_at(timeslot):
            new_sinr = self.known_sinr(
                u.node,
                v.node,
                timeslot=timeslot,
                additional_senders={source.node},
            )
            if new_sinr < self.sinrth:
                return True
        return False

    def _sinr_valid(self, source, target, timeslot):
        """Checks if link sinr is valid"""
        sinr = self.known_sinr(source.node, target.node, timeslot)
        return sinr >= self.sinrth

    def _unembedded_outlinks_left(self, enode):
        """Checks if there are any unembedded outgoing links left"""
        embedded = self.graph.nodes[enode].get("chosen_out", 0)

        num_out_links_to_embed = len(
            self.overlay.graph.out_edges(nbunch=[enode.acting_as])
        )

        return num_out_links_to_embed > embedded

    def _completes_already_embedded_link(self, source, target):
        """Checks if a new link would doubly embed a link"""
        if target.relay or source.acting_as is None:
            return False

        if (source.acting_as, target.acting_as) in self.embedded_links:
            return True

        return False

    def _valid_by_itself(self, source, target, timeslot):
        if not self._sinr_valid(source, target, timeslot):
            return False
        if self._completes_already_embedded_link(source, target):
            return False
        if not self._unembedded_outlinks_left(source):
            return False
        return True

    def _link_feasible(self, source, target, timeslot):
        if not self._valid_by_itself(source, target, timeslot):
            return False

        if self._invalidates_chosen(source, timeslot):
            return False
        return True

    def _remove_infeasible_links(self, timeslot):
        for (source, target, link_ts) in self.possibilities():
            if link_ts != timeslot:
                continue

            if not self._link_feasible(source, target, link_ts):
                self.remove_link(source, target, link_ts)

    def readjust(self, timeslot):
        """Removes now infeasible actions"""
        self._remove_infeasible_links(timeslot)

    def _all_known_transmissions_at(self, timeslot):
        return self._transmissions_at.get(timeslot, [])

    def power_at_node(
        self, node: str, timeslot: int, additional_senders: Iterable[str] = ()
    ):
        """
        Calculates the amount of power a node receives (signal+noise) at
        the given timeslot.

        It is assumed that only the already chosen
        edges and the currently considered edge are sending.
        """
        # We need to convert to watts for addition (log scale can only
        # multiply)
        received_power_watt = 0
        transmissions = self._all_known_transmissions_at(timeslot)

        # use a set to make sure broadcasts aren't counted twice
        sending_nodes = {u.node for (u, v) in transmissions}
        sending_nodes = sending_nodes.union(additional_senders)
        for sender in sending_nodes:
            p_r = self.infra.power_received_dbm(sender, node)
            received_power_watt += wsignal.dbm_to_watt(p_r)
        return wsignal.watt_to_dbm(received_power_watt)

    def known_sinr(
        self,
        source_node: str,
        target_node: str,
        timeslot: int,
        additional_senders: Iterable[str] = (),
        # https://www.quora.com/How-high-is-the-ambient-RF-noise-floor-in-the-2-4-GHz-spectrum-in-downtown-San-Francisco
        noise_floor_dbm: float = -80,
    ):
        """
        SINR assuming only already chosen edges and the currently
        considered edges are sending.
        """
        index = (
            source_node,
            target_node,
            tuple(additional_senders),
            noise_floor_dbm,
        )
        timeslot_cache = self._known_sinr_cache.get(timeslot, dict())
        cached = timeslot_cache.get(index)
        if cached is None:
            received_signal_dbm = self.infra.power_received_dbm(
                source_node, target_node
            )

            # make sure source node is already counted (which it will be
            # in the case of broadcast anyway), subtract it later
            additional_senders = set(additional_senders)
            additional_senders.add(source_node)

            received_power_dbm = self.power_at_node(
                target_node, timeslot, additional_senders=additional_senders
            )
            received_interference_dbm = wsignal.subtract_dbm(
                received_power_dbm, received_signal_dbm
            )

            cached = wsignal.sinr(
                received_signal_dbm, received_interference_dbm, noise_floor_dbm
            )
            timeslot_cache[index] = cached
            self._known_sinr_cache[timeslot] = timeslot_cache
        return cached

    def wire_up_outgoing(self, enode: ENode, timeslot: int):
        """Connect a new ENode to all its possible successors"""
        for relay in self._relays:
            if relay.node != enode.node:
                self.add_edge(enode, relay, timeslot)

        out_edges = self.overlay.graph.out_edges(nbunch=[enode.acting_as])

        for (_, v) in out_edges:
            for option in self._by_block.get(v, set()):
                self.add_edge(enode, option, timeslot)

    def add_timeslot(self):
        """Adds a new timeslot as an option"""
        self.used_timeslots += 1
        self._add_link_edges(self.used_timeslots)
        self._add_relay_edges(self.used_timeslots)

    def take_action(self, source: ENode, sink: ENode, timeslot: int):
        """Take an action represented by an edge and update the graph"""
        if (source, sink, timeslot) not in self.possibilities():
            return False

        self.remove_edges_between(source, sink)
        if source.relay:
            self.remove_edges_from(source)

        if not sink.relay:
            originating = source
            while originating.relay:
                originating = originating.predecessor
            # link completed, clean up originating block
            out_edges = self.graph.out_edges(nbunch=[originating], keys=True)
            for (u, v, k) in list(out_edges):
                if v.block == sink.block:
                    self.graph.remove_edge(u, v, k)

        target_block = sink.block
        target_node = sink.node
        new_enode = ENode(
            block=target_block, node=target_node, predecessor=source
        )

        self.add_node(new_enode, chosen=True)
        self.graph.add_edge(
            source, new_enode, chosen=True, timeslot=timeslot, key=timeslot
        )
        if not new_enode.relay:
            self.embedded_links += [(source.acting_as, new_enode.acting_as)]
        before = self.graph.nodes[source].get("chosen_out", 0)
        self.graph.nodes[source]["chosen_out"] = before + 1
        self._known_sinr_cache[timeslot] = dict()
        self._transmissions_at[timeslot] = self._transmissions_at.get(
            timeslot, []
        ) + [(source, new_enode)]

        # determine if we're actually just updating an existing node
        # (normal case) or really creating a new node (a link-specific
        # copy of a relay)
        actually_new = new_enode.relay
        if actually_new:
            self._by_block[new_enode.acting_as].add(new_enode)
            for ts in range(self.used_timeslots + 1):
                self.wire_up_outgoing(new_enode, ts)

        if timeslot >= self.used_timeslots:
            self.add_timeslot()

        self.readjust(timeslot)
        return True

    def chosen_subgraph(self):
        """
        Returns a subgraph containing only the already chosen components
        """
        edges = self.graph.edges(keys=True, data=True)
        chosen_edges = {(u, v, k) for (u, v, k, d) in edges if d["chosen"]}
        return self.graph.edge_subgraph(chosen_edges)

    def is_solvable(self):
        """Determines weather there is at least one valid path from each
        source to the sink (assuming no interfereing communications for each
        transmission, i.e. infinite timesteps)."""
        esink = list(self._by_block[self.overlay.sink])[0]
        for osource in self.overlay.sources:
            esource = list(self._by_block[osource])[0]
            if not has_path(self.graph, esource, esink):
                return False

        return True

    def is_complete(self):
        """Determines if all blocks and links are embedded"""
        # check that each link is embedded
        for link in self.overlay.links():
            if link not in self.embedded_links:
                return False
        return True


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
