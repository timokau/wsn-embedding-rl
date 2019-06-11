"""Model of wireless overlay networks"""

from typing import List, Tuple, Iterable
from collections import defaultdict
import math
from math import inf

import networkx as nx
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
    ):
        self.infra = infra
        self.overlay = overlay
        self.used_timeslots = -1

        self.graph = nx.MultiDiGraph()

        # just for ease of access
        self._relays = set()
        self._by_block = dict()
        self._by_node = defaultdict(set)
        self._taken_edges = dict()
        self._taken_embeddings = dict()
        self._num_outlinks_embedded = defaultdict(int)
        self._capacity_used = defaultdict(float)
        self._transmissions_at = dict()
        self._known_sinr_cache = dict()
        self.embedded_links = []

        self._build_possibilities_graph(source_mapping)

    def options(self):
        """Returns a list of not yet options, which may or may not be
        possible yet"""
        out_edges = self.graph.out_edges(data=True)
        options = [
            (source, target, data["timeslot"])
            for (source, target, data) in out_edges
            if not data["chosen"]
        ]
        return options

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
            requirement = self.overlay.requirement(block)
            assert requirement <= self.infra.capacity(node)
            self._capacity_used[(node, 0)] += requirement
            embedding = ENode(block, node)
            self.add_node(embedding, chosen=True)

    def _embed_sink(self):
        osink = self.overlay.sink
        isink = self.infra.sink
        # capacity used will be updated with the first incoming
        # connection
        assert self.overlay.requirement(osink) <= self.infra.capacity(isink)
        embedding = ENode(osink, isink)
        self.add_node(embedding, chosen=True)

    def _add_link_edges(self, timeslot: int):
        for (source_block, sink_block) in self.overlay.links():
            for source_embedding in self._by_block.get(source_block, set()):
                for sink_embedding in self._by_block.get(sink_block, set()):
                    if not sink_embedding.relay:
                        self.add_edge(
                            source_embedding, sink_embedding, timeslot
                        )

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
        self._by_node[node.node].add(node)

        kind = "intermediate"
        if node.block in self.overlay.sources:
            kind = "source"
        elif node.block == self.overlay.sink:
            kind = "sink"

        self.graph.add_node(node, chosen=chosen, relay=relay, kind=kind)
        if chosen and not relay:
            self._taken_embeddings[node.block] = node

    def _compute_min_datarate(self, source, target, timeslot):
        min_datarate = inf
        may_represent = self.graph.edges[(source, target, timeslot)][
            "may_represent"
        ]
        for (u, v) in may_represent:
            datarate = self.overlay.graph.edges[(u, v)]["datarate"]
            if datarate < min_datarate:
                min_datarate = datarate
        self.graph.edges[(source, target, timeslot)][
            "min_datarate"
        ] = min_datarate

    def add_edge(
        self, source: ENode, sink: ENode, timeslot: int, chosen=False
    ):
        """Adds a possible connection to the graph if it is feasible"""
        # make sure we don't accidentally un-choose an edge
        assert chosen or not self.graph.has_edge(source, sink, timeslot)
        may_represent = set()
        for (u, v) in self.overlay.graph.out_edges(nbunch=[source.acting_as]):
            if (u, v) not in self.embedded_links and (
                sink.block is None or sink.acting_as == v
            ):
                may_represent.add((u, v))

        self.graph.add_edge(
            source,
            sink,
            chosen=chosen,
            timeslot=timeslot,
            # edges are uniquely identified by (source, target, timeslot)
            key=timeslot,
            may_represent=may_represent,
        )
        self._compute_min_datarate(source, sink, timeslot)
        if not chosen and not self._link_feasible(source, sink, timeslot):
            self.remove_link(source, sink, timeslot)

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
        assert not self.graph.edges[(source, sink, timeslot)]["chosen"]
        self.graph.remove_edge(source, sink, timeslot)

    def _invalidates_chosen(self, source, timeslot):
        """Checks if node sending would invalidate datarate of chosen action"""
        for (u, v) in self._all_known_transmissions_at(timeslot):
            new_capacity = self.known_capacity(
                u.node,
                v.node,
                timeslot=timeslot,
                additional_senders={source.node},
            )
            thresh = self.graph.edges[(u, v, timeslot)]["min_datarate"]
            if new_capacity < thresh:
                return True
        return False

    def _datarate_valid(self, source, target, timeslot):
        """Checks if link datarate is valid"""
        thresh = self.graph.edges[(source, target, timeslot)]["min_datarate"]
        capacity = self.known_capacity(source.node, target.node, timeslot)
        return capacity >= thresh

    def _unembedded_outlinks_left(self, enode):
        """Checks if there are any unembedded outgoing links left"""
        block = enode.acting_as
        if block is None:
            return True

        embedded = self._num_outlinks_embedded[block]
        if enode.relay:
            # a relay was already counted, but is part of a not yet
            # completed link. It should not count at the tip of the
            # unfinished link
            if not self.graph.nodes[enode].get("has_out", False):
                embedded -= 1

        num_out_links_to_embed = self.overlay.graph.out_degree(block)

        return num_out_links_to_embed > embedded

    def _completes_already_embedded_link(self, source, target):
        """Checks if a new link would doubly embed a link"""
        if target.relay or source.acting_as is None:
            return False

        if (source.acting_as, target.acting_as) in self.embedded_links:
            return True

        return False

    def _link_already_taken(self, source, target):
        return (source, target) in self._taken_edges.keys()

    def _link_necessary(self, source, target):
        if self._completes_already_embedded_link(source, target):
            return False
        if not self._unembedded_outlinks_left(source):
            return False
        if self._link_already_taken(source, target):
            return False
        return True

    def _link_feasible(self, source, target, timeslot):
        return self._link_feasible_in_timeslot(
            source, target, timeslot
        ) and self._link_necessary(source, target)

    def _node_can_carry(self, node, block, timeslot):
        if block is None:
            return True
        used = self._capacity_used[(node, timeslot)]
        needed = self.overlay.requirement(block)
        available = self.infra.capacity(node)
        return used + needed <= available

    def _node_sending_other_data_in_timeslot(self, enode, timeslot):
        # the same embedding can send multiple times within a timeslot
        # (broadcasting results), but others cannot (which would send
        # other data)
        other_embeddings = self._by_node[enode.node] - set([enode])
        for u, v, data in self.graph.out_edges(
            nbunch=other_embeddings, data=True
        ):
            # loops are fine, they do not actually involve any sending
            is_loop = u.node == v.node
            if data["timeslot"] == timeslot and data["chosen"] and not is_loop:
                return True
        return False

    def _link_feasible_in_timeslot(self, source, target, timeslot):
        if self._node_sending_other_data_in_timeslot(source, timeslot):
            return False

        if not self._node_can_carry(target.node, target.block, timeslot):
            return False

        if not self._datarate_valid(source, target, timeslot):
            return False

        if self._invalidates_chosen(source, timeslot):
            return False
        return True

    def _remove_already_completed_inlinks(self, enode):
        for (u, v, k, d) in list(
            self.graph.in_edges(nbunch=[enode], keys=True, data=True)
        ):
            if not d["chosen"] and self._completes_already_embedded_link(u, v):
                self.remove_link(u, v, k)

    def _remove_other_outlinks_of(self, enode):
        """Removes not-chosen outlinks for an enode"""
        for (u, v, k, d) in list(
            self.graph.out_edges(nbunch=[enode], keys=True, data=True)
        ):
            if not d["chosen"]:
                self.remove_link(u, v, k)

    def _remove_other_options_for(self, block):
        """Removes not-chosen options for a block"""
        new_by_block = set()
        for node in self._by_block[block]:
            if not self.graph.node[node]["chosen"]:
                self.graph.remove_node(node)
            else:
                new_by_block.add(node)
        self._by_block[block] = new_by_block

    def _remove_links_between(self, source, target):
        """Removes all remaining unchosen links between two ENodes"""
        for timeslot in range(self.used_timeslots + 1):
            if self.graph.has_edge(source, target, timeslot):
                chosen = self.graph.edges[source, target, timeslot]["chosen"]
                if not chosen:
                    self.remove_link(source, target, timeslot)

    def _remove_links_infeasible_in(self, timeslot):
        """Removes links that are no longer feasible within a
        timeslot"""
        for (source, target, link_ts) in self.options():
            if link_ts != timeslot:
                continue

            if not self._link_feasible_in_timeslot(source, target, link_ts):
                self.remove_link(source, target, link_ts)

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

    def known_capacity(
        self,
        source_node: str,
        target_node: str,
        timeslot: int,
        additional_senders: Iterable[str] = (),
        noise_floor_dbm: float = -80,
    ):
        """
        Link capacity assuming only already chosen edges and the
        currently considered edges are sending.
        """
        sinr = self.known_sinr(
            source_node,
            target_node,
            timeslot,
            additional_senders,
            noise_floor_dbm,
        )
        bandwidth = self.infra.bandwidth
        shannon_capacity = bandwidth * math.log(1 + 10 ** (sinr / 10), 2)
        return shannon_capacity

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
                if not option.relay or option.predecessor is None:
                    # do not connect to used relays
                    self.add_edge(enode, option, timeslot)

    def add_timeslot(self):
        """Adds a new timeslot as an option"""
        self.used_timeslots += 1
        self._add_link_edges(self.used_timeslots)
        self._add_relay_edges(self.used_timeslots)

    def take_action(self, source: ENode, sink: ENode, timeslot: int):
        """Take an action represented by an edge and update the graph"""
        # pylint: disable=too-many-branches,too-many-statements
        if (source, sink, timeslot) not in self.possibilities():
            return False

        # this should never be false, that would be a bug
        assert self._link_feasible_in_timeslot(source, sink, timeslot)

        if not sink.relay:
            originating = source
            while originating.relay:
                originating = originating.predecessor
            # link completed, clean up originating block
            out_edges = self.graph.out_edges(nbunch=[originating], keys=True)
            for (u, v, k) in list(out_edges):
                if v.block == sink.block:
                    self.remove_link(u, v, k)

        target_block = sink.block
        target_node = sink.node
        new_enode = ENode(
            block=target_block, node=target_node, predecessor=source
        )

        if target_block is not None:
            requires = self.overlay.requirement(target_block)
            self._capacity_used[(target_node, timeslot)] += requires

        newly_embedded = not sink.relay and not self.graph.node[sink]["chosen"]
        self.add_node(new_enode, chosen=True)
        if newly_embedded:
            self._remove_other_options_for(new_enode.block)
        self.add_edge(source, new_enode, chosen=True, timeslot=timeslot)
        self._taken_edges[(source, sink)] = timeslot
        if not new_enode.relay:
            link = (source.acting_as, new_enode.acting_as)
            self.embedded_links += [link]
            for (u, v, d) in list(
                self.graph.out_edges(
                    nbunch=self._by_block[source.acting_as], data=True
                )
            ):
                if d["chosen"]:
                    continue
                try:
                    d["may_represent"].remove(link)
                except KeyError:
                    pass
                if len(d["may_represent"]) == 0:
                    self.remove_link(u, v, d["timeslot"])
                else:
                    self._compute_min_datarate(u, v, d["timeslot"])

        if not source.relay:
            # we count this as an embedded outlink, even if the link is
            # not completed yet. Once we have chosen the beginning of a
            # link, it does not make sense to begin the link in another
            # way too.
            self._num_outlinks_embedded[source.acting_as] += 1
        else:
            # if the link is originating as a relay, the link was
            # already counted once. It is only counted again if the path
            # forks, i.e. this is the second outlink of the relay.
            if self.graph.nodes[source].get("has_out", False):
                self._num_outlinks_embedded[source.acting_as] += 1
            self.graph.nodes[source]["has_out"] = True

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

        self._remove_links_between(source, sink)
        for enode in self._by_block[source.acting_as]:
            if not self._unembedded_outlinks_left(enode):
                self._remove_other_outlinks_of(enode)
        if not sink.relay:
            # check for other options that would have completed the same
            # link
            for enode in self._by_block[sink.acting_as]:
                self._remove_already_completed_inlinks(enode)
        self._remove_links_infeasible_in(timeslot)
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
        for osource in self.overlay.sources:
            esource = list(self._by_block[osource])[0]
            reachable_enodes = nx.descendants(self.graph, esource)
            reachable_blocks = {enode.block for enode in reachable_enodes}
            blocks_needing_info_from_source = nx.descendants(
                self.overlay.graph, esource.block
            )
            if not blocks_needing_info_from_source.issubset(reachable_blocks):
                return False

        return True

    def is_complete(self):
        """Determines if all blocks and links are embedded"""
        # check that each link is embedded
        for link in self.overlay.links():
            if link not in self.embedded_links:
                return False
        return True

    def __str__(self):
        result = "Embedding with:\n"
        result += self.infra.__str__()
        result += self.overlay.__str__()
        return result

    def construct_link_mappings(self):
        """Returns a mapping from links to paths"""
        result = dict()
        for (b1, b2) in self.embedded_links:
            source_embedding = self._taken_embeddings[b1]
            target_embedding = self._taken_embeddings[b2]
            cur = target_embedding
            path = []
            while cur != source_embedding:
                in_edges = self.graph.in_edges(keys=True, nbunch=[cur])
                for (u, v, k) in in_edges:
                    if u.acting_as == b1:
                        path.append((v, k))
                        cur = u
                        break
            path.reverse()
            result[(b1, b2)] = path
        return result

    def succinct_representation(self, distance_scale=3):
        """Returns a succinct representation of the embedding

        Only takes into account the choices that were taken, not all
        possibilities. As a result, it can represent much bigger graphs
        than the draw_embedding representation.
        """

        repr_graph = nx.MultiDiGraph()
        scale_factor = distance_scale / self.infra.min_node_distance()
        blocks_in_node = defaultdict(set)

        for enode in self.graph.nodes():
            if self.graph.node[enode]["chosen"] and enode.block is not None:
                blocks_in_node[enode.node].add(enode.block)

        for infra_node in self.infra.nodes():
            x, y = self.infra.position(infra_node)
            x *= scale_factor
            y *= scale_factor
            capacity = round(self.infra.capacity(infra_node), 1)
            power = round(self.infra.power(infra_node), 1)
            block_strings = []
            for block in blocks_in_node[infra_node]:
                # block = f'<FONT COLOR="#0000AA">{block}</FONT>'
                block_strings += [block]
            embedded_str = f"< {', '.join(block_strings)} >"
            style = "rounded"
            if infra_node in self.infra.sources:
                style = "bold"
            elif infra_node == self.infra.sink:
                style = "filled"
            repr_graph.add_node(
                infra_node,
                shape="polygon",
                style=style,
                label=f"{infra_node}\n{capacity}cap\n{power}dBm",
                xlabel=embedded_str,
                pos=f"{x},{y}!",
            )

        for (link, path) in self.construct_link_mappings().items():
            source = self._taken_embeddings[link[0]]
            target = self._taken_embeddings[link[1]]
            # first show the target link
            repr_graph.add_edge(
                source.node, target.node, style="dashed", color="blue"
            )

            # add the actual embedding
            for (target, timeslot) in path:
                sinr = self.known_sinr(source.node, target.node, timeslot)
                repr_graph.add_edge(
                    source.node,
                    target.node,
                    label=f"{link[0]}->{link[1]}\n{timeslot}",
                    penwidth=sinr / 20,
                )
                source = target

        return repr_graph


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
