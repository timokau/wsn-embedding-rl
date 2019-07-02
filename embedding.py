"""Model of wireless overlay networks"""

from typing import List, Tuple, Iterable
from collections import defaultdict
import math
from math import inf

import networkx as nx

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
        # cache the hashes, since hashing is quite expensive
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
        self._source_mapping = source_mapping
        self.used_timeslots = -1

        self.graph = nx.MultiDiGraph()

        # just for ease of access
        self._by_block = defaultdict(set)
        self._by_node = defaultdict(set)
        self._taken_edges = dict()
        self._taken_embeddings = dict()
        self._num_outlinks_embedded = defaultdict(int)
        self._capacity_used = defaultdict(float)
        self._transmissions_at = defaultdict(list)
        self.embedded_links = []

        self._build_possibilities_graph(source_mapping)

    def reset(self):
        """Returns a fresh, identically configured partial embedding"""
        return PartialEmbedding(self.infra, self.overlay, self._source_mapping)

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
            self.add_node(embedding)
            self.choose_node(embedding)

    def _embed_sink(self):
        osink = self.overlay.sink
        isink = self.infra.sink
        # capacity used will be updated with the first incoming
        # connection
        assert self.overlay.requirement(osink) <= self.infra.capacity(isink)
        embedding = ENode(osink, isink)
        self.add_node(embedding)
        self.choose_node(embedding)

    def _add_relay_nodes(self):
        for node in self.infra.nodes():
            embedding = ENode(None, node)
            self.add_node(embedding, relay=True)

    def add_node(self, node: ENode, relay=False):
        """Adds a given ENode to the graph"""
        self._by_block[node.block].add(node)
        self._by_node[node.node].add(node)

        kind = "intermediate"
        if node.block in self.overlay.sources:
            kind = "source"
        elif node.block == self.overlay.sink:
            kind = "sink"

        self.graph.add_node(node, chosen=False, relay=relay, kind=kind)

        # add the necessary edges
        self._by_block[node.acting_as].add(node)
        for ts in range(self.used_timeslots + 1):
            self._add_outedges(node, ts)

    def choose_node(self, node: ENode):
        """Marks an potential embedding as chosen and updates the rest
        of the graph with the consequences. Should only ever be done
        when the enode is the source, the sink or has an incoming chosen
        edge."""
        if self.graph.node[node]["chosen"]:
            return

        self.graph.node[node]["chosen"] = True

        if not node.relay:
            self._taken_embeddings[node.block] = node

            # remove other options for embedding this block
            for option in list(self._by_block[node.block]):
                if option != node:
                    self.remove_node(option)

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

    def add_edge(self, source: ENode, sink: ENode, timeslot: int):
        """Adds a possible connection to the graph if it is feasible"""
        may_represent = set()
        for (u, v) in self.overlay.graph.out_edges(nbunch=[source.acting_as]):
            if (u, v) not in self.embedded_links and (
                sink.block is None or sink.acting_as == v
            ):
                may_represent.add((u, v))

        self.graph.add_edge(
            source,
            sink,
            chosen=False,
            timeslot=timeslot,
            # edges are uniquely identified by (source, target, timeslot)
            key=timeslot,
            may_represent=may_represent,
        )
        self._compute_min_datarate(source, sink, timeslot)
        if not self._link_feasible(source, sink, timeslot):
            self.remove_link(source, sink, timeslot)

    def choose_edge(self, source: ENode, target: ENode, timeslot: int):
        """Marks a potential connection as chosen and updates the rest
        of the graph with the consequences. Should only ever be done
        when the source node is already chosen."""
        # pylint: disable=too-many-branches,too-many-statements
        assert self.graph.node[source]["chosen"]
        self.graph.edges[(source, target, timeslot)]["chosen"] = True

        self._capacity_used[
            (target.node, timeslot)
        ] += self.overlay.requirement(target.block)

        # if this completes a link
        if not target.relay:
            link = (source.acting_as, target.acting_as)
            self.embedded_links += [link]

            # update other edges that may have represented this link
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
                ts = d["timeslot"]
                self._compute_min_datarate(u, v, ts)
                if not self._link_feasible_in_timeslot(u, v, ts):
                    self.remove_link(u, v, ts)

        if source.relay:
            # if the link is originating as a relay, the link was
            # already counted once. It is only counted again if the path
            # forks, i.e. this is the second outlink of the relay.
            if self.graph.nodes[source].get("has_out", False):
                self._num_outlinks_embedded[source.acting_as] += 1
            self.graph.nodes[source]["has_out"] = True
        else:
            # we count this as an embedded outlink, even if the link is
            # not completed yet. Once we have chosen the beginning of a
            # link, it does not make sense to begin the link in another
            # way too.
            self._num_outlinks_embedded[source.acting_as] += 1

        self._transmissions_at[timeslot].append((source, target))

        for enode in self._by_block[source.acting_as]:
            if not self._unembedded_outlinks_left(enode):
                self._remove_other_outlinks_of(enode)
        if not target.relay:
            # check for other options that would have completed the same
            # link
            for enode in self._by_block[target.acting_as]:
                self._remove_already_completed_inlinks(enode)
        self._remove_links_infeasible_in(timeslot)

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

    def remove_node(self, node: ENode):
        """Removes a node if it is not chosen"""
        if self.graph.nodes[node]["chosen"]:
            return
        self._by_block[node.acting_as].remove(node)
        self._by_node[node.node].remove(node)
        self.graph.remove_node(node)

    def _invalidates_chosen(self, source, timeslot):
        """Checks if node sending would invalidate datarate of chosen action"""
        for (u, v) in self._transmissions_at[timeslot]:
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
        """Weather or not a node can support the computation for a block
        in a given timeslot"""
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

        if not self._datarate_valid(source, target, timeslot):
            return False

        if self._invalidates_chosen(source, timeslot):
            return False

        if not self._node_can_carry(target.node, target.block, timeslot):
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
        not_chosen_in_timeslot = [
            (source, target)
            for (source, target, data) in self.graph.out_edges(data=True)
            if not data["chosen"] and data["timeslot"] == timeslot
        ]
        for (source, target) in not_chosen_in_timeslot:
            if not self._link_feasible_in_timeslot(source, target, timeslot):
                self.remove_link(source, target, timeslot)

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
        transmissions = self._transmissions_at[timeslot]

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

        return wsignal.sinr(
            received_signal_dbm, received_interference_dbm, noise_floor_dbm
        )

    def _add_outedges(self, enode: ENode, timeslot: int):
        """Connect a new ENode to all its possible successors"""
        # avoid going in circles within a link embedding
        already_visited = [(enode.acting_as, enode.node)]
        cur = enode
        while cur.predecessor is not None:
            cur = cur.predecessor
            already_visited.append((cur.acting_as, cur.node))

        # could go to any node as a relay
        for node in self.infra.nodes():
            # Avoid circles. It is important that we can reliably
            # determine when there are no more actions to take and the
            # embedding is "failed", circles make that hard.
            if (enode.acting_as, node) not in already_visited:
                self.add_edge(enode, ENode(None, node), timeslot)

        out_edges = self.overlay.graph.out_edges(nbunch=[enode.acting_as])

        for (_, v) in out_edges:
            for option in self._by_block[v]:
                # do not connect to used relays, do not go in circles
                if (
                    not option.relay
                    and (v, option.node) not in already_visited
                ):
                    self.add_edge(enode, option, timeslot)

    def add_timeslot(self):
        """Adds a new timeslot as an option"""
        self.used_timeslots += 1
        for enode in self.nodes():
            self._add_outedges(enode, self.used_timeslots)

    def take_action(self, source: ENode, target: ENode, timeslot: int):
        """Take an action represented by an edge and update the graph"""
        if (source, target, timeslot) not in self.possibilities():
            return False

        # this should never be false, that would be a bug
        assert self._link_feasible(source, target, timeslot)

        self._taken_edges[(source, target)] = timeslot
        if target.relay:
            self._remove_links_between(source, target)
            target = ENode(
                block=target.block, node=target.node, predecessor=source
            )
            self.add_node(target)
            self.add_edge(source, target, timeslot)

        self.choose_node(target)
        self.choose_edge(source, target, timeslot)

        if timeslot >= self.used_timeslots:
            self.add_timeslot()
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
