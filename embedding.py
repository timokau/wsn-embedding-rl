"""Model of wireless overlay networks"""

from typing import List, Tuple, Iterable
from collections import defaultdict
import math

import networkx as nx

from infrastructure import InfrastructureNetwork
from overlay import OverlayNetwork

DEBUG = False


class ENode:
    """A node representing a possible or actual embedding"""

    def __init__(self, acting_as, node, target=None):
        self.node = node
        self.acting_as = acting_as
        self.target = target

        self.relay = self.target is not None
        self.block = self.acting_as if self.target is None else None

        if not self.relay:
            self.target = self.acting_as

        self._hash = None

    def __repr__(self):
        result = ""
        if not self.relay:
            result += str(self.block) + "-"
        elif self.acting_as is not None:
            result += f"({self.acting_as})-"
        result += str(self.node)
        if self.relay:
            result += f"-({self.target})"
        return result

    def __eq__(self, other):
        if not isinstance(other, ENode):
            return False

        return (
            self.acting_as == other.acting_as
            and self.node == other.node
            and self.target == other.target
        )

    def __lt__(self, other):
        # For deterministic sorting
        repr_tupl = (self.acting_as, self.node, self.target)
        other_repr = (other.acting_as, other.node, other.target)
        return repr_tupl < other_repr

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        # cache the hashes, since hashing is quite expensive
        if self._hash is None:
            self._hash = hash((self.acting_as, self.node, self.target))
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
        # per-timeslot, more scalable
        self.taken_edges_in = defaultdict(set)
        self._nodes_sending_in = defaultdict(set)
        self.taken_embeddings = dict()
        self._num_outlinks_embedded = defaultdict(int)
        self._capacity_used = defaultdict(float)
        self._transmissions_at = defaultdict(list)
        self.link_embeddings = dict()
        self.finished_embeddings = set()

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
        # make sure the order is deterministic
        possibilities.sort(key=lambda pos: (pos[0], pos[1], pos[2]))
        return possibilities

    def nodes(self):
        """Shortcut to get all ENodes of the underlying graph"""
        return self.graph.nodes()

    def remaining_capacity(self, node):
        """Returns the capacity remaining in a node"""
        available = self.infra.capacity(node)
        used = self._capacity_used[node]
        return available - used

    def _add_possible_intermediate_embeddings(self):
        for block in self.overlay.intermediates:
            for node in self.infra.graph.nodes():
                self.try_add_enode(ENode(block, node))

    def _embed_sources(self, source_mapping: List[Tuple[str, str]]):
        for (block, node) in source_mapping:
            embedding = ENode(block, node)
            self.add_enode(embedding)
            self.choose_embedding(embedding)

    def _embed_sink(self):
        osink = self.overlay.sink
        isink = self.infra.sink
        embedding = ENode(osink, isink)
        self.add_enode(embedding)
        self.choose_embedding(embedding)

    def _add_relay_nodes(self):
        for (u, v) in self.overlay.links():
            for node in self.infra.nodes():
                enode = ENode(u, node, v)
                self.add_enode(enode)

    def try_add_enode(self, enode: ENode):
        """Adds a given ENode to the graph"""
        if not self._node_can_carry(enode.node, enode.block):
            return False

        self._by_node[enode.node].add(enode)
        self._by_block[enode.acting_as].add(enode)

        kind = "intermediate"
        if enode.block in self.overlay.sources:
            kind = "source"
        elif enode.block == self.overlay.sink:
            kind = "sink"

        self.graph.add_node(enode, chosen=False, relay=enode.relay, kind=kind)

        # add the necessary edges
        for ts in range(self.used_timeslots + 1):
            self._add_outedges(enode, ts)

        return True

    def add_enode(self, enode: ENode):
        """Adds an enode, failing if it is not possible"""
        assert self.try_add_enode(enode)

    def choose_embedding(self, enode: ENode):
        """Marks an potential embedding as chosen and updates the rest
        of the graph with the consequences. Should only ever be done
        when the enode is the source, the sink or has an incoming chosen
        edge."""
        if self.graph.node[enode]["chosen"]:
            return

        self.graph.node[enode]["chosen"] = True
        requirement = self.overlay.requirement(enode.block)
        assert requirement <= self.infra.capacity(enode.node)
        self._capacity_used[enode.node] += requirement
        for other in list(self._by_node[enode.node]):
            if self.graph.node[other]["chosen"]:
                continue
            if not self._node_can_carry(other.node, other.block):
                self.remove_enode(other)

        if not enode.relay:
            self.taken_embeddings[enode.block] = enode

            # remove other options for embedding this block
            for option in list(self._by_block[enode.block]):
                if option != enode and not option.relay:
                    self.remove_enode(option)

            # remove unnecessary relays going over the same node
            for block in self.overlay.blocks():
                for option in [
                    ENode(block, enode.node, enode.acting_as),
                    ENode(enode.acting_as, enode.node, block),
                ]:
                    if self.graph.has_node(option):
                        self.remove_enode(option)

    def try_add_edge(self, source: ENode, target: ENode, timeslot: int):
        """Tries to a possible connection to the graph if it is
        feasible."""
        # networkx will silently create nodes, which can just as
        # silently introduce bugs
        assert self.graph.has_node(source)
        assert self.graph.has_node(target)

        # this kind of edge can never be feasible, therefore trying to
        # add it would be a bug
        assert not target.relay or target.acting_as == source.acting_as
        assert not source.relay or source.target == target.target

        self.graph.add_edge(
            source,
            target,
            chosen=False,
            timeslot=timeslot,
            # edges are uniquely identified by (source, target, timeslot)
            key=timeslot,
        )
        if not self._connection_feasible(source, target, timeslot):
            self.remove_connection(source, target, timeslot)
            return False
        return True

    def add_edge(self, source: ENode, target: ENode, timeslot: int):
        """Adds a possible connection to the graph. Fails if it is not
        feasible."""
        assert self.try_add_edge(source, target, timeslot)

    def choose_edge(self, source: ENode, target: ENode, timeslot: int):
        """Marks a potential connection as chosen and updates the rest
        of the graph with the consequences. Should only ever be done
        when the source node is already chosen."""
        # pylint: disable=too-many-branches,too-many-statements
        assert self.graph.node[source]["chosen"]
        self.graph.edges[(source, target, timeslot)]["chosen"] = True
        if source.node != target.node:
            self._nodes_sending_in[timeslot].add(source.node)

        link = (source.acting_as, target.target)
        # if this starts a link embedding
        if not source.relay:
            self.link_embeddings[link] = [(source, None)]
        self.link_embeddings[link].append((target, timeslot))

        # update other edges that may have represented this link
        for (u, v, d) in list(
            self.graph.out_edges(
                nbunch=self._by_block[source.acting_as], data=True
            )
        ):
            if not d["chosen"] and not self._connection_necessary(u, v):
                self.remove_connection(u, v, d["timeslot"])

        # if this finishes a link embedding
        if not target.relay:
            self.finished_embeddings.add(link)
            for node in self.infra.nodes():
                relay = ENode(link[0], node, link[1])
                data = self.graph.nodes.get(relay)
                if data is not None and not data["chosen"]:
                    self.remove_enode(relay)

        if source.relay:
            self._remove_other_connections_from(source)

        self._transmissions_at[timeslot].append((source, target))
        self._remove_connections_infeasible_in(timeslot)

    def _build_possibilities_graph(
        self, source_mapping: List[Tuple[str, str]]
    ):
        self._add_possible_intermediate_embeddings()
        self._add_relay_nodes()
        self._embed_sink()
        self._embed_sources(source_mapping)
        self.add_timeslot()
        self._remove_unnecessary_nodes()

    def remove_connection(self, source: ENode, target: ENode, timeslot: int):
        """Removes a connection given its source, target and timeslot"""
        assert not self.graph.edges[(source, target, timeslot)]["chosen"]
        self.graph.remove_edge(source, target, timeslot)

    def remove_enode(self, enode: ENode):
        """Removes a node if it is not chosen"""
        if self.graph.nodes[enode]["chosen"]:
            return False
        self._by_block[enode.acting_as].remove(enode)
        self._by_node[enode.node].remove(enode)
        self.graph.remove_node(enode)
        return True

    def _invalidates_chosen(self, source, timeslot):
        """Checks if node sending would invalidate datarate of chosen action"""
        for (u, v) in self._transmissions_at[timeslot]:
            new_capacity = self.known_capacity(
                u.node,
                v.node,
                timeslot=timeslot,
                additional_senders={source.node},
            )
            thresh = self.overlay.datarate(u.acting_as)
            if new_capacity < thresh:
                return True
        return False

    def _datarate_valid(self, source, target, timeslot):
        """Checks if connection datarate is valid"""
        thresh = self.overlay.datarate(source.acting_as)
        capacity = self.known_capacity(source.node, target.node, timeslot)
        return capacity >= thresh

    def _node_already_visited_on_path(self, source, target):
        # self loops within a block are okay
        if not source.relay and not target.relay:
            return False

        link = (source.acting_as, target.target)
        visited_nodes = [
            n.node for (n, t) in self.link_embeddings.get(link, [])
        ]
        return target.node in visited_nodes

    def _path_already_started(self, source, target):
        if source.relay:
            return False
        link = (source.acting_as, target.target)
        if self.link_embeddings.get(link) is not None:
            return True
        return False

    def _source_already_in_path(self, source, target):
        link = (source.acting_as, target.target)
        path = self.link_embeddings.get(link, [])
        node_path = [enode.node for (enode, _ts) in path]
        # last one doesn't count
        return source.node in node_path[:-1]

    def _connection_necessary(self, source, target):
        (unnecessary, _reason) = self._why_connection_not_necessary(
            source, target
        )
        return not unnecessary

    def _why_connection_not_necessary(self, source, target):
        """Returns why an edge is or is not necessary"""
        if self._node_already_visited_on_path(source, target):
            return (True, "Node already visited on path")
        if self._path_already_started(source, target):
            return (True, "Path already started")
        if self._source_already_in_path(source, target):
            return (True, "Source already in path")
        return (False, "")

    def _connection_feasible(self, source, target, timeslot):
        (infeasible, _reason) = self.why_infeasible(source, target, timeslot)
        return not infeasible

    def why_infeasible(self, source, target, timeslot):
        """Returns why an edge is or is not infeasible

        Intended for debugging.
        """
        (infeasible_in_ts, reason) = self._why_infeasible_in_timeslot(
            source, target, timeslot
        )
        if infeasible_in_ts:
            return (True, reason)
        (unneccessary, reason) = self._why_connection_not_necessary(
            source, target
        )
        if unneccessary:
            return (True, reason)
        return (False, "")

    def _node_can_carry(self, node, block):
        """Weather or not a node can support the computation for a block
        in a given timeslot"""
        needed = 0 if block is None else self.overlay.requirement(block)
        return needed <= self.remaining_capacity(node)

    def _node_sending_other_data_in_timeslot(self, enode, timeslot):
        # the same embedding can send multiple times within a timeslot
        # (broadcasting results), but others cannot (which would send
        # other data)
        for (u, v) in self.taken_edges_in[timeslot]:
            # loops within a node are okay
            if (
                u.node == enode.node
                and u.acting_as != enode.acting_as
                and not u.node == v.node
            ):
                return True
        return False

    def _node_receiving_data_in_timeslot(self, node, timeslot):
        """We work on the half-duplex assumption: sending and receiving
        is mutually exclusive."""
        for (u, v) in self.taken_edges_in[timeslot]:
            # loops within a node are okay
            if v.node == node and u.node != v.node:
                return True
        return False

    def _connection_feasible_in_timeslot(self, source, target, timeslot):
        (infeasible, _reason) = self._why_infeasible_in_timeslot(
            source, target, timeslot
        )
        return not infeasible

    def _why_infeasible_in_timeslot(self, source, target, timeslot):
        """Returns why an edge is or is not feasible in a timeslot"""
        # loops are always valid
        if source.node == target.node:
            return (False, "")

        if self._node_sending_other_data_in_timeslot(source, timeslot):
            return (True, "Node already sending other data in timeslot")

        if self._node_receiving_data_in_timeslot(source.node, timeslot):
            return (True, "Node already receiving data in timeslot")

        if not self._datarate_valid(source, target, timeslot):
            return (True, "Datarate is not valid")

        if self._invalidates_chosen(source, timeslot):
            return (True, "Invalidates an already chosen connection")

        return (False, "")

    def _remove_other_connections_from(self, enode):
        """Removes not-chosen outedges for an enode"""
        for (u, v, k, d) in list(
            self.graph.out_edges(nbunch=[enode], keys=True, data=True)
        ):
            if not d["chosen"]:
                self.remove_connection(u, v, k)

    def _remove_connections_between(self, source, target):
        """Removes all remaining unchosen connections between two ENodes"""
        for timeslot in range(self.used_timeslots + 1):
            if self.graph.has_edge(source, target, timeslot):
                chosen = self.graph.edges[source, target, timeslot]["chosen"]
                if not chosen:
                    self.remove_connection(source, target, timeslot)

    def _remove_connections_infeasible_in(self, timeslot):
        """Removes connections that are no longer feasible within a
        timeslot"""
        not_chosen_in_timeslot = [
            (source, target)
            for (source, target, data) in self.graph.out_edges(data=True)
            if not data["chosen"] and data["timeslot"] == timeslot
        ]
        for (source, target) in not_chosen_in_timeslot:
            if not self._connection_feasible_in_timeslot(
                source, target, timeslot
            ):
                self.remove_connection(source, target, timeslot)

    def known_capacity(
        self,
        source_node: str,
        target_node: str,
        timeslot: int,
        additional_senders: Iterable[str] = frozenset(),
    ):
        """
        Connection capacity assuming only already chosen edges and the
        currently considered edges are sending.
        """
        sinr = self.known_sinr(
            source_node, target_node, timeslot, additional_senders
        )
        bandwidth = self.infra.bandwidth
        shannon_capacity = bandwidth * math.log(1 + 10 ** (sinr / 10), 2)
        return shannon_capacity

    def known_sinr(
        self,
        source_node: str,
        target_node: str,
        timeslot: int,
        additional_senders: Iterable[str] = frozenset(),
    ):
        """SINR assuming only already chosen edges and the currently
        considered edges are sending"""
        senders = self._nodes_sending_in[timeslot].union(additional_senders)
        # always ignore the sending node in sinr calculations
        # (assuming broadcast, no self-interference)
        senders = frozenset(senders.difference({source_node}))
        return self.infra.sinr(source_node, target_node, senders)

    def _add_outedges(self, enode: ENode, timeslot: int):
        """Connect a new ENode to all its possible successors"""
        if enode.relay:
            outlinks = {(enode.acting_as, enode.target)}
        else:
            embedding_already_started = self.link_embeddings.keys()
            outlinks = set(
                self.overlay.graph.out_edges(nbunch=[enode.acting_as])
            ).difference(embedding_already_started)

        for (_, v) in outlinks:
            for node in self.infra.nodes():
                relay_target = ENode(enode.acting_as, node, v)
                if self.graph.nodes.get(relay_target) is not None:
                    self.try_add_edge(enode, relay_target, timeslot)
            for option in self._by_block[v]:
                if not option.relay:
                    self.try_add_edge(enode, option, timeslot)

    def add_timeslot(self):
        """Adds a new timeslot as an option"""
        self.used_timeslots += 1
        for enode in self.nodes():
            self._add_outedges(enode, self.used_timeslots)

    def _check_invariants(self):
        """For debugging only, slow"""
        # pylint: disable=too-many-return-statements,too-many-branches
        if not DEBUG:
            return (True, None)

        for (u, v, d) in self.graph.edges(data=True):
            t = d["timeslot"]
            chosen = d["chosen"]
            if not chosen and not self._connection_necessary(u, v):
                return (False, f"({u}, {v}, {t}) is not necessary")
            if not chosen and not self._connection_feasible_in_timeslot(
                u, v, t
            ):
                return (False, f"({u}, {v}, {t}) is not feasible")
            if (u.acting_as, v.target) not in self.overlay.links():
                return (False, f"({u}, {v}, {t}) does not represent any link")

        for enode in self.nodes():
            if enode.relay and self.graph.nodes[enode]["chosen"]:
                out_edges = self.graph.out_edges(nbunch=[enode], data=True)
                has_chosen_out = False
                for (u, v, d) in out_edges:
                    if d["chosen"]:
                        has_chosen_out = True
                        break
                if has_chosen_out and len(list(out_edges)) > 1:
                    return (
                        False,
                        f"Relay {enode} has too many out edges: {out_edges}",
                    )

        for (enode, deg) in self.graph.in_degree():
            if enode.relay and self.graph.nodes[enode]["chosen"]:
                if deg != 1:
                    return (False, f"Chosen relay {enode} has indeg {deg}")

        last_ts = self.used_timeslots
        poss = self.possibilities()
        for (u, v, t) in poss:
            # if an action is possible in any timeslot, it should also
            # be possible in the last timeslot
            assert (u, v, last_ts) in poss

        return (True, None)

    def _remove_unnecessary_nodes(self):
        removed = True
        while removed:
            indeg = self.graph.in_degree()
            removed = False
            for enode in [n for (n, deg) in indeg if deg == 0]:
                if self.remove_enode(enode):
                    removed = True

    def take_action(self, source: ENode, target: ENode, timeslot: int):
        """Take an action represented by an edge and update the graph"""
        if (source, target, timeslot) not in self.possibilities():
            return False

        # this should never be false, that would be a bug
        assert self._connection_feasible_in_timeslot(source, target, timeslot)
        assert self._connection_necessary(source, target)
        assert self._connection_feasible(source, target, timeslot)

        self._taken_edges[(source, target)] = timeslot
        self.taken_edges_in[timeslot].add((source, target))

        self.choose_embedding(target)
        self.choose_edge(source, target, timeslot)

        if timeslot >= self.used_timeslots:
            self.add_timeslot()

        self._remove_unnecessary_nodes()

        (result, reason) = self._check_invariants()
        if not result:
            raise Exception(
                # pylint: disable=line-too-long
                f"Action ({source}, {target}, {timeslot}) violates invariants: {reason}"
            )
        return True

    def is_complete(self):
        """Determines if all blocks and links are embedded"""
        return set(self.overlay.links()) == self.finished_embeddings

    def __str__(self):
        result = "Embedding with:\n"
        result += self.infra.__str__()
        result += self.overlay.__str__()
        return result

    def construct_link_mappings(self):
        """Returns a mapping from links to paths"""
        result = dict()
        for link in self.finished_embeddings:
            result[link] = self.link_embeddings[link]
        return result
