"""Implements the exact constraints described in the paper. Pure, but slow."""
from itertools import chain, combinations, permutations, product
from functools import lru_cache, partial
import math
from embedding import PartialEmbedding, ENode

# This follows the naming and semantics of the paper as closely as
# possible.
# pylint: disable=invalid-name, missing-docstring, no-self-use
# pylint: disable=too-many-instance-attributes

from pycallgraph import PyCallGraph, Config, GlobbingFilter
from pycallgraph.output import GraphvizOutput
config = Config()
config.trace_filter = GlobbingFilter(exclude=["pycallgraph.*"])
graphviz = GraphvizOutput(output_file=f"pc.png")
pcg = PyCallGraph(output=graphviz, config=config)


def _enode_to_triple(enode):
    return (
        enode.node,
        enode.acting_as,
        enode.target if enode.relay else enode.acting_as,
    )


def exists_elem(iterable, predicate):
    """Existence quantifier"""
    return _find_elem(iterable, predicate) is not None


def _find_elem(iterable, predicate):
    for value in iterable:
        if predicate(value):
            return value
    return None


def true_for_all(iterable, predicate):
    for value in iterable:
        if not predicate(value):
            return False
    return True


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def no(edge):
    (n, _bs, _bt) = edge
    return n


def sb(edge):
    (_n, bs, _bt) = edge
    return bs


def tb(edge):
    (_n, _bs, bt) = edge
    return bt


class Wrapper:
    # pylint: disable=too-many-public-methods
    def __init__(self, embedding: PartialEmbedding):
        self.embedding = embedding
        self.V = frozenset(
            [_enode_to_triple(enode) for enode in embedding.nodes()]
        )
        self.E = frozenset(
            [
                (_enode_to_triple(u), _enode_to_triple(v), t)
                for (u, v, t) in embedding.graph.edges(keys=True)
            ]
        )
        self.N = frozenset(embedding.infra.nodes())
        self.B = frozenset(embedding.overlay.blocks())
        self.L = frozenset(embedding.overlay.graph.edges())
        self.S = frozenset(embedding.overlay.sources)
        self.bsink = embedding.overlay.sink
        self.A = frozenset(
            [
                (_enode_to_triple(source), _enode_to_triple(target), timeslot)
                for (
                    (source, target),
                    timeslot,
                ) in embedding.taken_edges.items()
            ]
        )
        # close enough to the variant in the paper
        self.U = {t for (u, v, t) in self.A}

    def P(self, b):
        if b == self.bsink:
            return self.embedding.infra.sink
        for (block, node) in self.embedding.source_mapping:
            if block == b:
                return node
        raise Exception("P called on invalid block")

    def W(self, b):
        return self.embedding.overlay.requirement(b)

    def R(self, b):
        return self.embedding.overlay.datarate(b)

    def D(self, n1, n2, ninf):
        sinr = self.embedding.infra.sinr(n1, n2, frozenset(ninf))
        bandwidth = self.embedding.infra.bandwidth
        shannon_capacity = bandwidth * math.log(1 + 10 ** (sinr / 10), 2)
        return shannon_capacity

    def Phat(self, A):
        enodes = set()
        for (u, v, _t) in A:
            enodes.add(u)
            enodes.add(v)

        sources = {(self.P(b), b, b) for b in self.S}
        sink = set(((self.P(self.bsink), self.bsink, self.bsink),))
        return enodes.union(sources).union(sink)

    def routing(self, bs, bt, A):
        return {
            (u, v, t)
            for (u, v, t) in A
            if (sb(u) == bs and tb(u) in {bs, bt})
            and (tb(v) == bt and sb(v) in {bs, bt})
        }

    def M(self, n, A):
        return {bs for (n_, bs, bt) in self.Phat(A) if n == n_ and bs == bt}

    def C(self, n):
        return self.embedding.infra.capacity(n)

    def placement(self, e):
        return no(e) in self.N and sb(e) in self.B and sb(e) == tb(e)

    def canCarry(self, n, b, A):
        return sum(
            [self.W(block) for block in self.M(n, A).union((b,))]
        ) <= self.C(n)

    def placedElsewhere(self, e, A):
        return exists_elem(
            self.N, lambda n: n != no(e) and sb(e) in self.M(n, A)
        )

    def relayed(self, b, n, A):
        return exists_elem(
            self.B,
            lambda bprime: b != bprime
            and (
                (n, b, bprime) in self.Phat(A)
                or (n, bprime, b) in self.Phat(A)
            ),
        )

    def placementValid(self, e, A):
        return (
            self.placement(e)
            and self.canCarry(no(e), sb(e), A)
            and not self.placedElsewhere(e, A)
            and not self.relayed(sb(e), no(e), A)
        )

    def links(self, e):
        return (sb(e), tb(e)) in self.L

    def routedElsewhere(self, e, A):
        return e not in self.Phat(A) and self.completelyRouted(sb(e), tb(e), A)

    def placed(self, e, A):
        return sb(e) in self.M(no(e), A) or tb(e) in self.M(no(e), A)

    def places(self, e, b):
        return sb(e) == b and tb(e) == b

    def relayValid(self, e, A):
        return (
            self.links(e)
            and not self.routedElsewhere(e, A)
            and not self.placed(e, A)
        )

    def enodeValid(self, e, A):
        return self.placementValid(e, A) or self.relayValid(e, A)

    def verify_v(self):
        # pcg.start(reset=False)
        for n in self.N:
            for bs in self.B:
                for bt in self.B:
                    enode = (n, bs, bt)
                    should_exist = self.enodeValid(enode, self.A)
                    does_exist = enode in self.V
                    if should_exist != does_exist:
                        self.print_state()
                    if should_exist and not does_exist:
                        return (False, f"{enode} should exist but doesn't")
                    if not should_exist and does_exist:
                        return (False, f"{enode} shouldn't exist but does")
        # pcg.done()
        return (True, "")

    def advancesPath(self, u, v, t, A):
        if self.placement(v):
            return True
        if exists_elem(A, lambda a: a != (u, v, t) and a[1] == v):
            # loop
            return False
        if not self.placement(u) and exists_elem(
            A, lambda a: a != (u, v, t) and a[0] == u
        ):
            # already continued from here
            return False
        return True

    def edgeRepresentsLink(self, e1, e2):
        return (sb(e1), tb(e2)) in self.L

    def consistent(self, e1, e2):
        source_block = sb(e1)
        target_block = tb(e2)
        if e1 == e2:
            # more specifically: if its a relay and the relay was already
            # visited
            return False
        return (
            (source_block, target_block) in self.L
            and tb(e1) in {target_block, sb(e1)}
            and sb(e2) in {source_block, tb(e2)}
        )

    def radioNecessary(self, e1, e2):
        return no(e1) != no(e2)

    def sendingData(self, n, t, A):
        return {
            sb(a[0])
            for a in A
            if a[2] == t and no(a[0]) == n and no(a[1]) != n
        }

    def receivingData(self, n, t, A):
        return {
            sb(a[0])
            for a in A
            if a[2] == t and no(a[1]) == n and no(a[0]) != n
        }

    def radiosFree(self, u, v, t, A):
        u_sends = self.sendingData(no(u), t, A)
        u_receives = self.receivingData(no(u), t, A)
        v_sends = self.sendingData(no(v), t, A)
        v_receives = self.receivingData(no(v), t, A)
        data = sb(u)
        if (
            len(u_sends.difference((data,))) > 0
            or len(v_receives.difference((data,))) > 0
            or len(u_receives) > 0
            or len(v_sends) > 0
        ):
            return False
        return True

    def T(self, t, A):
        return {
            a[0][0] for a in A if a[2] == t
        }

    def datarateMet(self, u, v, t, A):
        sending = self.T(t, A).difference((no(u),))
        datarate_available = self.D(no(u), no(v), sending)
        required = self.R(sb(u))
        return datarate_available >= required

    def timeslotExists(self, t):
        return t == 0 or (t - 1) in self.U

    def restartsPath(self, u, v, t, A):
        link = (sb(u), tb(v))
        return self.placement(u) and exists_elem(
            A,
            lambda a: a != (u, v, t)
            and a[0] == u
            and sb(a[1]) == link[0]
            and tb(a[1]) == link[1],
        )

    def completelyRouted(self, bs, bt, A):
        routing = self.routing(bs, bt, A)
        return exists_elem(
            routing, lambda a: self.places(a[0], bs)
        ) and exists_elem(routing, lambda a: self.places(a[1], bt))

    def alreadyRoutedOtherwise(self, u, v, t, A):
        if self.completelyRouted(sb(u), tb(v), A):
            r = self.routing(sb(u), tb(v), A)
            if (u, v, t) not in r:
                return True
        return False

    def edgeValid(self, u, v, t, A):
        return (
            self.timeslotExists(t)
            and self.consistent(u, v)
            and (not self.radioNecessary(u, v) or self.radiosFree(u, v, t, A))
            and self.datarateMet(u, v, t, A)
            and not self.alreadyTakenInOtherTs(u, v, t, A)
            and not self.alreadyRoutedOtherwise(u, v, t, A)
            and not self.restartsPath(u, v, t, A)
            and self.advancesPath(u, v, t, A)
        )

    def alreadyTakenInOtherTs(self, u, v, t, A):
        return exists_elem(A, lambda a: a[0] == u and a[1] == v and a[2] != t)

    def _all_other_remain_valid(self, u, v, t, A):
        return true_for_all(
            A, lambda a: self.edgeValid(*a, A.union(((u, v, t),)))
        )

    def edgeInE(self, u, v, t, A):
        return self.edgeValid(u, v, t, A) and self._all_other_remain_valid(
            u, v, t, A
        )

    def print_state(self):
        return
        # to debug mismatches
        print("====Actions====")
        for a in self.A:
            print(a)

        print("====Links====")
        print(self.L)

        print("====Embedding====")
        print(self.embedding)

    def verify_e(self):
        # pcg.start(reset=False)
        for u in self.V:
            for v in self.V:
                for t in range(max(self.U) + 5):  # testing all N is not quite practical
                    should_exist = self.edgeInE(u, v, t, self.A)
                    does_exist = (u, v, t) in self.E
                    if should_exist != does_exist:
                        self.print_state()
                    if should_exist and not does_exist:
                        reason = self.embedding.why_infeasible(
                            ENode(sb(u), no(u), tb(u)),
                            ENode(sb(v), no(v), tb(v)),
                            t,
                        )[1]
                        return (
                            False,
                            f"{u}, {v}, {t} should exist but doesn't"
                            f" because: {reason}",
                        )
        # pcg.done()
        return (True, "")
