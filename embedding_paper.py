"""Implements the exact constraints described in the paper. Pure, but slow."""
import math
from embedding import PartialEmbedding, ENode

# This follows the naming and semantics of the paper as closely as
# possible.
# pylint: disable=invalid-name, missing-docstring, no-self-use
# pylint: disable=too-many-instance-attributes


def _enode_to_triple(enode):
    return (
        enode.node,
        enode.acting_as,
        enode.target if enode.relay else enode.acting_as,
    )


def exists_elem(iterable, predicate):
    """Existence quantifier"""
    for value in iterable:
        if predicate(value):
            return True
    return False


def true_for_all(iterable, predicate):
    return not exists_elem(iterable, lambda x: not predicate(x))


def no(enode):
    (n, _bs, _bt) = enode
    return n


def sb(enode):
    (_n, bs, _bt) = enode
    return bs


def tb(enode):
    (_n, _bs, bt) = enode
    return bt


def so(edge):
    return edge[0]


def ta(edge):
    return edge[1]


def k(edge):
    return edge[2]


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
            if (self.places(u, bs) or sb(u) == bs and tb(u) == bt)
            and (self.places(v, bt) or sb(v) == bs and tb(v) == bt)
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
        return (True, "")

    def alreadyReached(self, b, e, A):
        return exists_elem(A, lambda a: ta(a) == e and sb(so(a)) == b)

    def alreadyForwarded(self, b, e, A):
        return exists_elem(A, lambda a: so(a) == e and tb(ta(a)) == b)

    def advancesPath(self, u, v, t, A):
        if (u, v, t) in A:
            return True
        if u == v:
            return False
        if self.alreadyReached(sb(u), v, A):
            return False
        if self.alreadyForwarded(tb(v), u, A):
            return False
        return True

    def edgeRepresentsLink(self, u, v):
        return (sb(u), tb(v)) in self.L

    def consistent(self, u, v):
        return (self.placement(u) or tb(u) == tb(v)) and (
            self.placement(v) or sb(v) == sb(u)
        )

    def sendingData(self, n, t, A):
        return {
            sb(so(a))
            for a in A
            if k(a) == t and no(so(a)) == n and no(ta(a)) != n
        }

    def receivingData(self, n, t, A):
        return {
            sb(so(a))
            for a in A
            if k(a) == t and no(ta(a)) == n and no(so(a)) != n
        }

    def transmissionPossible(self, u, v, t, A):
        return no(u) == no(v) or (
            self.radiosFree(u, v, t, A) and self.datarateMet(u, v, t, A)
        )

    def radiosFree(self, u, v, t, A):
        return (
            self.sendingData(no(u), t, A).issubset((sb(u),))
            and self.receivingData(no(u), t, A) == set()
            and self.sendingData(no(v), t, A) == set()
            and self.receivingData(no(v), t, A).issubset((sb(u),))
        )

    def T(self, t, A):
        return {no(so(a)) for a in A if k(a) == t and no(so(a)) != no(ta(a))}

    def datarateMet(self, u, v, t, A):
        sending = self.T(t, A).difference((no(u),))
        datarate_available = self.D(no(u), no(v), sending)
        required = self.R(sb(u))
        return datarate_available >= required

    def timeslotExists(self, t):
        return t == 0 or (t - 1) in self.U

    def completelyRouted(self, bs, bt, A):
        routing = self.routing(bs, bt, A)
        return exists_elem(
            routing, lambda a: self.places(so(a), bs)
        ) and exists_elem(routing, lambda a: self.places(ta(a), bt))

    def edgeValid(self, u, v, t, A):
        return (
            self.timeslotExists(t)
            and self.edgeRepresentsLink(u, v)
            and self.consistent(u, v)
            and self.transmissionPossible(u, v, t, A)
            and self.advancesPath(u, v, t, A)
        )

    def _all_other_remain_valid(self, u, v, t, A):
        return true_for_all(
            A, lambda a: self.edgeValid(*a, A.union(((u, v, t),)))
        )

    def edgeInE(self, u, v, t, A):
        return self.edgeValid(u, v, t, A) and self._all_other_remain_valid(
            u, v, t, A
        )

    def print_state(self):
        # to debug mismatches
        print("====Actions====")
        for a in self.A:
            print(a)

        print("====Links====")
        print(self.L)

        print("====Embedding====")
        print(self.embedding)

    def verify_e(self):
        for u in self.V:
            for v in self.V:
                # testing all N is not quite practical
                for t in range(max(self.U) + 5):
                    should_exist = self.edgeInE(u, v, t, self.A)
                    does_exist = (u, v, t) in self.E
                    if should_exist != does_exist:
                        self.print_state()
                    if does_exist and not should_exist:
                        return (
                            False,
                            f"{u}, {v}, {t} shouldn't exist but does",
                        )
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
        return (True, "")
