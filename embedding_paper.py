"""Implements the exact constraints described in the paper. Pure, but slow."""
from itertools import chain, combinations
from embedding import PartialEmbedding

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


def sb(edge):
    (_n, bs, _bt) = edge
    return bs


def tb(edge):
    (_n, _bs, bt) = edge
    return bt


class Wrapper:
    def __init__(self, embedding: PartialEmbedding):
        self.embedding = embedding
        self.V = [_enode_to_triple(enode) for enode in embedding.nodes()]
        self.N = embedding.infra.nodes()
        self.B = embedding.overlay.blocks()
        self.L = embedding.overlay.graph.edges()
        self.S = embedding.overlay.sources
        self.bsink = embedding.overlay.sink
        self.R = [
            (_enode_to_triple(source), _enode_to_triple(target), timeslot)
            for ((source, target), timeslot) in embedding.taken_edges.items()
        ]

    def P(self, b):
        if b == self.bsink:
            return self.embedding.infra.sink
        for (block, node) in self.embedding.source_mapping:
            if block == b:
                return node
        raise Exception("P called on invalid block")

    def W(self, b):
        return self.embedding.overlay.requirement(b)

    def M(self, n, R):
        return {bs for (n_, bs, bt) in self.Phat(R) if n == n_ and bs == bt}

    def C(self, n):
        return self.embedding.infra.capacity(n)

    def is_in_vplace(self, n, bs, bt, R):
        a = bs == bt
        b = n in self.N
        c = bs in self.B
        d = sum(
            [self.W(block) for block in self.M(n, R).union((bs,))]
        ) <= self.C(n)
        e = not self.placedElsewhere((n, bs, bt), R)
        f = not exists_elem(
            self.Phat(R),
            lambda enode: enode[0] == n
            and (sb(enode) == bs or tb(enode) == bt)
            and sb(enode) != tb(enode),
        )
        result = a and b and c and d and e and f
        return result

    def is_in_vrelay(self, n, bs, bt, R):
        return (
            (n in self.N)
            and (bs in self.B)
            and (bt in self.B)
            and (bs, bt) in self.L
            and not self.routedElsewhere((n, bs, bt), R)
            and bs not in self.M(n, R)
            and bt not in self.M(n, R)
        )

    def is_in_v(self, n, bs, bt, R):
        return self.is_in_vplace(n, bs, bt, R) or self.is_in_vrelay(
            n, bs, bt, R
        )

    def routedElsewhere(self, e, R):
        return e not in self.Phat(R) and self.alreadyRouted(e[1], e[2], R)

    def alreadyRouted(self, bs, bt, R):
        # exists subset such that
        value = _find_elem(
            powerset(R), lambda path: self.isPathBetween(path, bs, bt)
        )
        return value is not None

    def enodes(self, R):
        result = set()
        for (u, v, _t) in R:
            result.add(u)
            result.add(v)
        return result

    def Phat(self, R):
        enodes = self.enodes(R)
        sources = {(self.P(b), b, b) for b in self.S}
        sink = set(((self.P(self.bsink), self.bsink, self.bsink),))
        return enodes.union(sources).union(sink)

    def alreadyPlaced(self, b, R):
        elem = _find_elem(
            self.Phat(R), lambda enode: enode[1] == enode[2] == b
        )
        if elem is not None:
            pass
            # print(f"Found {b} placed on {elem}")
        return elem is not None

    def placedElsewhere(self, e, R):
        result = (
            sb(e) == tb(e)
            and self.alreadyPlaced(sb(e), R)
            and e not in self.Phat(R)
        )
        if result:
            pass
            # print(f"Determined that {e} is already placed elsewhere")
        else:
            pass
            # print(f"Determined that {e} is NOT already placed elsewhere")
        return result

    def visitsPlacement(self, path, b):
        return exists_elem(path, lambda edge: self.isPlacementFor(edge[0], b))

    def isPlacementFor(self, e, b):
        return sb(e) == b and tb(e) == b

    def isPathBetween(self, path, bs, bt):
        return (
            exists_elem(path, lambda edge: sb(edge[0]) == tb(edge[0]) == bs)
            and len(path) > 0
            and true_for_all(
                path,
                lambda edge: (
                    self.isPlacementFor(edge[0], bs)
                    or sb(edge[0]) != tb(edge[0])
                )
                and (
                    self.isPlacementFor(edge[1], bt)
                    or (
                        sb(edge[1]) != tb(edge[1])
                        and self.hasEdgeFrom(path, edge[1])
                    )
                ),
            )
        )

    def hasEdgeFrom(self, path, e):
        return exists_elem(path, lambda edge: edge[0] == e)

    def verify_v(self):
        for n in self.N:
            for bs in self.B:
                for bt in self.B:
                    enode = (n, bs, bt)
                    should_exist = self.is_in_v(*enode, self.R)
                    does_exist = enode in self.V
                    if should_exist and not does_exist:
                        return (False, f"{enode} should exist but doesn't")
                    if not should_exist and does_exist:
                        return (False, f"{enode} shouldn't exist but does")
        return (True, "")
