"""Implements the exact constraints described in the paper. Pure, but slow."""
from itertools import chain, combinations, permutations, islice
from functools import lru_cache
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


def no(edge):
    (n, _bs, _bt) = edge
    return n


def sb(edge):
    (_n, bs, _bt) = edge
    return bs


def tb(edge):
    (_n, _bs, bt) = edge
    return bt


@lru_cache(maxsize=2 ** 23)
def _check_path_consistency(path):
    # recursion for more caching
    if len(path) == 0:
        return False
    if len(path) == 1:
        return True

    first_target = path[0][1]
    second_source = path[1][0]
    if first_target != second_source:
        return False
    return _check_path_consistency(path[1:])


@lru_cache(maxsize=None)
def _paths_for_subset(subset):
    return {
        permutation
        for permutation in permutations(subset)
        if _check_path_consistency(permutation)
    }


class WayTooBigException(Exception):
    pass


@lru_cache()
def _paths(A):
    print("Recomputing")
    # doesn't get much more inefficient than this
    result = set()

    # take care not to OOM when counting the elements
    subsets = len(list(islice(powerset(A), 2 ** 13)))
    print(subsets)
    if subsets > 2 ** 12:  # takes too long
        raise WayTooBigException("Won't even try.")

    for i, subset in enumerate(powerset(A)):
        if i % 1000 == 0:
            print(_check_path_consistency.cache_info())
        result = result.union(_paths_for_subset(subset))
    print("Done")
    return result


class Wrapper:
    def __init__(self, embedding: PartialEmbedding):
        self.embedding = embedding
        self.V = [_enode_to_triple(enode) for enode in embedding.nodes()]
        self.N = embedding.infra.nodes()
        self.B = embedding.overlay.blocks()
        self.L = embedding.overlay.graph.edges()
        self.S = embedding.overlay.sources
        self.bsink = embedding.overlay.sink
        self.A = [
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

    def Phat(self, A):
        enodes = set()
        for (u, v, _t) in A:
            enodes.add(u)
            enodes.add(v)

        sources = {(self.P(b), b, b) for b in self.S}
        sink = set(((self.P(self.bsink), self.bsink, self.bsink),))
        return enodes.union(sources).union(sink)

    def paths(self, A):
        # wrapper for caching, as this is *really* inefficient but I
        # want to test the actual math, not some optimized version
        return _paths(frozenset(A))

    def routing(self, bs, bt, A):
        def _pathsource(path):
            # first action, first part of connection, source block
            return path[0][0]

        def _pathtarget(path):
            # last action, second part of connection, target block
            return path[-1][1]

        return {
            path
            for path in self.paths(A)
            if self.places(_pathsource(path), bs)
            and self.places(_pathtarget(path), bt)
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
        return e not in self.Phat(A) and len(self.routing(sb(e), tb(e), A)) > 0

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
                    try:
                        should_exist = self.enodeValid(enode, self.A)
                        does_exist = enode in self.V
                        if should_exist and not does_exist:
                            return (False, f"{enode} should exist but doesn't")
                        if not should_exist and does_exist:
                            return (False, f"{enode} shouldn't exist but does")
                    except WayTooBigException:
                        pass  # gave up on checking, too big
        return (True, "")
