"""Various candidates for node and edge features"""

from embedding import PartialEmbedding, ENode


def frac(a, b):
    """Regular fraction, but result is 0 if a = b = 0"""
    if a == 0 and b == 0:
        return 0
    return a / b


class Feature:
    """A feature extractor"""

    def __init__(self, name, compute_fun, dim=1):
        self.name = name
        self.dim = dim
        self.compute_fun = compute_fun

    def compute(self, *args, **kwargs):
        """Extract a feature"""
        feature = self.compute_fun(*args, **kwargs)

        # always return an iterable
        if not hasattr(feature, "__len__"):
            feature = (feature,)

        # transparently convert bools etc. to float
        feature = [float(value) for value in feature]

        assert len(feature) == self.dim
        return feature


class NodeFeature(Feature):
    """A node feature extractor"""

    def __init__(self, name, compute_fun, dim=1):
        super().__init__("node_feature_" + name, compute_fun, dim)


class EdgeFeature(Feature):
    """An edge feature extractor"""

    def __init__(self, name, compute_fun, dim=1):
        super().__init__("edge_feature_" + name, compute_fun, dim)


def _embeddable_after(embedding: PartialEmbedding, enode: ENode):
    weight = embedding.overlay.requirement(enode.block)
    remaining = embedding.remaining_capacity(enode.node)
    embedded = set(embedding.taken_embeddings.keys()).union([enode.block])
    options = set(embedding.overlay.blocks()).difference(embedded)
    remaining_after = remaining - weight
    embeddable = {
        option
        for option in options
        if embedding.overlay.requirement(option) < remaining_after
    }
    nropt = len(options)
    return len(embeddable) / nropt if nropt > 0 else 1


SUPPORTED_NODE_FEATURES = [
    NodeFeature(
        "pos",
        lambda emb, enode: emb.infra.graph.node[enode.node]["pos"],
        dim=2,
    ),
    NodeFeature("relay", lambda emb, enode: enode.relay),
    NodeFeature("sink", lambda emb, enode: enode.node == emb.infra.sink),
    NodeFeature(
        "remaining_capacity",
        lambda emb, enode: frac(
            emb.overlay.requirement(enode.block),
            emb.remaining_capacity(enode.node),
        ),
    ),
    NodeFeature(
        "weight", lambda emb, enode: emb.overlay.requirement(enode.block)
    ),
    NodeFeature(
        "compute_fraction",
        lambda emb, enode: frac(
            emb.overlay.requirement(enode.block),
            emb.remaining_capacity(enode.node),
        ),
    ),
    NodeFeature("embeddable_after", _embeddable_after),
]


def _is_broadcast(embedding, source, _target, timeslot, _edge_data):
    for (other_so, _other_ta) in embedding.taken_edges_in[timeslot]:
        if other_so.block == source.block and other_so.node == source.node:
            return True
    return False


SUPPORTED_EDGE_FEATURES = [
    EdgeFeature("timeslot", lambda emb, u, v, t, d: t),
    EdgeFeature("chosen", lambda emb, u, v, t, d: d["chosen"]),
    EdgeFeature(
        "capacity",
        lambda emb, u, v, t, d: emb.known_capacity(u.node, v.node, t),
    ),
    EdgeFeature(
        "additional_timeslot", lambda emb, u, v, t, d: t >= emb.used_timeslots
    ),
    EdgeFeature(
        "datarate_requirement",
        lambda emb, u, v, t, d: t >= emb.overlay.datarate(u.block),
    ),
    EdgeFeature(
        "datarate_fraction",
        lambda emb, u, v, t, d: t
        >= frac(
            emb.overlay.datarate(u.block),
            emb.known_capacity(u.node, v.node, t),
        ),
    ),
    EdgeFeature("is_broadcast", _is_broadcast),
]
