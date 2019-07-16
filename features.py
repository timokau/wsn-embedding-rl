"""Various candidates for node and edge features"""

from embedding import PartialEmbedding, ENode


def frac(a, b):
    """Regular fraction, but result is 0 if a = b = 0"""
    if a == 0 and b == 0:
        return 0
    return a / b


class Feature:
    """A feature extractor"""

    def __init__(self, name, dim):
        self.name = name
        self.dim = dim


class NodeFeature(Feature):
    """A node feature extractor"""

    def __init__(self, name, compute_fun, dim=1):
        super().__init__("node_feature_" + name, dim)
        self.compute_fun = compute_fun

    def compute(self, embedding: PartialEmbedding, enode: ENode):
        """Extract a feature from a node"""
        feature = self.compute_fun(embedding, enode)

        # always return an iterable
        if not hasattr(feature, "__len__"):
            feature = (feature,)

        # transparently convert bools etc. to float
        feature = [float(value) for value in feature]

        assert len(feature) == self.dim
        return feature


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
