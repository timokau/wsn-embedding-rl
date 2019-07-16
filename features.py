"""Various candidates for node and edge features"""

from embedding import PartialEmbedding, ENode

SUPPORTED_NODE_FEATURES = set()


def frac(a, b):
    """Regular fraction, but result is 0 if a = b = 0"""
    if a == 0 and b == 0:
        return 0
    return a / b


class Feature:
    """A feature extractor"""

    def __init__(self, dim, name):
        self.dim = dim
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return hash(self) == hash(other)


class NodeFeature(Feature):
    """A node feature extractor"""

    def __init__(self, dim, name):
        super().__init__(dim, "node_feature_" + name)

    def compute(self, embedding: PartialEmbedding, enode: ENode):
        """Extract a feature from a node"""
        feature = self._compute(embedding, enode)

        if not hasattr(feature, "__len__"):
            feature = (feature,)

        feature = [float(value) for value in feature]

        assert len(feature) == self.dim
        return feature

    def _compute(self, embedding: PartialEmbedding, enode: ENode):
        raise NotImplementedError()


class PosFeature(NodeFeature):
    """The position of a node"""

    def __init__(self):
        super().__init__(2, "pos")

    def _compute(self, embedding: PartialEmbedding, enode: ENode):
        return embedding.infra.graph.node[enode.node]["pos"]


SUPPORTED_NODE_FEATURES.add(PosFeature())


class NodeIsRelayFeature(NodeFeature):
    """Whether or not a node is a relay"""

    def __init__(self):
        super().__init__(1, "relay")

    def _compute(self, embedding: PartialEmbedding, enode: ENode):
        return enode.relay


SUPPORTED_NODE_FEATURES.add(NodeIsRelayFeature())


class NodeIsSinkFeature(NodeFeature):
    """Whether or not a node is the sink"""

    def __init__(self):
        super().__init__(1, "sink")

    def _compute(self, embedding: PartialEmbedding, enode: ENode):
        return enode.node == embedding.infra.sink


SUPPORTED_NODE_FEATURES.add(NodeIsSinkFeature())


class NodeRemainingCapacityFeature(NodeFeature):
    """Whether or not a node is the sink"""

    def __init__(self):
        super().__init__(1, "remaining_capacity")

    def _compute(self, embedding: PartialEmbedding, enode: ENode):
        weight = embedding.overlay.requirement(enode.block)
        remaining = embedding.remaining_capacity(enode.node)
        return frac(weight, remaining)


SUPPORTED_NODE_FEATURES.add(NodeRemainingCapacityFeature())


class NodeWeightFeature(NodeFeature):
    """Whether or not a node is the sink"""

    def __init__(self):
        super().__init__(1, "weight")

    def _compute(self, embedding: PartialEmbedding, enode: ENode):
        return embedding.overlay.requirement(enode.block)


SUPPORTED_NODE_FEATURES.add(NodeWeightFeature())


class NodeComputeFractionFeature(NodeFeature):
    """Whether or not a node is the sink"""

    def __init__(self):
        super().__init__(1, "compute_fraction")

    def _compute(self, embedding: PartialEmbedding, enode: ENode):
        weight = embedding.overlay.requirement(enode.block)
        remaining = embedding.remaining_capacity(enode.node)
        return frac(weight, remaining)


SUPPORTED_NODE_FEATURES.add(NodeComputeFractionFeature())


class NodeEmbeddableAfterFeature(NodeFeature):
    """Whether or not a node is the sink"""

    def __init__(self):
        super().__init__(1, "embeddable_after")

    def _compute(self, embedding: PartialEmbedding, enode: ENode):
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


SUPPORTED_NODE_FEATURES = frozenset(SUPPORTED_NODE_FEATURES)
