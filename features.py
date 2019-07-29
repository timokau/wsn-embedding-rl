"""Various candidates for node and edge features"""

from embedding import PartialEmbedding, ENode


def frac(a, b):
    """Regular fraction, but result is 0 if a = b = 0"""
    if a == 0 and b == 0:
        return 0
    return a / b


def _postprocess_feature(feature):
    """Make a feature compatible with graph tuples"""
    # always return an iterable
    if not hasattr(feature, "__len__"):
        feature = (feature,)

    # transparently convert bools etc. to float
    return [float(value) for value in feature]


class Feature:
    """A feature extractor"""

    def __init__(self, name, edge_fun, node_fun, edge_dim, node_dim):
        self.name = name
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.edge_fun = edge_fun
        self.node_fun = node_fun

    def process_edge(
        self,
        embedding: PartialEmbedding,
        source: ENode,
        target: ENode,
        timeslot: int,
    ):
        """Extracts a feature from an edge"""
        feature = (
            self.edge_fun(embedding, source, target, timeslot)
            if self.edge_fun is not None
            else []
        )
        feature = _postprocess_feature(feature)
        assert len(feature) == self.edge_dim
        return feature

    def process_node(self, embedding: PartialEmbedding, enode: ENode):
        """Extracts a feature from a node"""
        feature = (
            self.node_fun(embedding, enode)
            if self.node_fun is not None
            else []
        )
        feature = _postprocess_feature(feature)
        assert len(feature) == self.node_dim
        return feature


class NodeFeature(Feature):
    """A node feature extractor"""

    def __init__(self, name, compute_fun, dim=1):
        super().__init__(
            "node_" + name,
            edge_dim=0,
            edge_fun=None,
            node_dim=dim,
            node_fun=compute_fun,
        )


class EdgeFeature(Feature):
    """An edge feature extractor"""

    def __init__(self, name, compute_fun, dim=1):
        super().__init__(
            "edge_" + name,
            edge_dim=dim,
            edge_fun=compute_fun,
            node_dim=0,
            node_fun=None,
        )


def _is_broadcast(embedding, source, target, timeslot):
    for (other_so, other_ta) in embedding.taken_edges_in[timeslot]:
        if other_so == source and other_ta == target:
            continue
        if other_so.node == other_ta.node:
            continue  # self loop, nothing is sent
        if other_so.block == source.block and other_so.node == source.node:
            return True
    return False


def _remaining_capacity_before_chosen(emb, enode):
    remaining = emb.remaining_capacity(enode.node)
    if emb.graph.nodes[enode]["chosen"]:
        remaining += emb.overlay.requirement(enode.block)
    return remaining


def _options_lost(embedding: PartialEmbedding, enode: ENode):
    weight = embedding.overlay.requirement(enode.block)
    remaining = _remaining_capacity_before_chosen(embedding, enode)

    not_yet_embedded = set(embedding.overlay.blocks()).difference(
        embedding.taken_embeddings.keys()
    )

    # assuming the node was chosen
    not_yet_embedded = not_yet_embedded.difference({enode.block})

    options_before = {
        block
        for block in not_yet_embedded
        if embedding.overlay.requirement(block) < remaining
    }

    remaining_after = remaining - weight
    options_after = {
        block
        for block in not_yet_embedded
        if embedding.overlay.requirement(block) < remaining_after
    }
    return len(options_before) - len(options_after)


def _capacity(emb, u, v, t):
    if u.node == v.node:
        # If this is a loop, it has an effective capacity of infty. Its
        # hard to learn with values of infinity though, so we just
        # pretend the capacity perfectly matches the requirement.
        return emb.overlay.requirement(u.acting_as)
    return emb.known_capacity(u.node, v.node, t)


SUPPORTED_FEATURES = [
    NodeFeature(
        "pos",
        lambda emb, enode: emb.infra.graph.node[enode.node]["pos"],
        dim=2,
    ),
    NodeFeature("relay", lambda emb, enode: enode.relay),
    NodeFeature(
        "sink",
        lambda emb, enode: enode.node == emb.infra.sink
        and enode.block == emb.overlay.sink,
    ),
    NodeFeature("remaining_capacity", _remaining_capacity_before_chosen),
    NodeFeature(
        "weight", lambda emb, enode: emb.overlay.requirement(enode.block)
    ),
    NodeFeature(
        "compute_fraction",
        lambda emb, enode: frac(
            emb.overlay.requirement(enode.block),
            _remaining_capacity_before_chosen(emb, enode),
        ),
    ),
    NodeFeature("options_lost", _options_lost),
    EdgeFeature("timeslot", lambda emb, u, v, t: t),
    EdgeFeature(
        "chosen", lambda emb, u, v, t: emb.graph.edges[u, v, t]["chosen"]
    ),
    EdgeFeature("capacity", _capacity),
    EdgeFeature(
        "additional_timeslot",
        lambda emb, u, v, t: u.node != v.node and t >= emb.used_timeslots,
    ),
    EdgeFeature(
        "datarate_requirement",
        lambda emb, u, v, t: emb.overlay.datarate(u.acting_as),
    ),
    EdgeFeature(
        "datarate_fraction",
        lambda emb, u, v, t: frac(
            emb.overlay.datarate(u.acting_as),
            emb.known_capacity(u.node, v.node, t),
        ),
    ),
    EdgeFeature("is_broadcast", _is_broadcast),
]


def features_by_name():
    """Results a dict of all supported features"""
    result = dict()
    for feature in SUPPORTED_FEATURES:
        result[feature.name] = feature
    return result
