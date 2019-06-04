"""Generation of new problem instances"""

from math import floor

import numpy as np

from infrastructure import random_infrastructure
from overlay import random_overlay
from embedding import PartialEmbedding


def random_embedding(max_embedding_nodes=32, rand=np.random):
    """Generate a random embedding that is guaranteed to be solvable"""
    solvable = False
    while not solvable:
        result = _random_embedding(max_embedding_nodes, rand)
        solvable = result.is_solvable()
    return result


def _random_embedding(max_embedding_nodes=32, rand=np.random):
    """Generate matching random infrastructure + overlay + embedding"""
    infra = random_infrastructure(
        rand, min_nodes=3, max_nodes=max_embedding_nodes / 3, num_sources=2
    )

    num_nodes = len(infra.graph.nodes())
    max_blocks = min([num_nodes, floor(max_embedding_nodes / num_nodes)])
    overlay = random_overlay(
        rand,
        min_blocks=3,
        max_blocks=max_blocks,
        num_sources=len(infra.sources),
    )
    source_mapping = dict()
    source_blocks = list(overlay.sources)
    source_nodes = list(infra.sources)
    rand.shuffle(source_nodes)
    source_mapping = list(zip(source_blocks, source_nodes))
    embedding = PartialEmbedding(infra, overlay, source_mapping)
    return embedding


def get_random_action(embedding: PartialEmbedding, rand=np.random):
    """Take a random action on the given partial embedding"""
    possibilities = embedding.possibilities()
    if len(possibilities) == 0:
        return None
    choice = rand.randint(0, len(possibilities))
    action = possibilities[choice]
    return action
