"""Generation of new problem instances"""

import time

import numpy as np
import networkx as nx

from overlay import OverlayNetwork
from infrastructure import InfrastructureNetwork
from embedding import PartialEmbedding
import baseline_agent


def random_infrastructure(num_sources: int):
    """Generates a randomize infrastructure with default parameters

    The resulting nodes will be distributed uniformly at random in a
    25m x 25m room.
    """
    rand = np.random

    num_intermediates = round(rand.exponential(10))
    pos_dist = lambda: rand.uniform(low=(0, 0), high=(25, 25))
    capacity_dist = lambda: rand.exponential(10)

    mean_transmit_power_dbm = 30  # FCC limit for a wifi router is 36dBm
    power_dist = lambda: rand.normal(mean_transmit_power_dbm, 10)

    return _random_infrastructure(
        num_intermediates, num_sources, pos_dist, capacity_dist, power_dist
    )


def random_overlay(num_sources: int):
    """Generates a randomized overlay graph with default parameters"""
    rand = np.random

    num_intermediates = round(rand.exponential(6))
    pairwise_connection = lambda: rand.random() < 0.01
    compute_requirement_dist = lambda: rand.exponential(5)
    # datarate in bits/s with an assumed bandwidth of 1 (i.e. equivalent
    # to SINRth)
    datarate_dist = lambda: rand.exponential(5)

    return _random_overlay(
        num_intermediates,
        num_sources,
        pairwise_connection,
        compute_requirement_dist,
        datarate_dist,
        connection_choice=rand.choice,
    )


def random_embedding():
    """Generate matching random infrastructure + overlay + embedding"""
    # at least one source, has to match between infra and overlay
    num_sources = round(np.random.exponential(2)) + 1

    while True:
        infra = random_infrastructure(num_sources)
        overlay = random_overlay(num_sources)
        source_mapping = list(zip(list(overlay.sources), list(infra.sources)))

        # make sure all sources and the sink are actually embeddable
        valid = True  # be optimistic
        for (block, node) in source_mapping + [(overlay.sink, infra.sink)]:
            if overlay.requirement(block) > infra.capacity(node):
                valid = False
        if valid:
            return PartialEmbedding(infra, overlay, source_mapping)


def validated_random():
    """Returns a random embedding that is guaranteed to be solvable
    together with a baseline solution"""
    t = 0
    before = time.time()
    while True:
        t += 1
        emb = random_embedding()
        (_, baseline) = baseline_agent.play_episode(emb, max_restarts=10)
        if baseline is not None:
            elapsed = time.time() - before
            print(f"Generated (try {t}, {round(elapsed ,1)}s)")
            return (emb.reset(), baseline)


def _random_infrastructure(
    num_intermediates: int,
    num_sources: int,
    pos_dist,  # () -> (float, float)
    capacity_dist,  # () -> float
    power_dist,  # () -> float (dBm)
):
    """
    Generates a randomized infrastructure

    Total number of nodes will be num_sources + num_intermediates + 1.
    """
    assert num_sources > 0

    infra = InfrastructureNetwork()

    # always one sink
    infra.set_sink(
        pos=pos_dist(),
        transmit_power_dbm=power_dist(),
        capacity=capacity_dist(),
    )

    # generate sources
    for _ in range(num_sources):
        infra.add_source(
            pos=pos_dist(),
            transmit_power_dbm=power_dist(),
            capacity=capacity_dist(),
        )

    # generate intermediates
    for _ in range(num_intermediates):
        infra.add_intermediate(
            pos=pos_dist(),
            transmit_power_dbm=power_dist(),
            capacity=capacity_dist(),
        )

    return infra


def _random_overlay(
    num_intermediates: int,
    num_sources: int,
    pairwise_connection: float,  # () -> bool
    compute_requirement_dist,  # () -> float
    datarate_dist,  # () -> float
    connection_choice,  # array -> element
):
    """Generates a randomized overlay graph

    Total number of blocks will be num_sources + num_intermediates + 1.
    """
    # This is a complicated function, but it would only get harder to
    # understand when split up into multiple single-use functions.
    # pylint: disable=too-many-branches

    assert num_sources > 0

    overlay = OverlayNetwork()

    # always one sink
    overlay.set_sink(requirement=compute_requirement_dist())

    # add sources
    for _ in range(num_sources):
        overlay.add_source(requirement=compute_requirement_dist())

    # add intermediates
    for _ in range(num_intermediates):
        overlay.add_intermediate(requirement=compute_requirement_dist())

    # randomly add links
    for source in overlay.graph.nodes():
        for target in overlay.graph.nodes():
            if target != source and pairwise_connection():
                overlay.add_link(source, target, datarate=datarate_dist())

    # add links necessary to have each block on a path from a source to
    # the sink
    accessible_from_source = set()
    not_accessible_from_source = set()
    has_path_to_sink = set()
    no_path_to_sink = set()
    for node in overlay.graph.nodes():
        # check if the node can already reach the sink
        if nx.has_path(overlay.graph, node, overlay.sink):
            has_path_to_sink.add(node)
        else:
            no_path_to_sink.add(node)

        # check if the node is already reachable from the source
        source_path_found = False
        for source in overlay.sources:
            if nx.has_path(overlay.graph, source, node):
                source_path_found = True
                break
        if source_path_found:
            accessible_from_source.add(node)
        else:
            not_accessible_from_source.add(node)

    # make sure all nodes are reachable from a source
    for node in not_accessible_from_source:
        connection = connection_choice(list(accessible_from_source))
        overlay.add_link(connection, node, datarate=datarate_dist())
        accessible_from_source.add(node)

    # make sure all nodes can reach the sink
    for node in no_path_to_sink:
        connection = connection_choice(list(has_path_to_sink))
        overlay.add_link(node, connection, datarate=datarate_dist())
        has_path_to_sink.add(node)

    return overlay


def get_random_action(embedding: PartialEmbedding, rand=np.random):
    """Take a random action on the given partial embedding"""
    possibilities = embedding.possibilities()
    if len(possibilities) == 0:
        return None
    choice = rand.randint(0, len(possibilities))
    action = possibilities[choice]
    return action


if __name__ == "__main__":
    print(validated_random())
