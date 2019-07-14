"""Generation of new problem instances"""

import time
import math

# fork of multiprocessing that uses dill for pickling (usage of lambdas)
from queue import Queue

import multiprocess as multiprocessing
import numpy as np
from scipy import stats
import networkx as nx

from overlay import OverlayNetwork
from infrastructure import InfrastructureNetwork
from embedding import PartialEmbedding
import baseline_agent


def truncnorm(rand, mean=0, sd=1, low=-np.infty, upp=np.infty):
    """Convenience wrapper around scipys truncnorm"""
    dist = stats.truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd
    )
    # for some reason this can't be set in the constructor
    dist.random_state = rand
    return float(dist.rvs())


class Generator:
    """Generates random problem instances from a given distribution"""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        # pylint: disable=too-many-arguments
        self,
        interm_nodes_dist=lambda r: round(truncnorm(r, mean=5, sd=3, low=0)),
        pos_dist=lambda r: r.uniform(low=(0, 0), high=(25, 25)),
        capacity_dist=lambda r: truncnorm(r, mean=10, sd=5, low=0),
        power_dist=lambda r: r.normal(30, 2),
        interm_blocks_dist=lambda r: round(truncnorm(r, mean=3, sd=2, low=0)),
        pairwise_connection=lambda r: r.rand() < 0.01,
        block_weight_dist=lambda r: truncnorm(r, mean=5, low=0, sd=2),
        # mean equivalent to a linear SINRth of 20, which is what marvelo uses
        requirement_dist=lambda r: truncnorm(
            r, mean=math.log(1 + 20, 2), low=0, sd=1
        ),
        num_sources_dist=lambda r: round(truncnorm(r, mean=2, sd=1, low=1)),
        connection_choice=lambda r, a: r.choice(a),
    ):
        self.interm_nodes_dist = interm_nodes_dist
        self.pos_dist = pos_dist
        self.capacity_dist = capacity_dist
        self.power_dist = power_dist
        self.interm_blocks_dist = interm_blocks_dist
        self.pairwise_connection = pairwise_connection
        self.block_weight_dist = block_weight_dist
        self.requirement_dist = requirement_dist
        self.num_sources_dist = num_sources_dist
        self.connection_choice = connection_choice

    def random_embedding(self, rand):
        """Generate matching random infrastructure + overlay + embedding"""
        # at least one source, has to match between infra and overlay
        num_sources = self.num_sources_dist(rand)

        while True:
            infra = self.random_infrastructure(num_sources, rand)
            overlay = self.random_overlay(num_sources, rand)
            source_mapping = list(
                zip(list(overlay.sources), list(infra.sources))
            )

            # make sure all sources and the sink are actually embeddable
            valid = True  # be optimistic
            for (block, node) in source_mapping + [(overlay.sink, infra.sink)]:
                if overlay.requirement(block) > infra.capacity(node):
                    valid = False
            if valid:
                return PartialEmbedding(infra, overlay, source_mapping)

    def validated_random(self, rand):
        """Returns a random embedding that is guaranteed to be solvable
        together with a baseline solution"""
        while True:
            before = time.time()
            emb = self.random_embedding(rand)
            (reward, baseline) = baseline_agent.play_episode(
                emb, max_restarts=10
            )
            elapsed = round(time.time() - before, 1)
            nodes = len(emb.infra.nodes())
            blocks = len(emb.overlay.blocks())
            links = len(emb.overlay.links())
            if baseline is not None:
                if elapsed > 60:
                    # pylint: disable=line-too-long
                    print(
                        f"Generated ({elapsed}s, {nodes} nodes, {blocks} blocks, {links} links, {reward})"
                    )
                return (emb.reset(), baseline)
            if elapsed > 60:
                # pylint: disable=line-too-long
                print(
                    f"Failed    ({elapsed}s, {nodes} nodes, {blocks} blocks, {links} links, {reward})"
                )

    def random_infrastructure(self, num_sources: int, rand):
        """Generates a randomized infrastructure"""
        assert num_sources > 0

        infra = InfrastructureNetwork()

        rand_node_args = lambda: {
            "pos": self.pos_dist(rand),
            "transmit_power_dbm": self.power_dist(rand),
            "capacity": self.capacity_dist(rand),
        }

        infra.set_sink(**rand_node_args())

        for _ in range(num_sources):
            infra.add_source(**rand_node_args())

        for _ in range(self.interm_nodes_dist(rand)):
            infra.add_intermediate(**rand_node_args())

        return infra

    def random_overlay(self, num_sources: int, rand):
        """Generates a randomized overlay graph"""
        # This is a complicated function, but it would only get harder to
        # understand when split up into multiple single-use functions.
        # pylint: disable=too-many-branches

        assert num_sources > 0

        overlay = OverlayNetwork()

        rand_block_args = lambda: {
            "requirement": self.block_weight_dist(rand),
            "datarate": self.requirement_dist(rand),
        }

        # always one sink
        overlay.set_sink(**rand_block_args())

        # add sources
        for _ in range(num_sources):
            overlay.add_source(**rand_block_args())

        # add intermediates
        for _ in range(self.interm_blocks_dist(rand)):
            overlay.add_intermediate(**rand_block_args())

        # randomly add links
        for source in sorted(overlay.graph.nodes()):
            for target in sorted(overlay.graph.nodes()):
                if target != source and self.pairwise_connection(rand):
                    overlay.add_link(source, target)

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
        for node in sorted(not_accessible_from_source):
            connection = self.connection_choice(
                rand, sorted(accessible_from_source)
            )
            overlay.add_link(connection, node)
            accessible_from_source.add(node)

        # make sure all nodes can reach the sink
        for node in sorted(no_path_to_sink):
            connection = self.connection_choice(rand, sorted(has_path_to_sink))
            overlay.add_link(node, connection)
            has_path_to_sink.add(node)

        return overlay


QUEUE_SIZE = 16


class ParallelGenerator:
    """Generator that uses multiprocessing to amortize generation"""

    def __init__(self, generator, seedgen):
        # reserver one cpu for the actual training
        cpus = max(1, multiprocessing.cpu_count() - 1)
        cpus = min(8, cpus)
        self._pool = multiprocessing.Pool(cpus)
        self._instance_queue = Queue(QUEUE_SIZE)
        self.generator = generator
        self.seedgen = seedgen

    def new_instance(self):
        """Transparently uses multiprocessing"""
        # similar to a lazy infinite imap; preserves the order of the
        # generated elements to prevent under-representation of long
        # running ones.
        while not self._instance_queue.full():
            rand = np.random.RandomState(self.seedgen())
            job = self._pool.map_async(self.generator.validated_random, [rand])
            self._instance_queue.put_nowait(job)

        next_job = self._instance_queue.get()
        if not next_job.ready():
            print("Blocked on queue")
        return next_job.get()[0]

    def __getstate__(self):
        state = self.__dict__.copy()
        # don't pickle pool or queue (this means that the generator will
        # become useless after pickling, but that is fine since it is
        # only pickled when an agent is saved and not used afterwards)
        del state["_pool"]
        del state["_instance_queue"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


def get_random_action(embedding: PartialEmbedding, rand):
    """Take a random action on the given partial embedding"""
    possibilities = embedding.possibilities()
    if len(possibilities) == 0:
        return None
    choice = rand.randint(0, len(possibilities))
    return possibilities[choice]


if __name__ == "__main__":
    print(Generator().validated_random(rand=np.random))
