"""Empirically determine PEG size as a function of input size"""

import csv
import numpy as np
from scipy import stats

import baseline_agent
from generator import Generator
from hyperparameters import GENERATOR_DEFAULTS


def _play_episode(emb):
    emb = emb.reset()
    enodes = [len(emb.graph.nodes())]
    edges = [len(emb.graph.edges(keys=True))]
    choices = [len(emb.possibilities())]
    while len(emb.possibilities()) > 0:
        action = baseline_agent.act(emb, randomness=0, rand=np.random)
        emb.take_action(*action)
        enodes.append(len(emb.graph.nodes()))
        edges.append(len(emb.graph.edges(keys=True)))
        choices.append(len(emb.possibilities()))
    return (enodes, edges, choices)


def _main(dirname):
    args = GENERATOR_DEFAULTS.copy()
    args["num_sources_dist"] = lambda r: 1
    rng = np.random.RandomState(1)
    for blocks in [2, 3, 4]:
        filename = f"{dirname}/peg_edges_b{blocks}.csv"
        with open(filename, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(("nodes", "edges", "sem"))
            for nodes in range(2, 56 + 1, 4):
                # This is intentional.
                # pylint: disable=cell-var-from-loop
                args["interm_nodes_dist"] = lambda r: nodes - 2
                args["interm_blocks_dist"] = lambda r: blocks - 2
                gen = Generator(**args)
                all_edges = []
                for _experiment in range(100):
                    embedding = gen.random_embedding(rng)
                    # n = len(embedding.infra.nodes())
                    # b = len(embedding.overlay.blocks())
                    # l = len(embedding.overlay.links())
                    # edge_bound = n * (n - 1) * l + 2 * n * l * n + l * n * n
                    # enode_bound = n * b + n * l
                    (_enodes, edges, _choices) = _play_episode(embedding)
                    all_edges.extend(edges)
                print(f"n{nodes}b{blocks}:", round(np.average(all_edges)))
                writer.writerow(
                    (nodes, np.average(all_edges), stats.sem(all_edges))
                )


if __name__ == "__main__":
    import sys

    _main(sys.argv[1])
