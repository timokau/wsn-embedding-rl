"""Profiling and testing (through random actions) of the model
implementation"""
import time
import numpy as np
from pycallgraph import PyCallGraph, Config, GlobbingFilter
from pycallgraph.output import GraphvizOutput
from generator import DefaultGenerator, get_random_action


def main(rand, pcg):
    """Keep randomly exploring embeddings"""
    while True:
        action_list = []
        before = time.time()
        embedding = DefaultGenerator().random_embedding(rand)
        while True:
            action = get_random_action(embedding, rand=rand)
            if action is None:
                break
            action_list.append(action)
            pcg.start(reset=False)
            embedding.take_action(*action)
            pcg.stop()
            elapsed_ms = round((time.time() - before) * 1000, 2)
            if elapsed_ms > 10000:
                pcg.done()
            if action is None:
                break
        actions = len(action_list)
        per_action = round(elapsed_ms / actions, 2)
        print(f"{elapsed_ms}ms ({actions}, {per_action}ms)")
        # import sys
        # sys.exit(1)


def profile(rand=np.random):
    """Profile the embedding"""
    config = Config()
    config.trace_filter = GlobbingFilter(exclude=["pycallgraph.*"])
    graphviz = GraphvizOutput(output_file=f"pc.png")
    pcg = PyCallGraph(output=graphviz, config=config)
    main(rand, pcg)


if __name__ == "__main__":
    profile(rand=np.random.RandomState(42))
