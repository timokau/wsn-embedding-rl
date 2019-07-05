"""Profiling and testing (through random actions) of the model
implementation"""
import time
import numpy as np
from pycallgraph import PyCallGraph, Config, GlobbingFilter
from pycallgraph.output import GraphvizOutput
from generator import random_embedding, get_random_action


def main(time_seconds=60):
    """Keep randomly exploring embeddings"""
    total_before = time.time()
    while True:
        action_list = []
        before = time.time()
        embedding = random_embedding(np.random)
        while True:
            try:
                complete = embedding.is_complete()
                action = get_random_action(embedding)
                if (complete and action is not None) or (
                    not complete and action is None
                ):
                    raise Exception("Problematic embedding")

                if action is None:
                    break
                action_list.append(action)
                embedding.take_action(*action)
                elapsed_ms = round((time.time() - before) * 1000, 2)
            except Exception as e:  # pylint:disable=broad-except
                print(e)
                print(embedding)
                for action in action_list[:50]:
                    print(action)
                return
        actions = len(action_list)
        per_action = round(elapsed_ms / actions, 2)
        print(f"{elapsed_ms}ms ({actions}, {per_action}ms)")
        if time.time() - total_before > time_seconds:
            return


def profile(size, time_seconds=60):
    """Profile the embedding"""
    config = Config()
    config.trace_filter = GlobbingFilter(exclude=["pycallgraph.*"])
    graphviz = GraphvizOutput(output_file=f"pc-{size}.png")
    with PyCallGraph(output=graphviz, config=config):
        main(time_seconds)


if __name__ == "__main__":
    main(99999999)
    # profile(20, 600)
