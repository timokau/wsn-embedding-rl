"""Profiling and testing (through random actions) of the model
implementation"""
import time
import numpy as np
from pycallgraph import PyCallGraph, Config, GlobbingFilter
from pycallgraph.output import GraphvizOutput
from generator import random_embedding, get_random_action


def main(time_seconds=60, rand=np.random):
    """Keep randomly exploring embeddings"""
    total_before = time.time()
    while True:
        action_list = []
        before = time.time()
        embedding = random_embedding(rand)
        while True:
            if time.time() - total_before > time_seconds:
                return
            action = get_random_action(embedding)
            if action is None:
                break
            action_list.append(action)
            embedding.take_action(*action)
            elapsed_ms = round((time.time() - before) * 1000, 2)
        actions = len(action_list)
        per_action = round(elapsed_ms / actions, 2)
        print(f"{elapsed_ms}ms ({actions}, {per_action}ms)")


def profile(time_seconds=60, rand=np.random):
    """Profile the embedding"""
    config = Config()
    config.trace_filter = GlobbingFilter(exclude=["pycallgraph.*"])
    graphviz = GraphvizOutput(output_file=f"pc.png")
    with PyCallGraph(output=graphviz, config=config):
        main(time_seconds, rand)


if __name__ == "__main__":
    profile(60)
