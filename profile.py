"""Profiling and testing (through random actions) of the model
implementation"""
import time
from pycallgraph import PyCallGraph, Config, GlobbingFilter
from pycallgraph.output import GraphvizOutput
from generator import random_embedding, get_random_action


def main(size=20):
    """Keep randomly exploring embeddings"""
    total_before = time.time()
    while True:
        action_list = []
        before = time.time()
        embedding = random_embedding(size)
        while True:
            complete = embedding.is_complete()
            action = get_random_action(embedding)
            if (complete and action is not None) or (
                not complete and action is None
            ):
                print(embedding)
                for action in action_list[:100]:
                    print(action)
                raise Exception("Problematic embedding")

            if action is None:
                break
            action_list.append(action)
            embedding.take_action(*action)
            elapsed_ms = round((time.time() - before) * 1000, 2)
        actions = len(action_list)
        per_action = round(elapsed_ms / actions, 2)
        print(f"{elapsed_ms}ms ({actions}, {per_action}ms)")
        if time.time() - total_before > 3600:
            return

def profile():
    """Profile the embedding"""
    config = Config()
    config.trace_filter = GlobbingFilter(exclude=["pycallgraph.*"])
    graphviz = GraphvizOutput(output_file="pc-50.png")
    with PyCallGraph(output=graphviz, config=config):
        main(50)

if __name__ == "__main__":
    profile()
