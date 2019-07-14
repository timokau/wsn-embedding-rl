"""Simple semi-greedy baseline agent"""
from math import inf
import time
import numpy as np
import generator
from embedding import PartialEmbedding


def act(emb: PartialEmbedding, randomness, rand):
    """Take a semi-greedy action"""
    min_ts_actions = None
    possible_actions = emb.possibilities()

    min_ts = inf
    for (u, v, t) in possible_actions:
        if t < min_ts:
            min_ts = t
            min_ts_actions = []
        if t == min_ts:
            min_ts_actions.append((u, v, t))

    preferred_actions = min_ts_actions

    not_relay_actions = []
    for (u, v, t) in preferred_actions:
        if not v.relay:
            not_relay_actions.append((u, v, t))

    if len(not_relay_actions) > 0:
        preferred_actions = not_relay_actions

    # break out of reset loops by acting random every once in a while
    if rand.rand() < randomness:
        preferred_actions = possible_actions
    choice_idx = rand.choice(range(len(preferred_actions)))
    return preferred_actions[choice_idx]


def play_episode(embedding, max_restarts, rand):
    """Play an entire episode and report the reward"""
    restarts = 0
    while max_restarts is None or restarts < max_restarts:
        if len(embedding.possibilities()) == 0:
            if embedding.is_complete():
                return embedding.used_timeslots
            restarts += 1
            embedding = embedding.reset()
            continue

        # gradually increase randomness up to 100%
        randomness = restarts / (max_restarts - 1)
        action = act(embedding, randomness=randomness, rand=rand)
        embedding.take_action(*action)
    return None


def evaluate(episodes=100):
    """Evaluate over many episodes"""
    results = []
    times = []
    rand = np.random.RandomState(42)
    for _ in range(episodes):
        before = time.time()
        emb = generator.DefaultGenerator().random_embedding(rand)
        timeslots = play_episode(emb, max_restarts=10, rand=rand)
        if timeslots is not None:
            results.append(timeslots)
            print(results[-1])
        times.append(time.time() - before)
    return results, times


def main():
    """Run the experiment"""
    results, times = evaluate(100)
    print("=====")
    print(f"Mean result: {np.mean(results)}")
    print(f"Mean time: {np.mean(times)}")


if __name__ == "__main__":
    main()
