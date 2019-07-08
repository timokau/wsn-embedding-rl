"""Simple semi-greedy baseline agent"""
from math import inf
import random
import time
import numpy as np
import gym_environment
import generator


def act(graph_tuple, randomness=0):
    """Take a semi-greedy action"""
    min_ts_actions = None
    possible_actions = []
    min_ts = inf
    i = 0
    for (u, v, d) in zip(
        graph_tuple.senders, graph_tuple.receivers, graph_tuple.edges
    ):
        possible = d[gym_environment.POSSIBLE_IDX] == 1
        if not possible:
            continue
        else:
            timeslot = int(d[gym_environment.TIMESLOT_IDX])
            possible_actions.append((u, v, d))
            if timeslot == min_ts:
                min_ts_actions.append(i)
            elif timeslot < min_ts:
                min_ts = timeslot
                min_ts_actions = [i]
            i += 1

    # break out of reset loops by acting random every once in a while
    if random.random() < randomness:
        return random.choice(range(i))

    preferred_actions = min_ts_actions

    not_relay_actions = []
    for action_idx in preferred_actions:
        (u, v, d) = possible_actions[action_idx]
        receiver = graph_tuple.nodes[v]
        receiver_is_relay = bool(receiver[gym_environment.RELAY_IDX])
        if not receiver_is_relay:
            not_relay_actions.append(action_idx)

    if len(not_relay_actions) > 0:
        preferred_actions = not_relay_actions

    choice = random.choice(preferred_actions)
    return choice


def play_episode(embedding, max_restarts=None):
    """Play an entire episode and report the reward"""
    env = gym_environment.WSNEnvironment()
    obs = env.reset(embedding)
    total_reward = 0
    if len(embedding.possibilities()) == 0:
        return (None, None)
    while max_restarts is None or env.restarts < max_restarts:
        # gradually increase randomness up to 100%
        randomness = env.restarts / (max_restarts - 1)
        action = act(obs, randomness=randomness)
        new_obs, rew, done, _ = env.step(action)
        total_reward += rew
        obs = new_obs
        if done:
            return (total_reward, env.env.used_timeslots)
    return (None, None)


def evaluate(episodes=100):
    """Evaluate over many episodes"""
    rewards = []
    times = []
    for _ in range(episodes):
        before = time.time()
        emb = generator.random_embedding(rand=np.random)
        (reward, _timeslots) = play_episode(emb, 100)
        if reward is not None:
            rewards.append(reward)
            print(rewards[-1])
        times.append(time.time() - before)
    return rewards, times


def main():
    """Run the experiment"""
    rewards, times = evaluate(100)
    print("=====")
    print(f"Mean reward: {np.mean(rewards)}")
    print(f"Mean time: {np.mean(times)}")


if __name__ == "__main__":
    main()
