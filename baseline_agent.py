"""Simple semi-greedy baseline agent"""
from math import inf
import random
import time
import numpy as np
from gym_environment import WSNEnvironment


def act(graph_tuple):
    """Take a semi-greedy action"""
    min_ts_actions = None
    possible_actions = []
    min_ts = inf
    i = 0
    for (u, v, d) in zip(
        graph_tuple.senders, graph_tuple.receivers, graph_tuple.edges
    ):
        possible = d[1] == 1
        if not possible:
            continue
        else:
            timeslot = int(d[2])
            possible_actions.append((u, v, d))
            if timeslot == min_ts:
                min_ts_actions.append(i)
            elif timeslot < min_ts:
                min_ts = timeslot
                min_ts_actions = [i]
            i += 1

    # break out of reset loops by acting random every once in a while
    # if random.random() < 0.01:
    #     return random.choice(range(i))

    preferred_actions = min_ts_actions

    not_relay_actions = []
    for action_idx in preferred_actions:
        (u, v, d) = possible_actions[action_idx]
        receiver = graph_tuple.nodes[v]
        # receiver_pos = (receiver[0], receiver[1])
        receiver_is_relay = bool(receiver[2])
        if not receiver_is_relay:
            not_relay_actions.append(action_idx)

    if len(not_relay_actions) > 0:
        preferred_actions = not_relay_actions

    return random.choice(preferred_actions)


def play_episode(env):
    """Play an entire episode and report the reward"""
    obs = env.reset()
    total_reward = 0
    while True:
        action = act(obs)
        new_obs, rew, done, _ = env.step(action)
        total_reward += rew
        obs = new_obs
        if done:
            print(total_reward)
            return total_reward


def evaluate(env, episodes=100):
    """Evaluate over many episodes"""
    rewards = []
    times = []
    for _ in range(episodes):
        before = time.time()
        rewards.append(play_episode(env))
        times.append(time.time() - before)
    return rewards, times


def main():
    """Run the experiment"""
    env = WSNEnvironment()
    rewards, times = evaluate(env, 100)
    print("=====")
    print(f"Mean reward: {np.mean(rewards)}")
    print(f"Mean time: {np.mean(times)}")


if __name__ == "__main__":
    main()
