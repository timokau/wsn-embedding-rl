"""Evaluate agent against marevlo results"""

import os
import re
import csv
import time
from collections import defaultdict
import numpy as np
from scipy import stats
import gym_environment
import marvelo_adapter


def load_agent_from_file(name):
    """Loads a pickled RL agent from file"""
    from baselines.deepq import load_act

    # needed to get the unpickling to work since the pickling is done
    # from a __name__=="__main__"
    # pylint: disable=unused-import
    from dqn_agent import deepq_graph_network

    act = load_act(name)
    return act


def play_episode(act, embedding):
    """Play an entire episode and report the reward"""
    env = gym_environment.WSNEnvironment()
    obs = env.reset(embedding)
    total_reward = 0
    before = time.time()
    while True:
        act_result = act(obs)
        action = act_result[0][0]
        new_obs, rew, done, _ = env.step(action)
        total_reward += rew
        obs = new_obs
        if done:
            elapsed = time.time() - before
            return (total_reward, env.env.used_timeslots, elapsed)


def compare_marvelo_with_agent(agent_file, marvelo_dir="marvelo_data"):
    """Runs a comparison of the MARVELO results against our agent"""
    act = load_agent_from_file(agent_file)
    marvelo_results = marvelo_adapter.load_from_dir(marvelo_dir)
    results = []
    for (embedding, marvelo_result, info) in marvelo_results:
        if embedding is None:
            continue
        (nodes, blocks, seed) = info
        (agent_reward, agent_ts, elapsed) = play_episode(act, embedding)
        print(
            f"MARVELO: {marvelo_result}, Agent: {agent_ts}, ({agent_reward})"
        )
        results.append(
            (nodes, blocks, seed, marvelo_result, agent_ts, elapsed)
        )
    return results


def marvelo_results_to_csvs(results, dirname):
    """Writes marvelo results to tables as expected by pgfplots"""
    # pylint: disable=too-many-locals
    by_block = defaultdict(list)
    for (nodes, blocks, seed, marvelo_result, agent_ts, elapsed) in results:
        by_block[blocks].append(
            (nodes, seed, marvelo_result, agent_ts, elapsed)
        )

    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass

    all_gaps = []
    all_times = []
    for (block, block_results) in by_block.items():
        filename = f"{dirname}/marvelo_b{block}.csv"
        gaps = defaultdict(list)
        times = defaultdict(list)
        for (nodes, seed, marvelo_result, agent_ts, elapsed) in block_results:
            gap = 100 * (agent_ts - marvelo_result) / marvelo_result
            gaps[nodes].append(gap)
            all_gaps.append(gap)
            all_times.append(1000 * elapsed)
            times[nodes].append(1000 * elapsed)

        with open(filename, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(("x", "y", "error-", "error+"))
            for (nodes, gap_vals) in gaps.items():
                mean = np.mean(gap_vals)
                err_low = stats.sem(gap_vals)
                err_high = stats.sem(gap_vals)
                writer.writerow((f"{nodes} nodes", mean, err_low, err_high))

        filename = f"{dirname}/times_b{block}.csv"
        with open(filename, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(("x", "y", "error-", "error+"))
            for (nodes, time_vals) in times.items():
                mean = np.mean(time_vals)
                err_low = stats.sem(time_vals)
                err_high = stats.sem(time_vals)
                writer.writerow((nodes, mean, err_low, err_high))

    mean = round(np.mean(all_gaps), 2)
    sem = round(stats.sem(all_gaps), 2)
    times_mean = round(np.mean(all_times), 2)
    print(f"Overall: mean {mean}, sem {sem}, elapsed {times_mean}")


def find_latest_model_in_pwd(regex=r"model.*\.pkl"):
    """Convenience function to locate a suitable model"""
    options = []
    name_regex = re.compile(regex)
    for root, _dirs, files in os.walk("."):
        for name in files:
            if name_regex.match(name) is not None:
                path = os.path.join(root, name)
                mtime = os.path.getmtime(path)
                options.append((path, mtime))
    options.sort(key=lambda a: a[1])
    return options[-1][0]


def main():
    """Runs the evaluation and formats the results"""
    import sys

    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = "results"
    if len(sys.argv) > 2:
        model_file = sys.argv[2]
    else:
        model_file = find_latest_model_in_pwd()
    print(f"Evaluating {model_file}, saving results to {target_dir}")

    marvelo_results_to_csvs(compare_marvelo_with_agent(model_file), target_dir)


if __name__ == "__main__":
    main()
