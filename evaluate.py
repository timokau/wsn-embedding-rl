"""Evaluate agent against marevlo results"""

import os
import re
import csv
import time
from collections import defaultdict

import numpy as np
from scipy import stats
import dill

import gym_environment
import marvelo_adapter


def load_config_from_file(name):
    """Loads an experiment config from file"""
    with open(name, "rb") as config_file:
        return dill.load(config_file)


def load_agent_from_file(name):
    """Loads a pickled RL agent from file"""
    from baselines.deepq import load_act

    # needed to get the unpickling to work since the pickling is done
    # from a __name__=="__main__"
    # pylint: disable=unused-import
    from q_network import EdgeQNetwork

    act = load_act(name)
    return act


def play_episode(act, env):
    """Play an entire episode and report the reward"""
    obs = env.reset()
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


def compare_marvelo_with_agent(act, features, marvelo_dir="marvelo_data"):
    """Runs a comparison of the MARVELO results against our agent"""
    marvelo_results = marvelo_adapter.load_from_dir(marvelo_dir)
    results = []
    for (embedding, marvelo_result, info) in marvelo_results:
        if embedding is None:
            continue
        (nodes, blocks, seed) = info
        env = gym_environment.WSNEnvironment(
            # pylint: disable=cell-var-from-loop
            problem_generator=lambda: (embedding, None),
            features=features,
            early_exit_factor=np.infty,
            # rewards don't really matter
            additional_timeslot_reward=-1,
            restart_reward=0,
            success_reward=0,
            seedgen=None,
        )
        (_agent_reward, agent_ts, elapsed) = play_episode(act, env)
        results.append(
            (nodes, blocks, seed, marvelo_result, agent_ts, elapsed)
        )
    return results


def gap(baseline, heuristic):
    """Computes the heuristic gap"""
    return 100 * (heuristic - baseline) / baseline


def process_marvelo_results(results):
    """Analyzes results of marvelo comparison"""
    by_block = defaultdict(list)
    for (nodes, blocks, _seed, marvelo_result, agent_ts, elapsed) in results:
        agent_gap = gap(marvelo_result, agent_ts)
        by_block[blocks].append((nodes, agent_gap, elapsed))
    return by_block


def marvelo_results_to_csvs(results, dirname):
    """Writes marvelo results to tables as expected by pgfplots"""
    # pylint: disable=too-many-locals
    by_block = process_marvelo_results(results)

    all_gaps = []
    all_times = []
    for (block, block_results) in by_block.items():
        filename = f"{dirname}/marvelo_b{block}.csv"
        agent_gaps = defaultdict(list)
        times = defaultdict(list)
        for (nodes, agent_gap, elapsed) in block_results:
            agent_gaps[nodes].append(agent_gap)
            all_gaps.append(agent_gap)
            all_times.append(1000 * elapsed)
            times[nodes].append(1000 * elapsed)

        with open(filename, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(("x", "y", "error-", "error+"))
            for (nodes, gap_vals) in agent_gaps.items():
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

    try:
        os.mkdir(target_dir)
    except FileExistsError:
        pass

    config_location = os.path.join(os.path.dirname(model_file), "config.pkl")
    features = load_config_from_file(config_location)
    act = load_agent_from_file(model_file)
    results = compare_marvelo_with_agent(act, features)
    marvelo_results_to_csvs(results, target_dir)


if __name__ == "__main__":
    main()
