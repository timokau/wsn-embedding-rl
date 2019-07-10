"""Evaluate agent against marevlo results"""

import os
import numpy as np
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
    while True:
        if np.random.rand() < 0.5:
            action = np.random.choice(len(env.actions))
        else:
            act_result = act(obs)
            action = act_result[0][0]
        new_obs, rew, done, _ = env.step(action)
        total_reward += rew
        obs = new_obs
        if done:
            return (total_reward, env.env.used_timeslots)


def compare_marvelo_with_agent(agent_file, marvelo_dir="marvelo_data"):
    """Runs a comparison of the MARVELO results against our agent"""
    act = load_agent_from_file(agent_file)
    marvelo_results = marvelo_adapter.load_from_dir(marvelo_dir)
    deltas = []
    for (embedding, marvelo_result, info) in marvelo_results:
        if embedding is None:
            continue
        print(info)
        (agent_reward, agent_ts) = play_episode(act, embedding)
        print(
            f"MARVELO: {marvelo_result}, Agent: {agent_ts}, ({agent_reward})"
        )
        deltas.append(agent_ts - marvelo_result)
    print(f"Average delta: {np.mean(deltas)}")


def find_latest_model_in_pwd(filename="model.pkl"):
    """Convenience function to locate a suitable model"""
    options = []
    for root, _dirs, files in os.walk("."):
        for name in files:
            if name == filename:
                path = os.path.join(root, name)
                mtime = os.path.getmtime(path)
                options.append((path, mtime))
    options.sort(key=lambda a: a[1])
    return options[-1][0]


if __name__ == "__main__":
    compare_marvelo_with_agent(find_latest_model_in_pwd())
