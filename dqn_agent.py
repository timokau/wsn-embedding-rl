"""Train a graph_nets DQN agent on the WSN environment"""

import subprocess
import datetime
from functools import partial

# needs this fork of baselines:
# https://github.com/timokau/baselines/tree/graph_nets-deepq
# see https://github.com/openai/baselines/pull/931
from baselines import logger
from baselines.deepq import learn
from networkx.drawing.nx_pydot import write_dot
import dill
import numpy as np

from q_network import EdgeQNetwork
import gym_environment
from observation import TIMESLOT_IDX, POSSIBLE_IDX
from generator import Generator, ParallelGenerator
from draw_embedding import succinct_representation
import evaluate


def save_episode_result_callback(lcl, _glb):
    """Saves the result of a "solved" episode as a dot file"""
    if not lcl["done"]:
        return
    episode = len(lcl["episode_rewards"])
    total_reward = round(lcl["env"].env.used_timeslots)
    write_dot(
        succinct_representation(lcl["env"].env),
        f"{logger.get_dir()}/result-{episode}-{-total_reward}.dot",
    )


def _eval_hook(act, log, features):
    results = evaluate.compare_marvelo_with_agent(act, features)
    results = evaluate.process_marvelo_results(results)
    all_gaps = []
    all_times = []
    for blocks in results.keys():
        gaps = [gap for (nodes, gap, _, elapsed) in results[blocks]]
        times = [elapsed for (nodes, gap, _, elapsed) in results[blocks]]
        log.record_tabular(f"marvelo b{blocks} gap", np.mean(gaps))
        all_gaps.extend(gaps)
        all_times.extend(times)
    log.record_tabular(f"marvelo total gap", np.mean(all_gaps))
    log.record_tabular(f"marvelo avg time", np.mean(times))
    log.dump_tabular()


def _git_describe():
    try:
        return (
            subprocess.check_output(["git", "describe", "--always"])
            .strip()
            .decode()
        )
    except subprocess.CalledProcessError:
        return "nogit"


def run_training(
    # pylint: disable=too-many-arguments, too-many-locals
    learnsteps,
    train_freq,
    batch_size,
    exploration_fraction,
    early_exit_factor,
    num_processing_steps,
    latent_size,
    num_layers,
    seedgen,
    rl_seed,
    experiment_name,
    prioritized,
    prioritized_replay_alpha,
    prioritized_replay_beta0,
    prioritized_replay_beta_iters,
    prioritized_replay_eps,
    learning_starts,
    buffer_size,
    lr,
    gamma,
    grad_norm_clipping,
    target_network_update_freq,
    features,
    generator_args,
    restart_reward,
    success_reward,
    additional_timeslot_reward,
):
    """Trains the agent with the given hyperparameters"""
    parallel_gen = ParallelGenerator(Generator(**generator_args), seedgen)
    env = gym_environment.WSNEnvironment(
        features=features,
        early_exit_factor=early_exit_factor,
        seedgen=seedgen,
        problem_generator=parallel_gen.new_instance,
        restart_reward=restart_reward,
        success_reward=success_reward,
        additional_timeslot_reward=additional_timeslot_reward,
    )

    git_label = _git_describe()
    time_label = datetime.datetime.now().isoformat()
    logdir = f"logs/{time_label}-{git_label}-{experiment_name}"
    logger.configure(dir=logdir, format_strs=["stdout", "csv", "tensorboard"])

    with open(f"{logdir}/config.pkl", "wb") as config_file:
        dill.dump(features, config_file, protocol=4)

    # needs to be lambda since the scope at constructor time is used
    # pylint: disable=unnecessary-lambda
    q_model = lambda inp: EdgeQNetwork(
        edge_filter_idx=POSSIBLE_IDX,
        num_processing_steps=num_processing_steps,
        latent_size=latent_size,
        num_layers=num_layers,
        # ignore medatadata features during learning
        ignore_first_edge_features=2,
    )(inp)
    assert TIMESLOT_IDX < 2 and POSSIBLE_IDX < 2

    learn(
        env,
        q_model,
        make_obs_ph=lambda name: env.observation_space.to_placeholders(),
        as_is=True,
        dueling=False,
        prioritized=prioritized,
        prioritized_replay_alpha=prioritized_replay_alpha,
        prioritized_replay_beta0=prioritized_replay_beta0,
        prioritized_replay_beta_iters=prioritized_replay_beta_iters,
        prioritized_replay_eps=prioritized_replay_eps,
        print_freq=1,
        train_freq=train_freq,
        batch_size=batch_size,
        exploration_fraction=exploration_fraction,
        learning_starts=learning_starts,
        buffer_size=buffer_size,
        lr=lr,
        gamma=gamma,
        grad_norm_clipping=grad_norm_clipping,
        target_network_update_freq=target_network_update_freq,
        checkpoint_freq=1000,
        eval_freq=1000,
        eval_hook=partial(_eval_hook, features=features),
        seed=rl_seed,
        total_timesteps=learnsteps * train_freq,
        checkpoint_path=logger.get_dir(),
        after_step_callback=save_episode_result_callback,
    )


if __name__ == "__main__":
    from hyperparameters import DEFAULT

    run_training(**DEFAULT)
