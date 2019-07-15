"""Train a graph_nets DQN agent on the WSN environment"""

import subprocess
import datetime

# needs this fork of baselines:
# https://github.com/timokau/baselines/tree/graph_nets-deepq
# see https://github.com/openai/baselines/pull/931
from baselines import logger
from baselines.deepq import learn
from networkx.drawing.nx_pydot import write_dot

from q_network import EdgeQNetwork
import gym_environment
from generator import Generator, ParallelGenerator
from draw_embedding import succinct_representation


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
    # pylint: disable=too-many-arguments
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
    node_feat_whitelist,
    node_feat_blacklist,
    edge_feat_whitelist,
    edge_feat_blacklist,
    generator_args,
):
    """Trains the agent with the given hyperparameters"""
    assert frozenset(node_feat_blacklist).issubset(node_feat_whitelist)
    assert frozenset(edge_feat_blacklist).issubset(edge_feat_whitelist)

    node_feat = frozenset(node_feat_whitelist).difference(node_feat_blacklist)
    edge_feat = frozenset(edge_feat_whitelist).difference(edge_feat_blacklist)

    parallel_gen = ParallelGenerator(Generator(**generator_args), seedgen)
    env = gym_environment.WSNEnvironment(
        node_features=node_feat,
        edge_features=edge_feat,
        early_exit_factor=early_exit_factor,
        seedgen=seedgen,
        problem_generator=parallel_gen.new_instance,
    )

    git_label = _git_describe()
    time_label = datetime.datetime.now().isoformat()
    logger.configure(
        dir=f"logs/{time_label}-{git_label}-{experiment_name}",
        format_strs=["stdout", "csv", "tensorboard"],
    )

    # needs to be lambda since the scope at constructor time is used
    # pylint: disable=unnecessary-lambda
    q_model = lambda inp: EdgeQNetwork(
        edge_filter_idx=gym_environment.POSSIBLE_IDX,
        num_processing_steps=num_processing_steps,
        latent_size=latent_size,
        num_layers=num_layers,
    )(inp)

    learn(
        env,
        q_model,
        make_obs_ph=lambda name: env.observation_space.to_placeholders(),
        as_is=True,
        dueling=False,
        prioritized=prioritized,
        print_freq=1,
        train_freq=train_freq,
        batch_size=batch_size,
        exploration_fraction=exploration_fraction,
        checkpoint_freq=1000,
        seed=rl_seed,
        total_timesteps=learnsteps * train_freq,
        checkpoint_path=logger.get_dir(),
        after_step_callback=save_episode_result_callback,
    )


if __name__ == "__main__":
    from hyperparameters import DEFAULT

    run_training(**DEFAULT)
