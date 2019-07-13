"""Train a graph_nets DQN agent on the WSN environment"""

import subprocess
import datetime
import tensorflow as tf
import numpy as np

# needs this fork of baselines:
# https://github.com/timokau/baselines/tree/graph_nets-deepq
# see https://github.com/openai/baselines/pull/931
from baselines import logger
from baselines.deepq import learn
from graph_nets.demos.models import EncodeProcessDecode
from networkx.drawing.nx_pydot import write_dot

import gym_environment
from draw_embedding import succinct_representation
from tf_util import ragged_boolean_mask

NUM_PROCESSING_STEPS = 5


def deepq_graph_network(inpt):
    """Takes an input_graph, returns q-values.

    graph_nets based model that takes an input graph and returns a
    (variable length) vector of q-values corresponding to the edges in
    the input graph that represent valid actions (according to the
    boolean edge attribute in second position)"""
    out = EncodeProcessDecode(
        edge_output_size=1, global_output_size=0, node_output_size=0
    )(inpt, NUM_PROCESSING_STEPS)
    out = out[-1]

    q_vals = tf.cast(tf.reshape(out.edges, [-1]), tf.float32)
    ragged_q_vals = tf.RaggedTensor.from_row_lengths(
        q_vals, tf.cast(out.n_edge, tf.int64)
    )

    def edge_is_possible_action(edge):
        possible = edge[gym_environment.POSSIBLE_IDX]
        return tf.math.equal(possible, 1)

    viable_actions_mask = tf.map_fn(
        edge_is_possible_action, inpt.edges, dtype=tf.bool
    )
    ragged_mask = tf.RaggedTensor.from_row_lengths(
        viable_actions_mask, tf.cast(inpt.n_edge, tf.int64)
    )

    result = ragged_boolean_mask(ragged_q_vals, ragged_mask)

    return result.to_tensor(default_value=tf.float32.min)


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
    learnsteps=100000,
    train_freq=1,
    batch_size=32,
    early_exit_factor=np.infty,
    seedgen=None,  # defaults to reproducible
    experiment_name="default",
    prioritized=True,
    node_feat_whitelist=gym_environment.SUPPORTED_NODE_FEATURES,
    node_feat_blacklist=frozenset(),
    edge_feat_whitelist=gym_environment.SUPPORTED_EDGE_FEATURES,
    edge_feat_blacklist=frozenset(),
):
    """Trains the agent with the given hyperparameters"""
    if seedgen is None:
        # reproducibility
        state = np.random.RandomState(42)
        seedgen = lambda: state.randint(0, 2 ** 32)

    assert frozenset(node_feat_blacklist).issubset(node_feat_whitelist)
    assert frozenset(edge_feat_blacklist).issubset(edge_feat_whitelist)

    node_feat = frozenset(node_feat_whitelist).difference(node_feat_blacklist)
    edge_feat = frozenset(edge_feat_whitelist).difference(edge_feat_blacklist)

    env = gym_environment.WSNEnvironment(
        node_features=node_feat,
        edge_features=edge_feat,
        early_exit_factor=early_exit_factor,
        seedgen=seedgen,
    )

    git_label = _git_describe()
    time_label = datetime.datetime.now().isoformat()
    logger.configure(
        dir=f"logs/{time_label}-{git_label}-{experiment_name}",
        format_strs=["stdout", "csv", "tensorboard"],
    )

    learn(
        env,
        deepq_graph_network,
        make_obs_ph=lambda name: env.observation_space.to_placeholders(),
        as_is=True,
        dueling=False,
        prioritized=prioritized,
        print_freq=1,
        train_freq=train_freq,
        batch_size=batch_size,
        checkpoint_freq=1000,
        total_timesteps=learnsteps * train_freq,
        checkpoint_path=logger.get_dir(),
        after_step_callback=save_episode_result_callback,
    )


if __name__ == "__main__":
    run_training()
