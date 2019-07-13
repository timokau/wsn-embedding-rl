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


def main():
    """Run the training"""
    # reproducibility
    state = np.random.RandomState(42)
    seedgen = lambda: state.randint(0, 2 ** 32)
    env = gym_environment.WSNEnvironment(seedgen=seedgen)
    try:
        git_label = (
            subprocess.check_output(["git", "describe", "--always"])
            .strip()
            .decode()
        )
    except subprocess.CalledProcessError:
        git_label = "nogit"
    time_label = datetime.datetime.now().isoformat()

    logger.configure(
        dir=f"logs/{time_label}-{git_label}",
        format_strs=["stdout", "csv", "tensorboard"],
    )
    learnsteps = 100000
    train_freq = 1
    learn(
        env,
        deepq_graph_network,
        make_obs_ph=lambda name: env.observation_space.to_placeholders(),
        as_is=True,
        dueling=False,
        prioritized=True,
        print_freq=1,
        train_freq=train_freq,
        batch_size=32,
        checkpoint_freq=1000,
        total_timesteps=learnsteps * train_freq,
        checkpoint_path=logger.get_dir(),
        after_step_callback=save_episode_result_callback,
    )


if __name__ == "__main__":
    main()
