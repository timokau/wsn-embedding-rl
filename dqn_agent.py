"""Train a graph_nets DQN agent on the WSN environment"""

import itertools
import numpy as np
import tensorflow as tf

import baselines.common.tf_util as baselines_tf_util

# needs this fork of baselines:
# https://github.com/timokau/baselines/tree/graph_nets-deepq
# see https://github.com/openai/baselines/pull/931
from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule

from graph_nets.demos.models import EncodeProcessDecode
from networkx.drawing.nx_pydot import write_dot

from gym_environment import WSNEnvironment
from tf_util import ragged_boolean_mask

# === hyperparameters ===
NUM_PROCESSING_STEPS = 5  # number of recurrent steps in the graph network
REPLAY_BUFFER_SIZE = 50000
BATCH_SIZE = 32
EPSILON_DECAY_TIMESTAMPS = 10000
TARGET_EPSILON = 0.02
UPDATE_TARGET_EVERY = 1000
LEARNING_RATE = 5e-4


def deepq_graph_network(inpt, scope, reuse=False):
    """Takes an input_graph, returns q-values.

    graph_nets based model that takes an input graph and returns a
    (variable length) vector of q-values corresponding to the edges in
    the input graph that represent valid actions (according to the
    boolean edge attribute in second position)"""
    with tf.variable_scope(scope, reuse=reuse):
        out = EncodeProcessDecode(
            edge_output_size=1, global_output_size=0, node_output_size=0
        )(inpt, NUM_PROCESSING_STEPS)
        out = out[-1]

        q_vals = tf.cast(tf.reshape(out.edges, [-1]), tf.float32)
        ragged_q_vals = tf.RaggedTensor.from_row_lengths(
            q_vals, tf.cast(out.n_edge, tf.int64)
        )

        def edge_is_possible_action(edge):
            possible = edge[1]  # second attribute, float repr of bool
            return tf.math.equal(possible, 1)

        viable_actions_mask = tf.map_fn(
            edge_is_possible_action, inpt.edges, dtype=tf.bool
        )
        ragged_mask = tf.RaggedTensor.from_row_lengths(
            viable_actions_mask, tf.cast(inpt.n_edge, tf.int64)
        )

        result = ragged_boolean_mask(ragged_q_vals, ragged_mask)

        return result.to_tensor(default_value=tf.float32.min)


def main():
    """Run the training"""
    # pylint: disable=too-many-locals
    with baselines_tf_util.make_session():
        env = WSNEnvironment()

        act, train, update_target, _debug = deepq.build_train(
            make_obs_ph=lambda name: env.observation_space.to_placeholders(),
            q_func=deepq_graph_network,
            optimizer=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE),
        )

        replay_buf = ReplayBuffer(REPLAY_BUFFER_SIZE)
        exploration = LinearSchedule(
            schedule_timesteps=EPSILON_DECAY_TIMESTAMPS,
            initial_p=1.0,
            final_p=TARGET_EPSILON,
        )

        baselines_tf_util.initialize()
        update_target()

        episode_rewards = [0.0]
        episode_times = []
        # training time per episode
        training_times = [0.0]
        obs = env.reset()
        import time

        episode_start = time.time()
        for t in itertools.count():
            # Take action and update exploration to the newest value
            action = act(obs, update_eps=exploration.value(t))[0]
            new_obs, rew, done, _ = env.step(action)
            # Store transition in the replay buffer.
            replay_buf.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                episode = len(episode_rewards)
                total_reward = episode_rewards[-1]
                write_dot(
                    env.env.succinct_representation(),
                    f"{logger.get_dir()}/result-{episode}-{-total_reward}.dot",
                )
                episode_rewards.append(0)
                training_times.append(0)
                episode_times.append(time.time() - episode_start)
                episode_start = time.time()
                obs = env.reset()

            # Minimize the error in Bellman's equation on a batch
            # sampled from replay buffer.
            if t > UPDATE_TARGET_EVERY:
                before = time.time()
                obses_t, actions, rews, obses_tp1, dones = replay_buf.sample(
                    BATCH_SIZE
                )
                train(
                    obses_t,
                    actions,
                    rews,
                    obses_tp1,
                    dones,
                    np.ones_like(rews),
                )
                training_times[-1] += time.time() - before

            if t % UPDATE_TARGET_EVERY == 0:
                update_target()

            if done and len(episode_rewards) % 10 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular(
                    "mean episode reward",
                    round(np.mean(episode_rewards[-101:-1]), 1),
                )
                logger.record_tabular(
                    "mean episode time",
                    round(np.mean(episode_times[-101:-1]), 1),
                )
                logger.record_tabular(
                    "mean training time",
                    round(np.mean(training_times[-101:-1]), 1),
                )
                logger.record_tabular(
                    "% time spent exploring", int(100 * exploration.value(t))
                )
                logger.dump_tabular()


if __name__ == "__main__":
    main()
