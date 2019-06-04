"""
Trains a graph net on the simple toy problem of predicting the number of
timeslots in use on a given embedding, e.g.
max(timeslot_attr on edges) - 1

This is very simple and useless on purpose. Its just a sanity check to
make sure I have the wiring of graph_nets right.
"""

import time
import itertools
import pickle
import shutil
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import networkx as nx

from graph_nets import utils_tf, utils_np
from graph_nets.demos import models
from joblib import Parallel, delayed

from generator import random_embedding, get_random_action

# cache generated samples to speed up experimentation (although that of
# course can introduce overfitting when doing too much experimentation)
PICKLE_NAME = "samples.pickle"


def generate_partial_embedding(rand):
    """
    Generates a partial embedding to train and evaluate the neural
    network. Embeddings are random and steps are taken at random.
    """
    embedding = random_embedding(80, rand=rand)
    actions = 0
    while rand.rand() < 0.90:
        actions += 1
        action = get_random_action(embedding)
        if action is None:
            break
        embedding.take_action(*action)
    return embedding


def embedding_to_input_target(embedding):
    """Produces an input with the necessary features and a targeted
    output"""
    # build graphs from stretch, since we need to change the node
    # indexing (graph_nets can only deal with integer indexed nodes)
    input_graph = nx.MultiDiGraph()
    target_graph = nx.MultiDiGraph()
    infra_graph = embedding.infra.graph
    node_to_index = dict()

    # add the nodes
    for (i, enode) in enumerate(embedding.graph.nodes()):
        inode = infra_graph.node[enode.node]
        node_to_index[enode] = i
        # the position isn't really important, but let the network
        # figure that out
        input_graph.add_node(
            i, features=np.array([inode["pos"][0], inode["pos"][1]])
        )
        target_graph.add_node(i, features=np.array([]))

    # add the edges
    for (u, v, k, d) in embedding.graph.edges(data=True, keys=True):
        u = node_to_index[u]
        v = node_to_index[v]
        input_graph.add_edge(
            u,
            v,
            k,
            # easier when including the timeslot, but a bit more
            # interesting with just the chosen information
            features=np.array([float(d["chosen"])]),
        )
        target_graph.add_edge(u, v, k, features=np.array([0.0]))

    # no globals in input
    input_graph.graph["features"] = np.array([0.0])
    # predict the amount of used timeslots, effectively the max of the
    # chosen edges (or the total max -1)
    target_graph.graph["features"] = np.array(
        [float(embedding.used_timeslots)]
    )

    return input_graph, target_graph


def _generate_training_sample(rand):
    embedding = generate_partial_embedding(rand)
    return embedding_to_input_target(embedding)


PICKLE_ITER = None
CONSUMED_EMBEDDINGS = 0


def generate_training_samples(rand, num_samples: int):
    """Generates training samples on demand, using the cache when
    possible (and refilling it when not)"""
    # I'm sure there are plenty of better ways to do this, but this is
    # just a quick-and-dirty experiment.
    # pylint: disable=global-statement
    global PICKLE_ITER, CONSUMED_EMBEDDINGS
    if PICKLE_ITER is None:
        Path(PICKLE_NAME).touch(exist_ok=True)
        PICKLE_ITER = stream_pickle(open(PICKLE_NAME, "rb"))
        for _ in range(CONSUMED_EMBEDDINGS):
            next(PICKLE_ITER)
    inputs = []
    targets = []
    embeddings = list(itertools.islice(PICKLE_ITER, num_samples))
    if len(embeddings) < num_samples:
        refill_pickle(rand, 2000)
        PICKLE_ITER = None
        # restart
        return generate_training_samples(rand, num_samples)

    for embedding in embeddings:
        (i, t) = embedding_to_input_target(embedding)
        inputs.append(i)
        targets.append(t)
    CONSUMED_EMBEDDINGS += num_samples
    return inputs, targets


def refill_pickle(rand, num_samples):
    """Generates new training examples while also writing them to the
    pickle file for later reuse"""
    # write to temporary file to make sure this is atomic
    shutil.copyfile(PICKLE_NAME, f"{PICKLE_NAME}.tmp")

    with open(f"{PICKLE_NAME}.tmp", "ab") as f:
        before = time.time()
        embeddings = _generate_partial_embeddings(rand, num_samples)
        elapsed = time.time() - before
        print(f"Generated {num_samples} samples in {round(elapsed)}s")
        for embedding in embeddings:
            pickle.dump(embedding, f, pickle.HIGHEST_PROTOCOL)
        del embeddings
        os.rename("samples.pickle.tmp", "samples.pickle")


def stream_pickle(f):
    """Read objects from a pickle file in a streaming fashion"""
    try:
        while True:
            yield pickle.load(f)
    except EOFError:
        return


def _generate_partial_embeddings(rand, num_embeddings: int):
    """Generate partial embeddings in parallel"""
    return Parallel(n_jobs=4)(
        delayed(generate_partial_embedding)(rand)
        for _ in range(num_embeddings)
    )


def create_placeholders(rand, batch_size: int):
    """Creates placeholder graph_tuples for tensorflow"""
    inputs, targets = generate_training_samples(rand, batch_size)
    input_placeholder = utils_tf.placeholders_from_networkxs(inputs)
    target_placeholder = utils_tf.placeholders_from_networkxs(targets)

    return input_placeholder, target_placeholder


def generate_training_feed_dict(
    rand, batch_size: int, input_placeholder, target_placeholder
):
    """Turns inputs into a tensorflow feed dict"""
    inputs, targets = generate_training_samples(rand, batch_size)
    inputs = utils_np.networkxs_to_graphs_tuple(inputs)
    targets = utils_np.networkxs_to_graphs_tuple(targets)
    feed_dict = {input_placeholder: inputs, target_placeholder: targets}
    return feed_dict


def make_all_runnable_in_session(*args):
    """Lets an iterable of TF graphs be output from a session as NP graphs."""
    return [utils_tf.make_runnable_in_session(a) for a in args]


def create_loss_ops(target_op, output_ops):
    """Computes the loss of a prediction using MSE"""
    loss_ops = [
        tf.losses.mean_squared_error(target_op.globals, output_op.globals)
        for output_op in output_ops
    ]
    return loss_ops


def main():
    """Train and test the network"""
    # pylint: disable=too-many-locals
    rand = np.random

    num_processing_steps_training = 5
    num_processing_steps_testing = 5

    num_training_iterations = 10000
    batch_size_training = 32
    batch_size_generalization = 100
    learning_rate = 1e-3

    log_every_seconds = 60

    input_placeholder, target_placeholder = create_placeholders(
        rand, batch_size_training
    )

    # encode input graph (by applying separate MLPs to edges, nodes and
    # globals). Then do multiple processing passes (with message
    # passing) and decode it again.
    model = models.EncodeProcessDecode(
        # don't care about edges and nodes
        # edge_output_size=None,
        # node_output_size=None,
        # number of timeslots used
        global_output_size=1
    )
    output_ops_training = model(
        input_placeholder, num_processing_steps_training
    )
    output_ops_generalization = model(
        input_placeholder, num_processing_steps_testing
    )
    loss_ops_training = create_loss_ops(
        target_placeholder, output_ops_training
    )
    loss_op_training = loss_ops_training[-1]
    loss_ops_generalization = create_loss_ops(
        target_placeholder, output_ops_generalization
    )
    loss_op_generalization = loss_ops_generalization[-1]

    optimizer = tf.train.AdamOptimizer(learning_rate)
    step_op = optimizer.minimize(loss_op_training)

    input_placeholder, target_placeholder = make_all_runnable_in_session(
        input_placeholder, target_placeholder
    )

    # do iterations, evaluate training each iteration, evaluate
    # generalization every couple of seconds, print some sample of every
    # iteration
    # adapted from the graph_nets shortest_path demo
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    last_iteration = 0
    logged_iterations = []
    losses_tr = []
    losses_ge = []

    print(
        "# (iteration number), T (elapsed seconds), "
        "Ltr (training loss), Lge (test/generalization loss)"
    )

    start_time = time.time()
    last_log_time = start_time
    for iteration in range(last_iteration, num_training_iterations):
        last_iteration = iteration
        feed_dict = generate_training_feed_dict(
            rand, batch_size_training, input_placeholder, target_placeholder
        )
        train_values = sess.run(
            {
                "step": step_op,
                "target": target_placeholder,
                "loss": loss_op_training,
                "outputs": output_ops_training,
            },
            feed_dict=feed_dict,
        )
        outputs = train_values["outputs"][-1]
        outputs = utils_np.graphs_tuple_to_data_dicts(outputs)
        targets = train_values["target"]
        targets = utils_np.graphs_tuple_to_data_dicts(targets)
        pred = outputs[-1]["globals"][0]
        target = targets[-1]["globals"][0]
        # just printing some sample results to the console, helps
        # getting an idea on what the model learns
        print(f"Last: {round(pred, 1)} ({target})")
        the_time = time.time()
        elapsed_since_last_log = the_time - last_log_time
        if elapsed_since_last_log > log_every_seconds:
            last_log_time = the_time
            feed_dict = generate_training_feed_dict(
                rand,
                batch_size_generalization,
                input_placeholder,
                target_placeholder,
            )
            test_values = sess.run(
                {
                    "target": target_placeholder,
                    "loss": loss_op_generalization,
                    "outputs": output_ops_generalization,
                },
                feed_dict=feed_dict,
            )
            elapsed = time.time() - start_time
            losses_tr.append(train_values["loss"])
            losses_ge.append(test_values["loss"])
            logged_iterations.append(iteration)
            print(
                "# {:05d}, T {:.1f}, Ltr {:.4f}, Lge {:.4f}".format(
                    iteration,
                    elapsed,
                    train_values["loss"],
                    test_values["loss"],
                )
            )


if __name__ == "__main__":
    main()
