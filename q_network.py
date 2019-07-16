# EncodeProcessDecode model is based on the graph_nets demo
# https://github.com/deepmind/graph_nets/blob/6f33ee4244ebe016b4d6296dd3eb99625fd9f3af/graph_nets/demos/models.py
"""Graph Q-Network"""

from functools import partial

from graph_nets import modules
from graph_nets import utils_tf
import sonnet as snt
import tensorflow as tf
from tf_util import ragged_boolean_mask

# The abstract sonnet _build function has a (*args, **kwargs) argument
# list, so we can pass whatever we want.
# pylint: disable=arguments-differ


def make_mlp_model(latent_size, num_layers):
    """Multilayer Perceptron followed by layer norm, parameters not
    shared"""
    return snt.Sequential(
        [
            # relu activation
            snt.nets.MLP(
                output_sizes=[latent_size] * num_layers, activate_final=True
            ),
            # normalize to mean 0, sd 1
            snt.LayerNorm(),
        ]
    )


class LinearGraphIndependent(snt.AbstractModule):
    """GraphIndependent with linear edge, node, and global models"""

    def __init__(
        self,
        edge_output_size=0,
        node_output_size=0,
        global_output_size=0,
        name="LinearGraphIndependent",
    ):
        super(LinearGraphIndependent, self).__init__(name=name)
        edge_fn = lambda: snt.Linear(edge_output_size, name="edge_output")
        node_fn = lambda: snt.Linear(node_output_size, name="node_output")
        global_fn = lambda: snt.Linear(
            global_output_size, name="global_output"
        )
        with self._enter_variable_scope():
            self._network = modules.GraphIndependent(
                edge_model_fn=edge_fn,
                node_model_fn=node_fn,
                global_model_fn=global_fn,
            )

    def _build(self, inputs):
        return self._network(inputs)


class MLPGraphIndependent(snt.AbstractModule):
    """GraphIndependent with MLP edge, node, and global models"""

    def __init__(
        self,
        # for simplicity, all layers have the same size and the edge,
        # node and global models use the same structure
        latent_size,
        num_layers,
        name="MLPGraphIndependent",
    ):
        super(MLPGraphIndependent, self).__init__(name=name)
        model_fn = partial(
            make_mlp_model, latent_size=latent_size, num_layers=num_layers
        )
        with self._enter_variable_scope():
            self._network = modules.GraphIndependent(
                edge_model_fn=model_fn,
                node_model_fn=model_fn,
                global_model_fn=model_fn,
            )

    def _build(self, inputs):
        return self._network(inputs)


class MLPGraphNetwork(snt.AbstractModule):
    """GraphNetwork with MLP edge, node, and global models"""

    def __init__(
        self,
        # for simplicity, all layers have the same size and the edge,
        # node and global models use the same structure
        latent_size,
        num_layers,
        name="MLPGraphNetwork",
    ):
        super(MLPGraphNetwork, self).__init__(name=name)
        model_fn = partial(
            make_mlp_model, latent_size=latent_size, num_layers=num_layers
        )
        with self._enter_variable_scope():
            self._network = modules.GraphNetwork(
                edge_model_fn=model_fn,
                node_model_fn=model_fn,
                global_model_fn=model_fn,
            )

    def _build(self, inputs):
        return self._network(inputs)


class EncodeProcessDecode(snt.AbstractModule):
    """Full encode-process-decode model

    The model we explore includes three components:
    - An "Encoder" graph net, which independently encodes the edge,
      node, and global attributes (does not compute relations etc.).
    - A "Core" graph net, which performs N rounds of processing
      (message-passing) steps. The input to the Core is the
      concatenation of the Encoder's output and the previous output of
      the Core (labeled "Hidden(t)" below, where "t" is the processing
      step).
    - A "Decoder" graph net, which independently decodes the edge, node,
      and global attributes (does not compute relations etc.), on each
      message-passing step.

                        Hidden(t)   Hidden(t+1)
                           |            ^
              *---------*  |  *------*  |  *---------*
              |         |  |  |      |  |  |         |
    Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
              |         |---->|      |     |         |
              *---------*     *------*     *---------*
    """

    def __init__(
        self,
        edge_output_size,
        node_output_size,
        global_output_size,
        # for simplicity, all layers have the same size and all MLPs use
        # the same structure
        latent_size,
        num_layers,
        name="EncodeProcessDecode",
    ):
        super(EncodeProcessDecode, self).__init__(name=name)
        self._encoder = MLPGraphIndependent(latent_size, num_layers)
        self._core = MLPGraphNetwork(latent_size, num_layers)
        self._decoder = MLPGraphIndependent(latent_size, num_layers)
        self._output_transform = LinearGraphIndependent(
            edge_output_size=edge_output_size,
            node_output_size=node_output_size,
            global_output_size=global_output_size,
        )

    def _build(self, input_op, num_processing_steps):
        latent = self._encoder(input_op)  # hidden(t)
        latent0 = latent
        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input)
            decoded_op = self._decoder(latent)
            output_ops.append(self._output_transform(decoded_op))
        return output_ops


class EdgeQNetwork(snt.AbstractModule):
    """Takes an input_graph, returns q-values.

    graph_nets based model that takes an input graph and returns a
    (variable length) vector of q-values corresponding to the edges in
    the input graph that represent valid actions (according to the
    boolean edge attribute in first position)"""

    def __init__(
        self,
        latent_size,
        num_layers,
        num_processing_steps,
        edge_filter_idx,
        ignore_first_edge_features,
        name="edge_q_network",
    ):
        self._latent_size = latent_size
        self._num_layers = num_layers
        self._num_processing_steps = num_processing_steps
        self._edge_filter_idx = edge_filter_idx
        self._ignore_first_edge_features = ignore_first_edge_features
        super(EdgeQNetwork, self).__init__(name=name)

    def _build(self, graph_tuple):
        model = EncodeProcessDecode(
            edge_output_size=1,  # edge output is the Q-value
            global_output_size=0,
            node_output_size=0,
            latent_size=self._latent_size,
            num_layers=self._num_layers,
        )
        # edges is 2d tensor of all edges in all graphs
        # ignore some columns for learning, for example possible bit and
        # edge id
        learn_graph_tuple = graph_tuple.map(
            lambda edges: tf.slice(
                edges, [0, self._ignore_first_edge_features], [-1, -1]
            ),
            fields=["edges"],
        )
        out = model(learn_graph_tuple, self._num_processing_steps)[-1]

        q_vals = tf.cast(tf.reshape(out.edges, [-1]), tf.float32)
        ragged_q_vals = tf.RaggedTensor.from_row_lengths(
            q_vals, tf.cast(out.n_edge, tf.int64)
        )

        def edge_is_possible_action(edge):
            possible = edge[self._edge_filter_idx]
            return tf.math.equal(possible, 1)

        viable_actions_mask = tf.map_fn(
            edge_is_possible_action, graph_tuple.edges, dtype=tf.bool
        )
        ragged_mask = tf.RaggedTensor.from_row_lengths(
            viable_actions_mask, tf.cast(graph_tuple.n_edge, tf.int64)
        )

        result = ragged_boolean_mask(ragged_q_vals, ragged_mask)

        return result.to_tensor(default_value=tf.float32.min)
