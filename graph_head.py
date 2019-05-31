"""Currently doesn't do anything, should implement the actual graph_nets
network"""

import tensorflow as tf
from rl_coach.architectures.head_parameters import HeadParameters
from rl_coach.architectures.tensorflow_components.heads.head import Head
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import QActionStateValue
from rl_coach.spaces import SpacesDefinition


class GraphEdgesQHeadParameters(HeadParameters):
    """Parameters for a GraphEdges Q Head"""

    def __init__(
        self,
        activation_function: str = "relu",
        name: str = "graph_edges_q_head_params",
    ):
        super().__init__(
            parameterized_class_name="GraphEdgesQHead",
            activation_function=activation_function,
            name=name,
        )

    @property
    def path(self):
        return "graph_head:GraphEdgesQHead"


class GraphEdgesQHead(Head):
    # out of my control
    # pylint: disable=too-many-instance-attributes
    """Head that takes a graph as an input and produces an ouput with
    one Q-Value per edge"""

    def __init__(
        # out of my control
        # pylint: disable=too-many-arguments
        self,
        agent_parameters: AgentParameters,
        spaces: SpacesDefinition,
        network_name: str,
        head_idx: int = 0,
        loss_weight: float = 1.0,
        is_local: bool = True,
        activation_function: str = "relu",
    ):
        self.actions = None
        super().__init__(
            agent_parameters,
            spaces,
            network_name,
            head_idx,
            loss_weight,
            is_local,
            activation_function,
        )
        self.name = "graph_edges_q_head"
        self.num_actions = len(self.spaces.action.actions)
        self.num_atoms = agent_parameters.algorithm.atoms
        self.return_type = QActionStateValue

    def _build_module(self, input_layer):
        self.actions = tf.placeholder(tf.int32, [None], name="actions")
        self.input = [self.actions]
        self.output = tf.constant(0)
        self.target = tf.constant(0)
        self.loss = tf.constant(0)
        tf.losses.add_loss(self.loss)
