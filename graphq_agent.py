"""A DQN agent using graph networks"""

from typing import Union
import numpy as np
import networkx as nx
import tensorflow as tf
import graph_nets as gn
from graph_nets.demos.models import EncodeProcessDecode
from graph_nets import utils_np, utils_tf

from rl_coach.agents.dqn_agent import (
    DQNNetworkParameters,
    DQNAlgorithmParameters,
)
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.agents.value_optimization_agent import (
    Agent,
    ValueOptimizationAgent,
)
from rl_coach.base_parameters import AgentParameters
from rl_coach.memories.non_episodic.experience_replay import (
    ExperienceReplayParameters,
)
from rl_coach.core_types import ActionInfo, StateType

from graph_head import GraphEdgesQHeadParameters


class GraphEdgesDQNNetworkParameters(DQNNetworkParameters):
    """Parameters for the GraphEdgesDQNNetwork"""

    def __init__(self):
        super().__init__()
        self.heads_parameters = [GraphEdgesQHeadParameters()]


class GraphEdgesDQNAlgorithmParameters(DQNAlgorithmParameters):
    """Parameters for the GraphEdgesDQNAlgorithm"""

    def __init__(self):
        super().__init__()
        self.v_min = -10.0
        self.v_max = 10.0
        self.atoms = 51


class GraphEdgesDQNAgentParameters(AgentParameters):
    """Parameters for the GraphEdgesDQNAgent"""

    def __init__(self):
        super().__init__(
            algorithm=GraphEdgesDQNAlgorithmParameters(),
            exploration=EGreedyParameters(),
            memory=ExperienceReplayParameters(),
            networks={"main": GraphEdgesDQNNetworkParameters()},
        )

    @property
    def path(self):
        return "graphq_agent:GraphEdgesDQNAgent"


class DynamicActionsValueOptimizationAgent(ValueOptimizationAgent):
    """Value optimization agent that leaves out certain statistics that
    assume that the state space is static"""

    # This is not intended for direct use
    # pylint: disable=abstract-method

    def init_environment_dependent_modules(self):
        # no per-action statistics
        Agent.init_environment_dependent_modules(self)

    def choose_action(self, curr_state):
        actions_q_values = self.get_all_q_values_for_states(curr_state)

        # choose action according to the exploration policy and the
        # current phase (evaluating or training the agent)
        action = self.exploration_policy.get_action(actions_q_values)
        self._validate_action(self.exploration_policy, action)

        if actions_q_values is not None:
            # this is for bootstrapped dqn
            if (
                isinstance(actions_q_values, list)
                and len(actions_q_values) > 0
            ):
                actions_q_values = self.exploration_policy.last_action_values

            # store the q values statistics for logging
            # self.q_values.add_sample(actions_q_values)

            # Do not squeeze, doesn't work with just one action. Not
            # sure what the purpose of this is anyway. Why not make
            # actions_q_values 1d in the first place?

            # actions_q_values = actions_q_values.squeeze()

            # for i, q_value in enumerate(actions_q_values):
            #     self.q_value_for_action[i].add_sample(q_value)

            action_info = ActionInfo(
                action=action,
                action_value=actions_q_values[action],
                max_action_value=np.max(actions_q_values),
            )
        else:
            action_info = ActionInfo(action=action)

        return action_info


def _states_to_graphs_tuple(states):
    result = []
    for idx in range(len(states["globals"])):
        if len(states["globals"]) == 1:
            idx = slice(None)
        graph = dict()
        graph["globals"] = states["globals"][idx].reshape(-1)
        graph["n_node"] = states["n_node"][idx].reshape(-1)[0]
        graph["n_edge"] = states["n_edge"][idx].reshape(-1)[0]
        graph["nodes"] = states["nodes"][idx].reshape(-1, 2)
        graph["edges"] = states["edges"][idx].reshape(-1, 3)
        graph["senders"] = states["senders"][idx].reshape(-1)
        graph["receivers"] = states["receivers"][idx].reshape(-1)
        result.append(graph)
    return utils_np.data_dicts_to_graphs_tuple(result)


def _find_edge_index(graphs_tuple, u, v, k):
    for i in range(graphs_tuple.n_edge[0]):
        if (
            graphs_tuple.senders[i] == u
            and graphs_tuple.receivers[i] == v
            and graphs_tuple.edges[i][1] == k
        ):
            return i
    return None


class GraphEdgesDQNAgent(DynamicActionsValueOptimizationAgent):
    """DQN Agent using graph edges as actions"""

    def __init__(
        self,
        agent_parameters,
        parent: Union["LevelManager", "CompositeAgent"] = None,
    ):
        super().__init__(agent_parameters=agent_parameters, parent=parent)
        self.output_ops = None
        self.loss_ops = None
        self.step_op = None
        self.sess = None
        self.input_placeholder = None
        self.target_placeholder = None

    def _get_prediction(self, graphs_tuple):
        if self.sess is None:
            self._init_session()
        graphs_tuple = graphs_tuple

        feed_dict = {self.input_placeholder: graphs_tuple}

        pred_values = self.sess.run(
            {"outputs": self.output_ops}, feed_dict=feed_dict
        )
        outputs = pred_values["outputs"][-1]

        return outputs

    def _states_to_networkx(self, states: StateType, pred=None):
        """Reads a GraphsTuple encoded in the state and converts it to a
        Networkx graph"""
        if pred is None:
            graphs_tuple = _states_to_graphs_tuple(states)
            pred = self._get_prediction(graphs_tuple)

        graph = nx.MultiDiGraph()
        for (u, v, attrs, prediction) in zip(
            states["senders"], states["receivers"], states["edges"], pred.edges
        ):
            chosen = attrs[0] == 1
            possible = attrs[1] == 1
            timeslot = int(attrs[2])
            graph.add_edge(
                u,
                v,
                timeslot,
                chosen=chosen,
                timeslot=timeslot,
                possible=possible,
                prediction=prediction[0],
            )

        return graph

    def _get_full_q_values_for_states(self, states, pred=None):
        graph = self._states_to_networkx(states, pred)
        result = []
        for (u, v, d) in graph.edges(data=True):
            timeslot = d["timeslot"]
            possible = d["possible"]
            prediction = d["prediction"]
            if possible:
                result.append((u, v, timeslot, prediction))

        # make sure ordering is the same as the environment expects
        def key(a):
            (u, v, t, _) = a
            return (u, v, t)

        result.sort(key=key)
        return result

    def get_all_q_values_for_states(self, states: StateType, pred=None):
        # Just added an optional argument for self-use.
        # pylint: disable=arguments-differ
        result = self._get_full_q_values_for_states(states, pred)
        result = np.array([q for (_, _, _, q) in result])
        return result

    def _create_placeholders(self):
        # Not sure why its protected, seems to be the most elegant way
        # to define placeholders.
        # pylint: disable=protected-access
        self.input_placeholder = utils_tf._build_placeholders_from_specs(
            dtypes=gn.graphs.GraphsTuple(
                nodes=tf.float64,
                edges=tf.float64,
                receivers=tf.int32,
                senders=tf.int32,
                globals=tf.float64,
                n_node=tf.int32,
                n_edge=tf.int32,
            ),
            shapes=gn.graphs.GraphsTuple(
                # x, y
                nodes=[None, 2],
                # chosen, timestep, possible
                edges=[None, 3],
                receivers=[None],
                senders=[None],
                globals=[None, 1],
                n_node=[None],
                n_edge=[None],
            ),
        )
        self.target_placeholder = utils_tf._build_placeholders_from_specs(
            dtypes=gn.graphs.GraphsTuple(
                nodes=tf.float64,
                edges=tf.float64,
                receivers=tf.int32,
                senders=tf.int32,
                globals=tf.float64,
                n_node=tf.int32,
                n_edge=tf.int32,
            ),
            shapes=gn.graphs.GraphsTuple(
                nodes=[None, 0],
                # predicted q-value
                edges=[None, 1],
                receivers=[None],
                senders=[None],
                globals=[None, 0],
                n_node=[None],
                n_edge=[None],
            ),
        )

    def _init_session(self):
        learning_rate = 1e-3
        num_processing_steps = 5

        tf.reset_default_graph()
        self._create_placeholders()

        gn_model = EncodeProcessDecode(
            edge_output_size=1, global_output_size=0, node_output_size=0
        )
        # decoded result of each processing step
        self.output_ops = gn_model(
            self.input_placeholder, num_processing_steps
        )

        self.loss_ops = [
            tf.losses.mean_squared_error(
                self.target_placeholder.edges, output_op.edges
            )
            for output_op in self.output_ops
        ]

        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.step_op = optimizer.minimize(self.loss_ops[-1])

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        tf.get_default_graph().finalize()

    def learn_from_batch(self, batch):
        # Will be reworked to properly use the coach architecture
        # instead of implementing everything here.
        # pylint: disable=too-many-locals
        if self.sess is None:
            self._init_session()

        cur_states = _states_to_graphs_tuple(
            batch.states(gn.graphs.ALL_FIELDS)
        )
        next_states = _states_to_graphs_tuple(
            batch.next_states(gn.graphs.ALL_FIELDS)
        )

        next_states_qs = self._get_prediction(next_states)
        cur_states_qs = self._get_prediction(cur_states)
        td_targets = []

        #  only update the action that we have actually done in this transition
        td_errors = []
        for i in range(batch.size):
            cur_state = utils_np.get_graph(cur_states, i)
            cur_state_q = utils_np.get_graph(cur_states_qs, i)
            next_state = utils_np.get_graph(next_states, i)
            next_state_q = utils_np.get_graph(next_states_qs, i)

            q_vals = self._get_full_q_values_for_states(
                cur_state._asdict(), cur_state_q
            )
            selected_action = np.argmax([q for (_, _, _, q) in q_vals])

            u, v, k, _ = q_vals[selected_action]
            idx = _find_edge_index(cur_state, u, v, k)

            next_q_vals = self.get_all_q_values_for_states(
                next_state._asdict(), next_state_q
            )
            if not batch.game_overs()[i]:
                greedy_val_of_next_state = np.max(next_q_vals)
            else:
                greedy_val_of_next_state = 0
            new_target = (
                batch.rewards()[i]
                + (1.0 - batch.game_overs()[i])
                * self.ap.algorithm.discount
                * greedy_val_of_next_state
            )
            cur_state_q.edges[idx] = new_target
            td_targets.append(
                utils_np.graphs_tuple_to_data_dicts(cur_state_q)[0]
            )
            td_error = np.abs(new_target - q_vals[batch.actions()[i]])
            td_errors.append(td_error)

        td_targets = utils_np.data_dicts_to_graphs_tuple(td_targets)
        feed_dict = utils_tf.get_feed_dict(self.input_placeholder, cur_states)
        feed_dict.update(
            utils_tf.get_feed_dict(self.target_placeholder, td_targets)
        )

        tf_results = self.sess.run(
            {
                "step": self.step_op,
                "target": self.target_placeholder,
                "losses": self.loss_ops,
                "outputs": self.output_ops,
            },
            feed_dict=feed_dict,
        )
        outputs = tf_results["outputs"][-1]
        outputs = utils_np.graphs_tuple_to_data_dicts(outputs)
        targets = tf_results["target"]
        targets = utils_np.graphs_tuple_to_data_dicts(targets)
        per_step_losses = tf_results["losses"]
        # loss of decode network applied to each processing step
        total_loss = per_step_losses[-1]

        # ignore this for now, only needed for logging
        unclipped_grads = [0]
        return total_loss, per_step_losses, unclipped_grads
