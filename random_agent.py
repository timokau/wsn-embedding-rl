"""Randomly acting agent"""

from rl_coach.agents.agent import Agent
from rl_coach.base_parameters import (
    AgentParameters,
    AlgorithmParameters,
    EmbedderScheme,
    MiddlewareScheme,
    NetworkParameters,
)
from rl_coach.exploration_policies.greedy import GreedyParameters
from rl_coach.memories.episodic import SingleEpisodeBufferParameters
from rl_coach.architectures.head_parameters import HeadParameters
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters

from rl_coach.spaces import SpacesDefinition
from rl_coach.architectures.tensorflow_components.heads.head import Head


class RandomAgent(Agent):
    """Agent that samples a random action at every step"""

    def learn_from_batch(self, batch):
        """Learns nothing"""
        return (0, [], [])

    def choose_action(self, _curr_state):
        """Chooses a random action"""
        return self.spaces.action.sample_with_info()

    def create_networks(self):
        from rl_coach.core_types import RunPhase

        self.phase = RunPhase.TRAIN
        return {}

    def set_environment_parameters(self, spaces):
        super().set_environment_parameters(spaces)
        self.spaces = spaces  # NO deepcopy

    def run_off_policy_evaluation(self):
        raise NotImplementedError()


class RandomAlgorithmParameters(AlgorithmParameters):
    """Parameters for the Random Algorithm"""

    def __init__(self):
        super().__init__()
        self.heatup_using_network_decisions = True


class IdentityHead(Head):
    """Network head that does nothing"""

    def __init__(
        self,
        agent_parameters: AgentParameters,
        spaces: SpacesDefinition,
        network_name: str,
    ):
        super().__init__(agent_parameters, spaces, network_name)
        self.name = "identity_head"

    def _build_module(self, input_layer):
        pass


class IdentityHeadParameters(HeadParameters):
    """Parameters for the IdentityHead"""

    def __init__(self):
        super().__init__("parameterized class name")

    @property
    def path(self):
        return "random_agent:IdentityHead"


class IdentityNetworkParameters(NetworkParameters):
    """Parameters for a network that does nothing"""

    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {
            "observation": InputEmbedderParameters(scheme=EmbedderScheme.Empty)
        }
        self.middleware_parameters = FCMiddlewareParameters(
            scheme=MiddlewareScheme.Empty
        )
        self.heads_parameters = [IdentityHeadParameters()]


class RandomAgentParameters(AgentParameters):
    """Parameters for the RandomAgent"""

    def __init__(self):
        super().__init__(
            algorithm=RandomAlgorithmParameters(),
            exploration=GreedyParameters(),
            memory=SingleEpisodeBufferParameters(),
            networks={"main": IdentityNetworkParameters()},
        )

    @property
    def path(self):
        return "random_agent:RandomAgent"
