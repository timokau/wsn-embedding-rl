"""Trains an agent to produce WSN embeddings"""

from rl_coach.base_parameters import VisualizationParameters
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.core_types import (
    TrainingSteps,
    EnvironmentEpisodes,
    EnvironmentSteps,
)
from rl_coach.memories.memory import MemoryGranularity

from graphq_agent import GraphEdgesDQNAgentParameters
from wsn_environment import WSNEnvironmentParameters


def main():
    """Run the random agent on the wsn environment"""
    agent_params = GraphEdgesDQNAgentParameters()
    agent_params.memory.max_size = (MemoryGranularity.Transitions, 40000)

    env_params = WSNEnvironmentParameters()

    schedule_params = ScheduleParameters()
    schedule_params.improve_steps = TrainingSteps(1000000)
    schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(100)
    schedule_params.evaluation_steps = EnvironmentEpisodes(100)
    schedule_params.heatup_steps = EnvironmentSteps(100)

    graph_manager = BasicRLGraphManager(
        agent_params=agent_params,
        env_params=env_params,
        schedule_params=schedule_params,
        vis_params=VisualizationParameters(render=False),
    )

    graph_manager.improve()


if __name__ == "__main__":
    main()
