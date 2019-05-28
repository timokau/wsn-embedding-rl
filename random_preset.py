"""Trains an agent to produce WSN embeddings"""

from rl_coach.graph_managers.graph_manager import SimpleSchedule
from rl_coach.core_types import EnvironmentSteps
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager

from random_agent import RandomAgentParameters
from wsn_environment import WSNEnvironmentParameters


def main():
    """Run the random agent on the wsn environment"""
    agent_params = RandomAgentParameters()

    env_params = WSNEnvironmentParameters()

    schedule_params = SimpleSchedule()
    schedule_params.heatup_steps = EnvironmentSteps(1000)
    schedule_params.improve_steps = EnvironmentSteps(0)
    schedule_params.evaluation_steps = EnvironmentSteps(0)

    vis_params = VisualizationParameters(render=False)

    graph_manager = BasicRLGraphManager(
        agent_params=agent_params,
        env_params=env_params,
        schedule_params=schedule_params,
        vis_params=vis_params,
    )

    graph_manager.improve()


if __name__ == "__main__":
    main()
