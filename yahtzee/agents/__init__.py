# Import and add your model here
from .random_agent.random_agent import RandomAgent
from .lowest_action_agent.lowest_action_agent import LowestActionAgent
from .very_greedy_agent.very_greedy_agent import VeryGreedyAgent
from .mcts_agent.mcts_agent import MctsAgent
from .dqn_agent.dqn_agent import DQNAgent

__all__ = ["RandomAgent", "LowestActionAgent", "VeryGreedyAgent", "MctsAgent", "DQNAgent"]
