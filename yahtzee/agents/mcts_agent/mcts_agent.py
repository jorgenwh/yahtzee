from yahtzee.agent import Agent
from yahtzee.yahtzee import State

from .mcts import MonteCarloTreeSearch


class MctsAgent(Agent):
    def __init__(self):
        self.mcts = MonteCarloTreeSearch()

    def get_action(self, state: State) -> int:
        self.mcts = MonteCarloTreeSearch()
        action = self.mcts.search(state, 3000)
        return action

    def get_name(self) -> str:
        return "MCTS Agent"
