from yahtzee.agent import Agent
from yahtzee.yahtzee import State

from .mcts import MonteCarloTreeSearch


MCTS_ROLLOUTS = 5000


class MctsAgent(Agent):
    def __init__(self):
        self.mcts = MonteCarloTreeSearch()

    def get_action(self, state: State) -> int:
        self.mcts = MonteCarloTreeSearch()
        action = self.mcts.search(state, MCTS_ROLLOUTS)
        return action

    def get_name(self) -> str:
        return "MCTS Agent [" + str(MCTS_ROLLOUTS) + " rollouts]"
