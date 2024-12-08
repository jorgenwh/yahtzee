import random

from yahtzee.agent import Agent
from yahtzee.yahtzee import State

class RandomAgent(Agent):
    def get_action(self, state: State) -> int:
        valid_action_indices = [i for i, valid_action in enumerate(state.valid_actions) if valid_action]
        action = random.choice(valid_action_indices)
        return action

    def get_name(self) -> str:
        return "Random Agent"
