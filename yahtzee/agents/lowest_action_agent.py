from yahtzee.agent import Agent
from yahtzee.yahtzee import State

class LowestActionAgent(Agent):
    def get_action(self, state: State) -> int:
        for i in range(len(state.valid_actions)):
            if state.valid_actions[i]:
                return i
        raise ValueError("No valid actions")

    def get_name(self) -> str:
        return "Lowest Action Agent"
