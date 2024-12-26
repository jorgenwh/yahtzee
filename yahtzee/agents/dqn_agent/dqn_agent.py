import torch
import random
from typing import List

from yahtzee.agent import Agent
from yahtzee.yahtzee import State

from yahtzee.agents.dqn_agent.utils import state_to_tensor
from yahtzee.agents.dqn_agent.model import Model


def select_action(action_scores: torch.Tensor, valid_actions: List[int], greedy: bool) -> int:
    if torch.rand(1).item() > 0.1 or greedy:
        return int(torch.argmax(action_scores).item())

    return random.choice([
        i for i, valid_action in enumerate(valid_actions) if valid_action
    ])


class DQNAgent(Agent):
    def __init__(self, model_path: str = "", greedy: bool = False):
        if model_path:
            self.model = torch.load(model_path)
        else:
            self.model = Model()
        self.greedy = greedy

    def get_action(self, state: State) -> int:
        state_tensor = state_to_tensor(state)
        action_scores = self.model(state_tensor)

        # Mask away invalid actions
        valid_action_tensor = torch.tensor(state.valid_actions, dtype=torch.float32)
        action_scores = torch.where(valid_action_tensor == 1, action_scores, torch.tensor(-1e8))

        action = select_action(action_scores, state.valid_actions, self.greedy)

        return action

    def get_name(self) -> str:
        return "DQN Agent"
