import os
import torch

from yahtzee.agent import Agent
from yahtzee.yahtzee import State

from yahtzee.agents.dqn_agent.utils import state_to_tensor, get_device
from yahtzee.agents.dqn_agent.model import Model


class DQNAgent(Agent):
    def __init__(self, model_path: str = "model.pth", use_cuda: bool = False):
        self.device = get_device(use_cuda)
        self.model = Model().to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def get_action(self, state: State) -> int:
        state_tensor = state_to_tensor(state, self.device)
        action_scores = self.model(state_tensor)

        # Mask away invalid actions
        valid_action_tensor = torch.tensor(
            state.valid_actions, dtype=torch.float32, device=self.device
        )
        action_scores = torch.where(
            valid_action_tensor == 1, action_scores, torch.tensor(-1e8, device=self.device)
        )

        action = int(torch.argmax(action_scores).item())

        return action

    def get_name(self) -> str:
        return "DQN Agent"
