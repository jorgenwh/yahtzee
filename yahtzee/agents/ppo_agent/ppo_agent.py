import os
import torch

from yahtzee.agent import Agent
from yahtzee.yahtzee import State

from yahtzee.agents.ppo_agent.utils import state_to_tensor, get_device
from yahtzee.agents.ppo_agent.model import ActorCritic


class PPOAgent(Agent):
    def __init__(self, model_path: str = "ppo_model.pth", use_cuda: bool = False):
        self.device = get_device(use_cuda)
        self.model = ActorCritic().to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def get_action(self, state: State) -> int:
        state_tensor = state_to_tensor(state, self.device)
        valid_actions_mask = torch.tensor(
            state.valid_actions, dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            action_probs = self.model.get_action_probs(
                state_tensor.unsqueeze(0), valid_actions_mask.unsqueeze(0)
            )

        # Select action with highest probability (greedy)
        action = int(torch.argmax(action_probs).item())

        return action

    def get_name(self) -> str:
        return "PPO Agent"
