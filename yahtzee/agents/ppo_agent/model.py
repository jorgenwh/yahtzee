import torch
from torch import nn
from torch.nn import functional as F

from yahtzee.constants import ACTION_SPACE


# 6 dice counts + 13 categories + 2 remaining rolls = 21
INPUT_SIZE = 21
HIDDEN_SIZE = 256


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        # Shared backbone
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

        # Actor head (policy)
        self.actor = nn.Linear(HIDDEN_SIZE, ACTION_SPACE)

        # Critic head (value function)
        self.critic = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.actor(x), self.critic(x)

    def get_action_probs(
        self, x: torch.Tensor, valid_actions_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get action probabilities with masking for invalid actions."""
        logits, _ = self.forward(x)
        # Mask invalid actions with large negative value
        masked_logits = torch.where(
            valid_actions_mask == 1,
            logits,
            torch.tensor(-1e8, device=logits.device),
        )
        return F.softmax(masked_logits, dim=-1)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get state value estimate."""
        _, value = self.forward(x)
        return value
