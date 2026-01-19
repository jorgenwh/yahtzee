import torch
from typing import List
from dataclasses import dataclass


@dataclass
class Transition:
    state: torch.Tensor
    action: int
    reward: float
    done: bool
    log_prob: float
    value: float
    valid_actions: List[int]


class RolloutBuffer:
    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.transitions: List[Transition] = []

    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        valid_actions: List[int],
    ) -> None:
        self.transitions.append(
            Transition(
                state=state,
                action=action,
                reward=reward,
                done=done,
                log_prob=log_prob,
                value=value,
                valid_actions=valid_actions,
            )
        )

    def clear(self) -> None:
        self.transitions = []

    def __len__(self) -> int:
        return len(self.transitions)

    def compute_returns_and_advantages(
        self, last_value: float, device: torch.device
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Compute returns and GAE advantages for all transitions in the buffer.

        Returns:
            states, actions, log_probs, returns, advantages, valid_actions_mask
        """
        n = len(self.transitions)
        if n == 0:
            raise ValueError("Buffer is empty")

        # Collect data
        rewards = [t.reward for t in self.transitions]
        values = [t.value for t in self.transitions]
        dones = [t.done for t in self.transitions]

        # Compute GAE advantages
        advantages = torch.zeros(n, device=device)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(dones[t])
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - float(dones[t])

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            )
            advantages[t] = last_gae

        # Compute returns
        returns = advantages + torch.tensor(values, device=device)

        # Stack states, actions, log_probs, valid_actions
        states = torch.stack([t.state for t in self.transitions]).to(device)
        actions = torch.tensor([t.action for t in self.transitions], device=device)
        log_probs = torch.tensor(
            [t.log_prob for t in self.transitions], dtype=torch.float32, device=device
        )
        valid_actions_mask = torch.tensor(
            [t.valid_actions for t in self.transitions],
            dtype=torch.float32,
            device=device,
        )

        return states, actions, log_probs, returns, advantages, valid_actions_mask
