import torch
import random
from collections import deque
from dataclasses import dataclass


@dataclass
class Transition:
    state: torch.Tensor
    action: int
    reward: torch.Tensor
    next_state: torch.Tensor
    next_valid_actions: list
    done: bool


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        next_valid_actions: list,
        done: bool,
    ) -> None:
        self.memory.append(
            Transition(state, action, reward, next_state, next_valid_actions, done)
        )

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
