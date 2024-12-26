import torch
import random
from collections import deque
from dataclasses import dataclass


BUFFER_SIZE = 5000


@dataclass
class Transition:
    state: torch.Tensor
    action: int
    reward: torch.Tensor
    next_state: torch.Tensor
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
        done: bool,
    ) -> None:
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
