import torch
from typing import List
from collections import deque

from yahtzee.yahtzee import State
from yahtzee.agents.ppo_agent.model import INPUT_SIZE


def get_device(use_cuda: bool = False) -> torch.device:
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def state_to_tensor(state: State, device: torch.device | None = None) -> torch.Tensor:
    t = torch.zeros(INPUT_SIZE, dtype=torch.float32)

    # Dice counts (permutation-invariant): count of 1s, 2s, 3s, 4s, 5s, 6s
    for dice in state.dice:
        t[dice - 1] += 1

    offset = 6

    # Available categories
    for value in state.available_categories:
        t[offset] = value
        offset += 1

    # Remaining rolls
    t[offset] = int(state.remaining_rolls >= 1)
    t[offset + 1] = int(state.remaining_rolls >= 2)

    if device is not None:
        t = t.to(device)

    return t


class ValueTracker:
    def __init__(self, buffer_size: int = 100):
        self.buffer_size = buffer_size
        self.buffer: deque[float] = deque(maxlen=buffer_size)
        self.history: List[float] = []
        self.min = float("inf")
        self.max = -float("inf")
        self.cntr = 0

    def add(self, value: float) -> int:
        self.buffer.append(value)
        self.cntr += 1

        self.min = min(self.min, value)
        self.max = max(self.max, value)

        if self.cntr == self.buffer_size:
            self.history.append(sum(self.buffer) / len(self.buffer))
            self.cntr = 0

        if value == self.min:
            return -1
        if value == self.max:
            return 1
        return 0

    def get_history(self) -> List[float]:
        return self.history
