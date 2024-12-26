import torch
from typing import List
from collections import deque

from yahtzee.yahtzee import State
from yahtzee.agents.dqn_agent.model import INPUT_SIZE


def state_to_tensor(state: State) -> torch.Tensor:
    t = torch.zeros(INPUT_SIZE, dtype=torch.float32)

    offset = 0

    for dice in state.dice:
        roll = dice - 1
        t[offset + roll] = 1
        offset += 6

    for value in state.available_categories:
        t[offset] = value
        offset += 1

    t[offset] = int(state.remaining_rolls >= 1)
    t[offset + 1] = int(state.remaining_rolls >= 2)

    return t


class ValueTracker:
    def __init__(self, buffer_size: int = 100):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.history = []
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
