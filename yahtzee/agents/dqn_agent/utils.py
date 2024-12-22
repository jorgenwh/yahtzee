import torch

from yahtzee.yahtzee import State
from yahtzee.agents.dqn_agent.model import INPUT_SIZE

def state_to_tensor(state: State) -> torch.Tensor:
    t = torch.zeros(INPUT_SIZE, dtype=torch.float32)

    offset = 0

    for dice in state.dice:
        roll = dice - 1
        t[offset + roll] = 1
        offset += 6

    for _, value in state.available_categories.items():
        t[offset] = value
        offset += 1

    t[offset] = int(state.remaining_rolls >= 1)
    t[offset + 1] = int(state.remaining_rolls >= 2)

    return t


class AverageMeter():
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f"{round(self.avg, 4)}"

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
