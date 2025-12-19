from torch import nn
from torch.nn import functional as F

from yahtzee.constants import ACTION_SPACE


# 6 dice counts + 13 categories + 2 remaining rolls = 21
INPUT_SIZE = 21
HIDDEN_SIZE = 256


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, ACTION_SPACE)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
