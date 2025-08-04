import pandas as pd
import numpy as np
import matplotlib as plt

import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader


# -------------------
# -- Initialize NN --
# -------------------

# Inputs: state - s: tuple (x, y)
# Outputs: action - a: tuple (x, y)
class ActorNetwork(nn.Module):
    def __init__(self, input_dim=2, output_dim=2):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.out(x)) # Apply tanh activation function, bounded to [-1, 1]
        return x

# Inputs: state - s: tuple (x, y), action - a: tuple (x, y) -> (x, y, dx, dy)
# Outputs: reward - r -> single, scalar output
class CriticNetwork(nn.Module):
    def __init__(self, input_dim=4, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
