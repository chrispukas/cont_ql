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
    def __init__(self, 
                 input_dim: int=2,
                 output_dim: int=2, 
                 dropout: float=0.2, 
                 layer_width: int=64,
                 device: torch.device=torch.device("cpu"),
                 ) -> None:
        
        super(ActorNetwork, self).__init__()
        self.device = device

        self.fc1 = nn.Linear(input_dim, layer_width) 
        self.fc2 = nn.Linear(layer_width, layer_width)
        self.out = nn.Linear(layer_width, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.normalise_1 = nn.LayerNorm(layer_width)
        self.normalise_2 = nn.LayerNorm(layer_width)

    def forward(self, 
                x: torch.Tensor
                ) -> torch.Tensor:
        
        x = F.relu(self.normalise_1(self.fc1(x)))
        x = F.relu(self.normalise_2(self.fc2(x)))
        x = self.dropout(x)
        x = self.out(x)
        x = torch.tanh(x)  # Apply tanh bounding, bounded to [-1, 1]
        return x

# Inputs: state - s: tuple (x, y), action - a: tuple (x, y) -> (x, y, dx, dy)
# Outputs: reward - r -> single, scalar output
class CriticNetwork(nn.Module):
    def __init__(self, 
                 input_dim: int=4,
                 output_dim: int=1, 
                 layer_width: int=20,
                 device: torch.device=torch.device("cpu")
                 ) -> None:
        self.device = device

        super().__init__()
        self.fc1 = nn.Linear(input_dim, layer_width)
        self.fc2 = nn.Linear(layer_width, layer_width)
        self.out = nn.Linear(layer_width, output_dim)

        self.normalise_1 = nn.LayerNorm(layer_width)
        self.normalise_2 = nn.LayerNorm(layer_width)

    def forward(self, 
                x: torch.Tensor
                ) -> torch.Tensor:
        x = F.relu(self.normalise_1(self.fc1(x)))
        x = F.relu(self.normalise_2(self.fc2(x)))
        x = self.out(x)
        return x