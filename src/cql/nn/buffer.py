import torch
import random

from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=1000, device: torch.device=torch.device("cpu")) -> None:
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, 
             state: torch.Tensor, 
             action: torch.Tensor, 
             reward: torch.Tensor, 
             new_state: torch.Tensor, 
             result: torch.Tensor) -> None:
        experience = (state, action, reward, new_state, result)
        self.buffer.append(experience)

    def sample(self, 
               batch_size: int
               ) -> tuple:

        if len(self.buffer) < batch_size:
            print(f"Only {len(self.buffer)} out of {batch_size} samples available")
            return None

        batch = random.sample(self.buffer, batch_size)
        state, action, reward, new_state, result = zip(*batch)
        return (
            torch.stack(state).to(self.device),
            torch.stack(action).to(self.device),
            torch.stack(reward).to(self.device),
            torch.stack(new_state).to(self.device),
            torch.stack(result).to(self.device))
    def __len__(self) -> int:
        return len(self.buffer)