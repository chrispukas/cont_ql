import torch
import random

from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

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
            torch.stack(state),
            torch.stack(action),
            torch.stack(reward),
            torch.stack(new_state),
            torch.stack(result))
    def __len__(self) -> int:
        return len(self.buffer)