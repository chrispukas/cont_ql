import torch
import random

from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, new_state, result):
        experience = (state, action, reward, new_state, result)
        self.buffer.append(experience)

    def sample(self, 
               batch_size: int
               ) -> tuple:
        
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, new_state, result = zip(*batch)
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(new_state, dtype=torch.float32),
            torch.tensor(result, dtype=torch.float32))
    def __len__(self) -> int:
        return len(self.buffer)