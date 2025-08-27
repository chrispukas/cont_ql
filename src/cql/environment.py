import torch
import numpy as np


class Environment:
    def __init__(self, 
                 dim: int, 
                 target_pos: tuple,
                 start_pos: tuple,
                 device: torch.device = torch.device("cpu")) -> None:
        self.env = torch.zeros(dim)
        self.device = device
        self.target_pos = torch.tensor(target_pos, device=device)
        self.start_pos = torch.tensor(start_pos, device=device)
        self.shape = torch.tensor(dim, device=device, dtype=torch.float32)

    def set_weight(self, 
                   pos: torch.tensor, 
                   weight: float):
        if not self.check_if_in_bounds(pos):
            print(f"Coordinate ({pos[0]}, {pos[1]}) is out of bounds!")
            return 0
        print(f"Setting weight with value: {weight}, at coordinate: ({pos[0]}, {pos[1]})")
        self.env[pos] = torch.tensor(weight, device=self.device)

        
    def get_reward(self, 
                   pos: torch.Tensor):
        pos = pos.int()
        if not self.check_if_in_bounds(pos):
            penalty_scale = torch.tensor(-1)
            distance = self.get_distance_to_target(pos)
            penalty = (1 + distance) * penalty_scale
            return penalty
        # If in bounds, return True, if in a 'hole' (negative reward), therefore out of bounds
        val = self.env[pos[0]][pos[1]]
        reward = torch.tensor(val if val >= 0 else -1, device=self.device)
        return reward


    def set_weights(self, 
                    weights):
        pos, weight = zip(*weights)
        for i in range(len(pos)):
            self.set_weight(pos[i], weight[i])

    def check_if_in_bounds(self, 
                           pos: torch.Tensor):
        x, y = pos
        w, h = self.get_size()
        return 0 <= x < w and 0 <= y < h
    def check_if_at_goal(self,
                         pos: torch.Tensor) -> bool:
        dist = self.get_distance_to_target(pos)
        return dist < 0.1

    def get_distance_to_target(self, 
                             pos: torch.Tensor):
        diff = pos - self.target_pos
        return (diff**2).sum().sqrt()

    def get_size(self):
        return self.shape[1].item(), self.shape[0].item()
    
    def display(self):
        print(self.env)

