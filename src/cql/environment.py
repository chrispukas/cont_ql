import numpy as np


class Environment:
    def __init__(self, dim, goal):
        self.env = np.zeros(dim)
        self.goal = goal

    def set_weight(self, pos, weight):
        x, y = pos
        if not self.check_if_in_bounds(pos):
            print(f"Coordinate ({pos[0]}, {pos[1]}) is out of bounds!")
            return 0
        print(f"Setting weight with value: {weight}, at coordinate: ({pos[0]}, {pos[1]})")
        self.env[y, x] = weight
    
    def set_weights(self, weights):
        pos, weight = zip(*weights)
        for i in range(len(pos)):
            self.set_weight(pos[i], weight[i])

    def check_if_in_bounds(self, pos):
        x, y = pos
        w, h = self.get_size()
        return 0 <= x < w and 0 <= y < h
    
    def get_reward(self, pos):
        x, y = np.round(pos).astype(int)

        if not self.check_if_in_bounds((x, y)):
            penalty_scale = -10
            distance = self.get_distance_to_goal(pos)
            penalty = (1 + distance) * penalty_scale
            return penalty, False
        # If in bounds, return True, if in a 'hole' (negative reward), therefore out of bounds
        return self.env[y, x], True if self.env[y, x] >= 0 else False

    def get_distance_to_goal(self, pos):
        x, y = pos
        gx, gy = self.goal
        return ((x-gx)**2 + (y-gy)**2)**0.5

    def get_size(self):
        height, width = self.env.shape
        return width, height
    
    def display(self):
        print(self.env)

