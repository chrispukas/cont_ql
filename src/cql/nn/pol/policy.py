import torch
import cql.environment as env

def aggregate_rewards(environment: env.Environment, 
                      position: torch.Tensor, 
                      weights: torch.Tensor) -> float:
    
    # Positional Observations
    x_diff = position[0] - environment.goal[0]
    y_diff = position[1] - environment.goal[1]

    # Boundary Constraints
    boundary_constraint = -1 if environment.check_if_in_bounds(position) else 0

    # Area Rewards
    area_reward = environment.get_reward(position)

    agg_reward = x_diff * weights[0] + y_diff * weights[1] + boundary_constraint * weights[2] + area_reward * weights[3]
    return agg_reward