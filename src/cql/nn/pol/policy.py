from turtle import position
import torch
import cql.environment as env

def aggregate_rewards(environment: env.Environment, 
                      position: torch.Tensor, 
                      weights: torch.Tensor) -> torch.Tensor:
    
    # Positional Observations
    target_pos = torch.tensor(environment.target_pos, dtype=torch.float, device=environment.device)
    diff = position - target_pos
    linear_diff = -(diff**2).sqrt()

    # Boundary Constraints
    boundary_constraint = -1 if environment.check_if_in_bounds(position) else 0
    boundary_constraint_tensor = torch.tensor(boundary_constraint, dtype=torch.float).to(environment.device)

    # Area Rewards
    area_reward = environment.get_reward(position)

    # Goal Reward
    goal_radius = 0.2  # or whatever you use in your environment
    distance = torch.norm(position - target_pos)
    if distance < goal_radius:
        goal_reward = torch.tensor(10.0, dtype=torch.float, device=environment.device)  # Large positive reward
    else:
        goal_reward = -distance

    stack = torch.stack([linear_diff[0], linear_diff[1], area_reward, boundary_constraint_tensor, goal_reward], dim=0)
    return torch.dot(stack, weights)