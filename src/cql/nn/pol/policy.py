import torch
import cql.environment as env

def aggregate_rewards(environment: env.Environment, 
                      position: torch.Tensor, 
                      weights: torch.Tensor) -> torch.Tensor:
    
    # Positional Observations
    target_pos = torch.tensor(environment.target_pos, dtype=torch.float, device=position.device)
    diff = position - target_pos
    linear_diff = -(diff**2).sqrt()

    # Boundary Constraints
    boundary_constraint = -1 if environment.check_if_in_bounds(position) else 0
    boundary_constraint_tensor = torch.tensor(boundary_constraint, dtype=torch.float).to(position.device)

    # Area Rewards
    area_reward = environment.get_reward(position)

    print(f"Linear Diff: {linear_diff}, Area Reward: {area_reward}, Boundary Constraint: {boundary_constraint_tensor}")
    stack = torch.stack([linear_diff[0], linear_diff[1], area_reward[0], boundary_constraint_tensor], dim=0)
    return torch.dot(stack, weights)