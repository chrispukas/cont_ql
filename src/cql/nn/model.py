import random
import torch

from cql.nn import networks, model, buffer

import cql.environment as env
import cql.nn.pol.policy as pol
import cql.nn.pol.loss as loss

import cql.nn.training as train


def run_sim(environment: env.Environment, 
            weights: torch.Tensor,
            device: torch.device,
            learning_rate: float,
            discount_factor: float,
            max_epochs: int,
            max_episodes: int,
            training_batch_size: int,
            critic_training_step: int,
            actor_training_step: int) -> list[tuple[float, float]]:
      # Tuple in the form: (state: tuple (x, y), action: tuple (dx, dy), reward: float, next_state: tuple (x + dx, y + dy), result: bool)
      replay_buffer = buffer.ReplayBuffer(device=device)

      # Initialize agents and dock to device
      actor = networks.ActorNetwork(device=device).to(device)
      critic = networks.CriticNetwork(device=device, layer_width=training_batch_size).to(device)

      actor.optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
      critic.optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)
      
      paths = []

      for iter in range(max_epochs):
            print(f"Current Epoch: {iter}")
            path = epoch(replay_buffer=replay_buffer, 
                  actor=actor, 
                  critic=critic, 
                  environment=environment,
                  weights=weights,
                  discount_factor=discount_factor,
                  starting_position=environment.start_pos,
                  max_episodes=max_episodes,
                  critic_training_step=critic_training_step,
                  actor_training_step=actor_training_step,
                  training_batch_size=training_batch_size)
            if path is not None:
                  paths.append(path)
      return paths

def epoch(replay_buffer: buffer.ReplayBuffer, 
          actor: networks.ActorNetwork, 
          critic: networks.CriticNetwork, 
          environment: env.Environment,
          weights: torch.Tensor,
          discount_factor: float,
          starting_position: torch.Tensor,
          max_episodes: int,
          critic_training_step: int,
          actor_training_step: int,
          training_batch_size: int) -> list[list[tuple[float, float]]]:
      path = []

      position = starting_position
      print(f"Starting Position: {position}")

      print(type(position))

      # Run the actor, load experiences into the replay buffer
      for ep in range(max_episodes + 1):
            new_position, in_bounds = episode(replay_buffer=replay_buffer,
                  actor=actor,
                  critic=critic,
                  environment=environment,
                  position=position,
                  weights=weights,
                  discount_factor=discount_factor,
                  curr_episode=ep,
                  critic_training_step=critic_training_step,
                  actor_training_step=actor_training_step,
                  training_batch_size=training_batch_size)
            position = new_position
            path.append(tuple(new_position.detach().cpu().numpy()))

            in_bounds_bool = in_bounds.detach().item()
            if in_bounds_bool is False:
                  print(f"Entered invalid position, breaking epoch {ep}.")
                  return path
            at_goal = environment.check_if_at_goal(position)


      return path


# For each episode ->
# 1. Select an action for the actor using the current policy

def episode(replay_buffer: buffer.ReplayBuffer,
            actor: networks.ActorNetwork,
            critic: networks.CriticNetwork,
            environment: env.Environment,
            position: torch.Tensor,
            weights: torch.Tensor,
            discount_factor: float,
            curr_episode: int,
            critic_training_step: int,
            actor_training_step: int,
            training_batch_size: int) -> torch.Tensor:

            # Take actors prediction, and use for the current action
            predicted_action = actor(position)
            new_position = torch.add(predicted_action, position)
            reward = pol.aggregate_rewards(environment, new_position, weights=weights).unsqueeze(0)
            in_bounds = environment.check_if_in_bounds(new_position)
            print(in_bounds)

            print(f"{position} -> {new_position}, action: {predicted_action}")

            # Commit to replay buffer for training, in the form of a tuple: (state, action, reward, new_state, valid)
            replay_buffer.push(position.detach(), 
                               predicted_action.detach(), 
                               reward.detach(), 
                               new_position.detach(), 
                               in_bounds.detach(),)
            position = new_position

            if curr_episode % critic_training_step == 0 and curr_episode > 0 and len(replay_buffer) >= training_batch_size:
                sample = replay_buffer.sample(training_batch_size)
                if sample is None:
                    return position
                train.train_critic(sample, actor, critic, discount_factor)

            if curr_episode % actor_training_step == 0 and curr_episode > 0 and len(replay_buffer) >= training_batch_size:
                 sample = replay_buffer.sample(training_batch_size)
                 if sample is None:
                     return position
                 train.train_actor(sample, actor, critic, discount_factor)

            return (position, in_bounds)