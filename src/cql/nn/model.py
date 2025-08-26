import random
import torch

from cql.nn import networks, model, buffer

import cql.environment as env
import cql.nn.pol.policy as pol
import cql.nn.pol.loss as loss


# ----------------
# -- PARAMETERS --
# ----------------

max_epochs = 50000
max_episodes = 4000

training_batch_size = 200
learning_rate = 0.02
discount_factor = 0.99
exploration_probability = 0.5
exploration_discount = 0.9

starting_pos = (4, 4)
target_pos = (0, 0)
env_dim = (5, 5)
env_obs = [((1, 1), -9999), ((2, 1), -9999), ((1, 2), -9999), ((2, 2), -9999)]

goal_radius = 0.2

e = env.Environment(env_dim, target_pos)
e.set_weights(env_obs)
e.display()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently using {device} device")


def run_sim(environment: env.Environment, 
            weights: torch.Tensor) -> None:
    # Tuple in the form: (state: tuple (x, y), action: tuple (dx, dy), reward: float, next_state: tuple (x + dx, y + dy), result: bool)
    replay_buffer = buffer.ReplayBuffer()

    # Initialize agents and dock to device
    actor = networks.ActorNetwork().to(device)
    critic = networks.CriticNetwork().to(device)
    target_critic = networks.CriticNetwork().to(device)

    actor.optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    critic.optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)
    target_critic.optimizer = torch.optim.Adam(target_critic.parameters(), lr=learning_rate)

    for iter in range(max_epochs):
        print(f"Current Epoch: {iter}")
        epoch(replay_buffer=replay_buffer, 
              actor=actor, 
              critic=critic, 
              target_critic=target_critic,
              environment=environment,
              weights=weights)

def epoch(replay_buffer: buffer.ReplayBuffer, 
          actor: networks.ActorNetwork, 
          critic: networks.CriticNetwork, 
          target_critic: networks.CriticNetwork,
          environment: env.Environment,
          weights: torch.Tensor) -> None:

    pos = torch.tensor(starting_pos, dtype=torch.float).to(device)
    temp_exploration_probability = exploration_probability

    # Run the actor, load experiences into the replay buffer
    for ep in range(max_episodes + 1):
         position = torch.tensor(starting_pos, dtype=torch.float).to(device)
         episode(replay_buffer=replay_buffer,
                 actor=actor,
                 critic=critic,
                 target_critic=target_critic,
                 environment=environment,
                 position=position,
                 weights=weights)



# For each episode ->
# 1. Select an action for the actor using the current policy

def episode(replay_buffer: buffer.ReplayBuffer,
            actor: networks.ActorNetwork,
            critic: networks.CriticNetwork,
            target_critic: networks.CriticNetwork,
            environment: env.Environment,
            position: torch.Tensor,
            weights: torch.Tensor) -> None:

            predicted_action = actor(position)
            new_position = torch.add(predicted_action, position)

            reward = pol.aggregate_rewards(environment, new_position, weights=weights)
            print(reward)

            # Commit to replay buffer for training, in the form of a tuple: (state, action, reward, new_state, result)
            replay_buffer.push(position, predicted_action, reward, new_position, False)

            position = new_position

weights = torch.tensor([1.0, 1.0, 10.0, 5.0], dtype=torch.float).to(device)
run_sim(e, weights)