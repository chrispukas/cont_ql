from collections import deque
import random

import pandas as pd
import numpy as np
import matplotlib as plt

import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader

from src.nn import *
from environment import Environment


class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, new_state, result = zip(*batch)
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(new_state, dtype=torch.float32),
            torch.tensor(result, dtype=torch.float32) )
    def len(self):
        return len(self.buffer)


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

env = Environment(env_dim, target_pos)
env.set_weights(env_obs)
env.display()

# Tuple in the form: (state: tuple (x, y), action: tuple (dx, dy), reward: float, next_state: tuple (x + dx, y + dy), result: bool)
replay_buffer = ReplayBuffer()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently using {device} device")

# Initialize agents, assign to device
actor = ActorNetwork().to(device)
critic = CriticNetwork().to(device)

actor.optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
critic.optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)

# Training loop
for epoch in range(max_epochs):
    pos = torch.tensor(starting_pos, dtype=torch.float).to(device)

    # Run the actor, load experiences into the replay buffer
    for episode in range(max_episodes + 1):

        # -------------------
        # -- Actor Actions --
        # -------------------

        # Exploratory ->

        exploration_roll = random.random()
        explore = exploration_roll < exploration_probability

        if explore:
            exploration_action = (random.uniform(-1, 1), random.uniform(-1, 1))
            exploration_tensor = torch.tensor(exploration_action, dtype=torch.float).to(device)

            action = exploration_tensor

            print(f" # epoch:eps ({epoch}:{episode}) -> Rolled {exploration_roll}, taking random action: {exploration_action}")
        else:
            action = actor(pos)

        exploration_probability *= exploration_discount
        new_pos = pos + action

        np_new_pos = new_pos.detach().cpu().numpy()
        distance = env.get_distance_to_goal(np_new_pos)

        distance_reward = -distance
        location_reward, in_bounds = env.get_reward(np_new_pos)
        if not in_bounds:
            print(f" # epoch:eps ({epoch}:{episode}) -> not in bounds!!, tried to move to {new_pos.tolist()}")
            break

        total_reward = distance_reward + location_reward
        outcome = in_bounds if not in_bounds else distance < goal_radius

        print(f" # epoch:eps ({epoch}:{episode}) -> outcome: {outcome}, position: {pos.tolist()}, action: {action.tolist()}, reward: {total_reward}, new_position: {new_pos.tolist()}, outcome: {outcome}")

        replay_buffer.push((
            pos.detach().cpu().numpy(),
            action.detach().cpu().numpy(),
            total_reward,
            new_pos.detach().cpu().numpy(),
            outcome,
        ))

        pos = new_pos.detach()

        # ---------------------
        # -- Critic Training --
        # ---------------------

        if episode % 1 != 0 or replay_buffer.len() < training_batch_size:
            continue
        states, actions, rewards, new_states, results = replay_buffer.sample(training_batch_size)
        print(f" # epoch:eps ({epoch}:{episode}) -> Training critic on batch size of {training_batch_size}")

        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        new_states = new_states.to(device)
        results = results.to(device)

        critic_predicted_buffer = torch.cat([states, actions], dim=1)
        critic_predicted_q_values = critic(critic_predicted_buffer)
        with torch.no_grad():
            critic_target_actions = actor(new_states)
            critic_target_buffer = torch.cat([new_states, critic_target_actions], dim=1)
            critic_target_q_values = critic(critic_target_buffer)

            # q_expected = r + γ * Q (s', a')
            expected_q_values = rewards + (1 - results.float()) * discount_factor * critic_target_q_values

        loss = F.mse_loss(critic_predicted_q_values, expected_q_values)

        critic.optimizer.zero_grad()
        loss.backward()
        critic.optimizer.step()

        # --------------------
        # -- Actor Training --
        # --------------------

        if episode % 5 != 0 or replay_buffer.len() < training_batch_size:
            continue
        actor_predicted_actions = actor(states)

        actor_actions_buffer = torch.cat([states, actor_predicted_actions], dim=1)
        actor_predicted_q_values = critic(actor_actions_buffer)
        
        actor_loss = -torch.mean(actor_predicted_q_values)

        actor.optimizer.zero_grad()
        actor_loss.backward()
        actor.optimizer.step()