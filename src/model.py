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

max_epochs = 5
max_episodes = 200

training_batch_size = 20
learning_rate = 0.001

starting_pos = (4, 4)
target_pos = (0, 0)
env_dim = (5, 5)
env_obs = [((1, 1), -99), ((2, 1), -99), ((1, 2), -99), ((2, 2), -99)]

goal_radius = 0.2

env = Environment(env_dim, target_pos)
env.set_weights(env_obs)
env.display()

# Tuple in the form: (state: tuple (x, y), action: tuple (dx, dy), reward: float, next_state: tuple (x + dx, y + dy), result: bool)
replay_buffer = ReplayBuffer()

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Currently using {device} device")

# Initialize agents, assign to device
actor = ActorNetwork().to(device)
critic = CriticNetwork().to(device)

actor.optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
critic.optimiser = torch.optim.Adam(actor.parameters(), lr=learning_rate)

# Training loop
for epoch in range(max_epochs):
    pos = torch.tensor(starting_pos, dtype=torch.float).to(device)

    # Run the actor, load experiences into the replay buffer
    for episode in range(max_episodes + 1):

        # -------------------
        # -- Actor Actions --
        # -------------------

        action = actor(pos)
        new_pos = pos + action

        np_new_pos = new_pos.detach().cpu().numpy()
        distance = env.get_distance_to_goal(np_new_pos)

        distance_reward = -distance
        location_reward = env.get_reward(np_new_pos)
        total_reward = distance_reward + location_reward
        outcome = distance < goal_radius

        print(f" # epoch:eps ({epoch}:{episode}) -> outcome: {outcome}, position: {pos.tolist()}, action: {action.tolist()}, reward: {total_reward}, new_position: {new_pos.tolist()}")

        replay_buffer.push((
            pos.detach().cpu().numpy(),
            action.detach().cpu().numpy(),
            total_reward,
            new_pos.detach().cpu().numpy(),
            outcome,
        ))

        pos.copy_(new_pos)

        # ---------------------
        # -- Critic Training --
        # ---------------------

        if episode % training_batch_size != 0 or replay_buffer.len() < training_batch_size:
            continue
        states, actions, rewards, new_states, results = replay_buffer.sample(training_batch_size)
        print(f" # epoch:eps ({epoch}:{episode}) -> Training critic on batch size of {training_batch_size}")

        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        new_states = new_states.to(device)
        results = results.to(device)

        predicted_q_values = critic(states, actions)

        with torch.no_grad():
            target_actions = actor(new_states)
            expected_q_values = critic(states, target_actions)

        loss = F.mse_loss(predicted_q_values, expected_q_values)

        critic.optimizer.zero_grad()
        loss.backward()
        critic.optimizer.step()
