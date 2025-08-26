import torch
import torch.nn as nn

from cql.nn.networks import *
import cql.nn.pol.loss as loss






def train_critic(sample: tuple,
                 actor: ActorNetwork,
                 critic: CriticNetwork,
                 discount: float) -> None:
        states, actions, rewards, new_states, results = sample

        with torch.no_grad():
            next_actions = actor(new_states)
            next_q = critic(torch.cat([new_states, next_actions], dim=1))
            
            expected_q = expected_q_value(
                 rewards=rewards,
                 discount=discount,
                 predicted_q_values=next_q,
                 valids=results
                 ).detach()
            
        predicted_q = critic(torch.cat([states, actions], dim=1))
        
        # Calculate losses, and run backpropogation algorithm to update weights
        loss = F.mse_loss(predicted_q, expected_q)
        critic.optimizer.zero_grad()
        loss.backward()
        critic.optimizer.step()

def expected_q_value(rewards: torch.Tensor,
                     discount: float,
                     predicted_q_values: torch.Tensor,
                     valids) -> torch.Tensor:
    
    # If valid_step is true, then there is a future reward
    return rewards + discount * predicted_q_values * valids


def train_actor(sample: tuple,
                actor: ActorNetwork,
                critic: CriticNetwork,
                discount: float) -> None:
    states, _, _, _, _ = sample

    actions = actor(states)
    predicted_q = critic(torch.cat([states, actions], dim=1))

    loss = -predicted_q.mean()

    actor.optimizer.zero_grad()
    loss.backward()
    actor.optimizer.step()

