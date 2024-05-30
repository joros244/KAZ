import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # Orthogonal initialization. Implementation trick PPO:
    # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ 2nd point
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PolicyValueNetwork(nn.Module):
    # CleanRL PPONetwork: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
    def __init__(self, obs_space, action_space):
        # 2 Networks in the same class -> Actor (Policy) and Critic (Value). Backbone is not shared:
        # see implementation trick PPO: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ 13th
        # point
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_space.n), std=0.01),
        )

    def _get_value(self, x):
        return self.critic(x)

    def get_values(self, t):
        values = []
        for i in range(t.shape[0]):
            x = t[i]
            x = x.flatten()
            values.append(self._get_value(x))
        values = torch.stack(values)
        return values

    def _get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_actions_and_values(self, t, a=None):
        actions = []
        log_probs = []
        entropies = []
        values = []

        for i in range(t.shape[0]):
            x = t[i]
            x = x.flatten()
            action, log_prob, entropy, value = self._get_action_and_value(x, a)
            actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)

        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        values = torch.stack(values)
        return actions, log_probs, entropies, values
