"""Actor-Critic networks for PPO."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def orthogonal_init(module, gain=np.sqrt(2)):
    """Apply orthogonal initialization to a linear layer."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module


class Actor(nn.Module):
    """Gaussian policy network.

    Maps observations to action mean; log_std is a learnable parameter.
    """

    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256),
                 activation=nn.Tanh, log_std_init=-0.5):
        super().__init__()

        layers = []
        prev_dim = obs_dim
        for h in hidden_dims:
            layers.append(orthogonal_init(nn.Linear(prev_dim, h)))
            layers.append(activation())
            prev_dim = h
        layers.append(orthogonal_init(nn.Linear(prev_dim, act_dim), gain=0.01))

        self.net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))

        # Configurable log_std bounds (updated per curriculum phase)
        self.log_std_lo = -3.0
        self.log_std_hi = -0.7

    def forward(self, obs):
        """Return action mean."""
        return self.net(obs)

    def get_distribution(self, obs):
        """Return Normal distribution over actions."""
        mean = self.forward(obs)
        log_std = torch.clamp(self.log_std, self.log_std_lo, self.log_std_hi)
        std = log_std.exp()
        return Normal(mean, std)

    def set_log_std_bounds(self, lo, hi):
        """Update log_std clamp bounds (called on phase advance)."""
        self.log_std_lo = lo
        self.log_std_hi = hi

    def get_action(self, obs, deterministic=False):
        """Sample action and return (action, log_prob, mean)."""
        dist = self.get_distribution(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, dist.mean

    def evaluate_actions(self, obs, actions):
        """Evaluate log_prob and entropy for given obs-action pairs."""
        dist = self.get_distribution(obs)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


class Critic(nn.Module):
    """Value function V(s)."""

    def __init__(self, obs_dim, hidden_dims=(256, 256), activation=nn.Tanh):
        super().__init__()

        layers = []
        prev_dim = obs_dim
        for h in hidden_dims:
            layers.append(orthogonal_init(nn.Linear(prev_dim, h)))
            layers.append(activation())
            prev_dim = h
        layers.append(orthogonal_init(nn.Linear(prev_dim, 1), gain=1.0))

        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class ActorCritic(nn.Module):
    """Combined actor-critic module."""

    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256),
                 activation=nn.Tanh, log_std_init=-0.5):
        super().__init__()
        self.actor = Actor(obs_dim, act_dim, hidden_dims, activation, log_std_init)
        self.critic = Critic(obs_dim, hidden_dims, activation)

    def get_action_and_value(self, obs, deterministic=False):
        """Get action, log_prob, value for a single step."""
        action, log_prob, mean = self.actor.get_action(obs, deterministic)
        value = self.critic(obs)
        return action, log_prob, value, mean

    def evaluate(self, obs, actions):
        """Evaluate actions: log_prob, entropy, value."""
        log_prob, entropy = self.actor.evaluate_actions(obs, actions)
        value = self.critic(obs)
        return log_prob, entropy, value
