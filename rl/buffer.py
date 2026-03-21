"""Rollout buffer with per-env GAE computation for PPO."""

import numpy as np
import torch


class RolloutBuffer:
    """Stores rollout data in (rollout_steps, n_envs, ...) shape.

    GAE is computed per-env to avoid crossing episode boundaries
    between different environments.
    """

    def __init__(self, rollout_steps, n_envs, obs_dim, act_dim, gamma=0.99, gae_lambda=0.95):
        self.rollout_steps = rollout_steps
        self.n_envs = n_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.obs = np.zeros((rollout_steps, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_steps, n_envs, act_dim), dtype=np.float32)
        self.log_probs = np.zeros((rollout_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((rollout_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((rollout_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((rollout_steps, n_envs), dtype=np.float32)

        self.advantages = np.zeros((rollout_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((rollout_steps, n_envs), dtype=np.float32)

        self.step_ptr = 0

    def add_step(self, obs, actions, log_probs, rewards, dones, values):
        """Add one timestep for all envs.

        Args:
            obs: (n_envs, obs_dim)
            actions: (n_envs, act_dim)
            log_probs: (n_envs,)
            rewards: (n_envs,)
            dones: (n_envs,)
            values: (n_envs,)
        """
        self.obs[self.step_ptr] = obs
        self.actions[self.step_ptr] = actions
        self.log_probs[self.step_ptr] = log_probs
        self.rewards[self.step_ptr] = rewards
        self.dones[self.step_ptr] = dones
        self.values[self.step_ptr] = values
        self.step_ptr += 1

    def compute_gae(self, last_values):
        """Compute GAE advantages per-env (no cross-env contamination).

        Args:
            last_values: (n_envs,) — V(s_{T+1}) for each env
        """
        n = self.step_ptr
        gae = np.zeros(self.n_envs, dtype=np.float32)

        for t in reversed(range(n)):
            if t == n - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]

            # Per-env: delta and GAE accumulation
            delta = (self.rewards[t]
                     + self.gamma * next_values * (1 - self.dones[t])
                     - self.values[t])
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            self.advantages[t] = gae

        self.returns[:n] = self.advantages[:n] + self.values[:n]
        # Advantage normalization is done per-minibatch in ppo.py update()

    def get_batches(self, minibatch_size, device="cpu"):
        """Flatten (steps, n_envs) -> (steps*n_envs,), shuffle, yield minibatches."""
        n = self.step_ptr
        total = n * self.n_envs

        # Flatten to (total, ...)
        obs_flat = self.obs[:n].reshape(total, -1)
        actions_flat = self.actions[:n].reshape(total, -1)
        log_probs_flat = self.log_probs[:n].ravel()
        advantages_flat = self.advantages[:n].ravel()
        returns_flat = self.returns[:n].ravel()
        values_flat = self.values[:n].ravel()

        indices = np.random.permutation(total)

        for start in range(0, total, minibatch_size):
            end = min(start + minibatch_size, total)
            idx = indices[start:end]

            yield {
                "obs": torch.FloatTensor(obs_flat[idx]).to(device),
                "actions": torch.FloatTensor(actions_flat[idx]).to(device),
                "old_log_probs": torch.FloatTensor(log_probs_flat[idx]).to(device),
                "advantages": torch.FloatTensor(advantages_flat[idx]).to(device),
                "returns": torch.FloatTensor(returns_flat[idx]).to(device),
                "old_values": torch.FloatTensor(values_flat[idx]).to(device),
            }

    def reset(self):
        """Reset buffer for next rollout."""
        self.step_ptr = 0
