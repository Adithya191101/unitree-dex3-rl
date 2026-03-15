"""PPO algorithm implementation from scratch in PyTorch."""

import torch
import torch.nn as nn
import numpy as np

from rl.actor_critic import ActorCritic
from rl.buffer import RolloutBuffer


class RunningMeanStd:
    """Welford's online algorithm for running mean/std."""

    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, batch):
        batch = np.asarray(batch)
        if batch.ndim == 1:
            batch = batch[np.newaxis]
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x):
        return (x - self.mean.astype(np.float32)) / (
            np.sqrt(self.var.astype(np.float32)) + 1e-8
        )


class PPO:
    """Proximal Policy Optimization with clipped surrogate objective."""

    def __init__(self, obs_dim, act_dim, config=None, device="cpu"):
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        config = config or {}

        ppo_cfg = config.get("ppo", {})
        net_cfg = config.get("network", {})

        self.gamma = ppo_cfg.get("gamma", 0.99)
        self.gae_lambda = ppo_cfg.get("gae_lambda", 0.95)
        self.clip_eps = ppo_cfg.get("clip_eps", 0.2)
        self.value_coef = ppo_cfg.get("value_coef", 1.0)
        self.entropy_coef = ppo_cfg.get("entropy_coef", 0.0)
        self.max_grad_norm = ppo_cfg.get("max_grad_norm", 0.5)
        self.n_epochs = ppo_cfg.get("n_epochs", 4)
        self.minibatch_size = ppo_cfg.get("minibatch_size", 512)
        self.rollout_steps = ppo_cfg.get("rollout_steps", 64)

        # Build network
        hidden_dims = net_cfg.get("hidden_dims", [256, 256])
        activation_name = net_cfg.get("activation", "tanh")
        activation = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_name]
        log_std_init = net_cfg.get("log_std_init", -0.5)

        self.ac = ActorCritic(
            obs_dim, act_dim, hidden_dims, activation, log_std_init
        ).to(device)

        lr = ppo_cfg.get("lr", 3e-4)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr, eps=1e-5)

        # Observation normalization
        self.obs_rms = RunningMeanStd(obs_dim)
        self.normalize_obs = config.get("training", {}).get("normalize_obs", True)

        # Buffer is created externally by train_parallel.py (needs n_envs)
        self.buffer = None

    def _normalize_obs(self, obs):
        """Normalize observation using running statistics."""
        if self.normalize_obs:
            normed = self.obs_rms.normalize(obs)
            normed = np.clip(normed, -10.0, 10.0)
            normed = np.nan_to_num(normed, nan=0.0, posinf=10.0, neginf=-10.0)
            return normed
        return obs

    def select_action(self, obs, deterministic=False):
        """Select action for a single observation."""
        obs_norm = self._normalize_obs(obs)
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_norm).unsqueeze(0).to(self.device)
            action, log_prob, value, _ = self.ac.get_action_and_value(
                obs_t, deterministic=deterministic
            )
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    def update(self):
        """Run PPO update with value clipping.

        Returns:
            dict with loss statistics
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        grad_norm = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        n_updates = 0

        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.minibatch_size, self.device):
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["old_values"]

                # Evaluate current policy
                new_log_probs, entropy, values = self.ac.evaluate(obs, actions)

                # Policy loss (clipped surrogate)
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with clipping
                v_clipped = old_values + torch.clamp(
                    values - old_values, -self.clip_eps, self.clip_eps
                )
                v_loss1 = (values - returns) ** 2
                v_loss2 = (v_clipped - returns) ** 2
                value_loss = torch.max(v_loss1, v_loss2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.ac.parameters(), self.max_grad_norm
                )

                # NaN guard: skip optimizer step if loss or gradients are NaN
                has_nan = not torch.isfinite(loss)
                if not has_nan:
                    for p in self.ac.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            has_nan = True
                            break
                if has_nan:
                    self.optimizer.zero_grad()  # discard NaN gradients
                    print("[WARN] NaN in loss/gradients — skipping optimizer step")
                    continue

                self.optimizer.step()

                # Logging
                with torch.no_grad():
                    approx_kl = (old_log_probs - new_log_probs).mean().item()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_approx_kl += approx_kl
                n_updates += 1

        self.buffer.reset()

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "approx_kl": total_approx_kl / max(n_updates, 1),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        }

    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.ac.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "obs_rms_mean": self.obs_rms.mean,
            "obs_rms_var": self.obs_rms.var,
            "obs_rms_count": self.obs_rms.count,
        }, path)

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.ac.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "obs_rms_mean" in checkpoint:
            self.obs_rms.mean = checkpoint["obs_rms_mean"]
            self.obs_rms.var = checkpoint["obs_rms_var"]
            self.obs_rms.count = checkpoint["obs_rms_count"]
