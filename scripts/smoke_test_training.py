"""Smoke test: run the full training pipeline for a few updates per phase.

Verifies:
- All 9 curriculum phases load and apply correctly
- Observations, actions, rewards are valid (no NaN/Inf)
- PPO update produces finite losses
- Phase transitions work without crashes
- Reward overrides are applied per phase

Usage:
    python scripts/smoke_test_training.py --config configs/ppo_v2_runpod.yaml
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.vec_env import SubprocVecEnv, make_env_fn
from rl.ppo import PPO
from rl.buffer import RolloutBuffer


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def smoke_test(config_path):
    config = load_config(config_path)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Reduced settings for speed
    n_envs = 8
    rollout_steps = 5
    updates_per_phase = 3

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cpu"

    # Create envs
    xml_path = os.path.join(project_root, config["env"]["xml_path"])
    env_config = {
        "frameskip": config["env"].get("frameskip", 10),
        "max_episode_steps": config["env"].get("max_episode_steps", 300),
        "reward": config.get("reward", {}),
        "action_scale": config["env"].get("action_scale", 0.3),
        "use_proxy_contacts": config["env"].get("use_proxy_contacts", False),
        "continuous_episodes": False,
    }

    print(f"Starting {n_envs} envs for smoke test...")
    env_fns = [make_env_fn(xml_path, env_config, seed=seed + i) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    print(f"Envs ready: obs_dim={vec_env.obs_dim}, act_dim={vec_env.act_dim}")

    # PPO agent
    agent = PPO(vec_env.obs_dim, vec_env.act_dim, config=config, device=device)
    agent.buffer = RolloutBuffer(
        rollout_steps, n_envs, vec_env.obs_dim, vec_env.act_dim,
        gamma=agent.gamma, gae_lambda=agent.gae_lambda,
    )

    phases = config.get("curriculum", {}).get("phases", [])
    print(f"\n{'='*70}")
    print(f"SMOKE TEST: {len(phases)} phases, {updates_per_phase} updates each, "
          f"{n_envs} envs, {rollout_steps} rollout steps")
    print(f"{'='*70}\n")

    all_passed = True

    for phase_idx, phase in enumerate(phases):
        phase_name = phase["name"]
        print(f"--- Phase {phase_idx+1}/{len(phases)}: {phase_name} ---")

        # Apply phase settings
        vec_env.set_curriculum_angle(phase.get("max_angle", 0.0))
        vec_env.set_available_faces(phase["faces"])
        if "start_faces" in phase:
            vec_env.set_start_faces(phase["start_faces"])
        else:
            vec_env.set_start_faces([1, 2, 3, 4, 5, 6])
        if "max_episode_steps" in phase:
            vec_env.set_max_episode_steps(phase["max_episode_steps"])
        if "action_scale" in phase:
            vec_env.set_action_scale(phase["action_scale"])
        if "log_std_bounds" in phase:
            lo, hi = phase["log_std_bounds"]
            agent.ac.actor.set_log_std_bounds(lo, hi)
        if "entropy_coef" in phase:
            agent.entropy_coef = phase["entropy_coef"]

        # Reward overrides
        base_reward = config.get("reward", {}).copy()
        if "reward_overrides" in phase:
            base_reward.update(phase["reward_overrides"])
        vec_env.set_reward_overrides(base_reward)

        # Reset envs for this phase
        target_faces = [int(np.random.choice(phase["faces"])) for _ in range(n_envs)]
        obs = vec_env.reset_all(target_faces=target_faces)

        phase_errors = []
        all_rewards = []
        all_obs_mins = []
        all_obs_maxs = []

        for upd in range(updates_per_phase):
            agent.buffer.reset()

            # Collect rollout
            for step in range(rollout_steps):
                obs_norm = agent._normalize_obs(obs)

                # Check obs
                if not np.isfinite(obs_norm).all():
                    phase_errors.append(f"NaN/Inf in obs_norm at step {step}")
                if obs_norm.shape != (n_envs, vec_env.obs_dim):
                    phase_errors.append(f"obs shape {obs_norm.shape} != ({n_envs}, {vec_env.obs_dim})")

                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs_norm).to(device)
                    act_t, lp_t, val_t, _ = agent.ac.get_action_and_value(obs_t)
                    actions = act_t.cpu().numpy()
                    log_probs = lp_t.cpu().numpy()
                    values = val_t.cpu().numpy()

                # Check actions
                if not np.isfinite(actions).all():
                    phase_errors.append(f"NaN/Inf in actions at step {step}")
                # Gaussian policy can sample outside [-1,1]; env clips before use.
                # Only flag extreme values that indicate a broken policy.
                if np.abs(actions).max() > 5.0:
                    phase_errors.append(f"Actions extremely out of range: max={np.abs(actions).max():.3f}")

                next_obs, rewards, dones, infos = vec_env.step(actions)

                # Check rewards
                if not np.isfinite(rewards).all():
                    phase_errors.append(f"NaN/Inf in rewards at step {step}")
                all_rewards.extend(rewards.tolist())

                # Update running stats
                if agent.normalize_obs:
                    agent.obs_rms.update(obs)
                if agent.normalize_reward:
                    agent.reward_rms.update(rewards.reshape(-1, 1))
                    reward_std = np.sqrt(agent.reward_rms.var.item()) + 1e-8
                    rewards_for_buffer = rewards / reward_std
                else:
                    rewards_for_buffer = rewards

                agent.buffer.add_step(obs_norm, actions, log_probs, rewards_for_buffer, dones, values)

                all_obs_mins.append(obs_norm.min())
                all_obs_maxs.append(obs_norm.max())
                obs = next_obs

            # GAE + PPO update
            obs_norm_last = agent._normalize_obs(obs)
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs_norm_last).to(device)
                last_values = agent.ac.critic(obs_t).cpu().numpy()
            agent.buffer.compute_gae(last_values)

            stats = agent.update()

            # Check PPO stats
            for key in ["policy_loss", "value_loss", "entropy", "approx_kl"]:
                val = stats[key]
                if not np.isfinite(val):
                    phase_errors.append(f"NaN/Inf in {key}: {val}")

        # Report
        rewards_arr = np.array(all_rewards)
        drop_penalty = base_reward.get("drop_penalty", -50)
        passed = len(phase_errors) == 0

        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  [{status}] faces={phase['faces']}, start={phase.get('start_faces', 'angle')}")
        print(f"    Reward: mean={rewards_arr.mean():.3f}, min={rewards_arr.min():.1f}, "
              f"max={rewards_arr.max():.1f}, drop_penalty={drop_penalty}")
        print(f"    Obs range: [{min(all_obs_mins):.2f}, {max(all_obs_maxs):.2f}]")
        print(f"    PPO: pi_loss={stats['policy_loss']:.4f}, v_loss={stats['value_loss']:.4f}, "
              f"entropy={stats['entropy']:.4f}, kl={stats['approx_kl']:.4f}")
        if "action_scale" in phase:
            print(f"    action_scale={phase['action_scale']}, "
                  f"log_std_bounds={phase.get('log_std_bounds', 'default')}, "
                  f"entropy_coef={phase.get('entropy_coef', 'default')}")
        if phase_errors:
            for err in phase_errors:
                print(f"    ERROR: {err}")
        print()

    vec_env.close()

    print(f"{'='*70}")
    if all_passed:
        print(f"ALL {len(phases)} PHASES PASSED")
    else:
        print(f"SOME PHASES FAILED — check errors above")
    print(f"{'='*70}")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo_v2_runpod.yaml")
    args = parser.parse_args()
    success = smoke_test(args.config)
    sys.exit(0 if success else 1)
