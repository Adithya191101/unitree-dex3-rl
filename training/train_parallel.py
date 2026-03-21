"""PPO training with parallel environments, per-env GAE, phase-aware curriculum."""

import os
import sys
import time
import yaml
import numpy as np
import torch
from collections import deque
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.vec_env import SubprocVecEnv, make_env_fn
from rl.ppo import PPO
from rl.buffer import RolloutBuffer

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate_per_face(agent, vec_env, max_steps=110):
    """Unbiased per-face evaluation: run exactly one episode per env per face.

    With n_envs=64, this gives 64 episodes per face (384 total).
    Each env runs one full episode, then we record success/failure.
    No first-to-finish bias.
    """
    n_envs = vec_env.n_envs
    # Disable continuous episodes for single-target eval
    vec_env.set_continuous_episodes(False)
    per_face_sr = {}
    per_face_dr = {}
    per_face_dist = {}

    for face in range(1, 7):
        # Set available_faces so auto-resets also use this face
        vec_env.set_available_faces([face])
        obs = vec_env.reset_all(target_faces=[face] * n_envs)

        # Track first episode completion per env
        done_mask = np.zeros(n_envs, dtype=bool)
        successes = np.zeros(n_envs, dtype=bool)
        drops = np.zeros(n_envs, dtype=bool)
        dists = np.ones(n_envs)

        for step in range(max_steps):
            obs_norm = agent._normalize_obs(obs)
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs_norm).to(agent.device)
                act_t, _, _, _ = agent.ac.get_action_and_value(obs_t, deterministic=True)
                actions = act_t.cpu().numpy()

            obs, rewards, dones, infos = vec_env.step(actions)

            for i in range(n_envs):
                if dones[i] and not done_mask[i]:
                    done_mask[i] = True
                    successes[i] = infos[i].get("achieved_goal", False)
                    drops[i] = infos[i].get("dropped", False)
                    dists[i] = infos[i].get("quat_dist", 1.0)

            if done_mask.all():
                break

        per_face_sr[face] = float(successes.sum()) / n_envs
        per_face_dr[face] = float(drops.sum()) / n_envs
        per_face_dist[face] = float(dists.mean())

    mean_sr = np.mean(list(per_face_sr.values()))
    min_sr = min(per_face_sr.values())
    mean_dr = np.mean(list(per_face_dr.values()))

    # Restore continuous_episodes to training setting (config says False)
    # BUG FIX: was hardcoded True, silently enabling continuous episodes mid-training
    vec_env.set_continuous_episodes(False)

    return per_face_sr, per_face_dr, per_face_dist, mean_sr, min_sr, mean_dr


def train(config_path=None, checkpoint=None, start_phase=0):
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "ppo_parallel_config.yaml"
        )
    config = load_config(config_path)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    seed = config.get("training", {}).get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device_name = config.get("training", {}).get("device", "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
        print("CUDA not available, falling back to CPU")
    device = torch.device(device_name)
    print(f"Using device: {device}")

    # Parallel envs
    n_envs = config.get("training", {}).get("n_envs", 8)
    xml_path = os.path.join(project_root, config["env"]["xml_path"])
    env_config = {
        "frameskip": config["env"].get("frameskip", 10),
        "max_episode_steps": config["env"].get("max_episode_steps", 300),
        "reward": config.get("reward", {}),
        "action_scale": config["env"].get("action_scale", 0.3),
        "use_proxy_contacts": config["env"].get("use_proxy_contacts", False),
        "continuous_episodes": config["env"].get("continuous_episodes", True),
    }

    print(f"Starting {n_envs} parallel environments...")
    env_fns = [make_env_fn(xml_path, env_config, seed=seed + i) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    print(f"Created {n_envs} parallel environments: obs_dim={vec_env.obs_dim}, act_dim={vec_env.act_dim}")

    # PPO agent
    agent = PPO(vec_env.obs_dim, vec_env.act_dim, config=config, device=device_name)

    # Load checkpoint if provided
    freeze_obs_norm = config.get("training", {}).get("freeze_obs_norm", False)
    if checkpoint is not None:
        agent.load(checkpoint)
        print(f"Resumed from checkpoint: {checkpoint}")
        if freeze_obs_norm:
            print("  Obs normalization FROZEN (using loaded stats, no updates)")
            agent.normalize_obs = True  # still normalize, just don't update stats

    # Create structured buffer: (rollout_steps, n_envs, ...)
    ppo_cfg = config.get("ppo", {})
    rollout_steps = ppo_cfg.get("rollout_steps", 64)
    agent.buffer = RolloutBuffer(
        rollout_steps, n_envs, vec_env.obs_dim, vec_env.act_dim,
        agent.gamma, agent.gae_lambda
    )
    total_transitions = rollout_steps * n_envs
    print(f"PPO agent: {sum(p.numel() for p in agent.ac.parameters())} params, "
          f"{total_transitions} transitions/update ({n_envs} envs x {rollout_steps} steps)")

    # TensorBoard
    writer = None
    if HAS_TB:
        run_name = f"ppo_v2_{time.strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(os.path.join(log_dir, run_name))
        print(f"TensorBoard: {os.path.join(log_dir, run_name)}")

    # CSV log
    csv_path = os.path.join(log_dir, "training_log.csv")
    with open(csv_path, "w") as f:
        f.write("update,timestep,mean_reward,mean_ep_len,success_rate,"
                "policy_loss,value_loss,entropy,approx_kl,phase,phase_name,"
                "mean_quat_dist,drop_rate,action_std,eval_sr,eval_min_sr,"
                "eval_f1,eval_f2,eval_f3,eval_f4,eval_f5,eval_f6\n")

    # Multi-phase curriculum
    phases = config.get("curriculum", {}).get("phases", [
        {"name": "HOLD", "max_angle": 0.0, "faces": [1],
         "advance_threshold": 0.90, "min_episodes": 500},
    ])

    # Apply start_phase (for resuming training in a later phase)
    current_phase = max(0, min(start_phase, len(phases) - 1))
    if start_phase > 0:
        print(f"Skipping to phase {start_phase} (0-indexed)")

    def apply_phase(phase_idx):
        """Apply curriculum phase settings to all envs and agent."""
        phase = phases[phase_idx]
        vec_env.set_curriculum_angle(phase.get("max_angle", 0.0))
        vec_env.set_available_faces(phase["faces"])
        if "start_faces" in phase:
            vec_env.set_start_faces(phase["start_faces"])
        else:
            vec_env.set_start_faces([1, 2, 3, 4, 5, 6])

        # Per-phase episode length
        if "max_episode_steps" in phase:
            vec_env.set_max_episode_steps(phase["max_episode_steps"])

        # Per-phase action scale
        if "action_scale" in phase:
            vec_env.set_action_scale(phase["action_scale"])

        # Per-phase exploration schedule
        if "log_std_bounds" in phase:
            lo, hi = phase["log_std_bounds"]
            agent.ac.actor.set_log_std_bounds(lo, hi)
        if "entropy_coef" in phase:
            agent.entropy_coef = phase["entropy_coef"]

        # Per-phase reward overrides (reset to base config first, then apply phase overrides)
        base_reward = config.get("reward", {}).copy()
        if "reward_overrides" in phase:
            base_reward.update(phase["reward_overrides"])
        vec_env.set_reward_overrides(base_reward)
        if "reward_overrides" in phase:
            tqdm.write(f"  Reward overrides: {phase['reward_overrides']}")

        tqdm.write(f"\n>>> Phase {phase_idx+1}/{len(phases)}: {phase['name']} "
                   f"(max_angle={phase.get('max_angle', 0.0):.2f} rad, faces={phase['faces']}, "
                   f"start_faces={phase.get('start_faces', 'all')}, "
                   f"advance_sr={phase['advance_threshold']:.0%}, "
                   f"min_updates={phase.get('min_updates_override', 'default')}, "
                   f"max_steps={phase.get('max_episode_steps', 'default')}, "
                   f"action_scale={phase.get('action_scale', 'default')}, "
                   f"log_std_bounds={phase.get('log_std_bounds', 'default')}, "
                   f"entropy_coef={phase.get('entropy_coef', 'default')})")

    # Training config
    total_updates = ppo_cfg.get("total_updates", 10000)
    checkpoint_interval = config.get("training", {}).get("checkpoint_interval", 200)
    log_interval = config.get("training", {}).get("log_interval", 5)
    eval_interval = config.get("training", {}).get("eval_interval", 50)

    # LR annealing
    lr_start = ppo_cfg.get("lr", 3e-4)
    lr_end = ppo_cfg.get("lr_end", lr_start)

    total_timesteps = 0
    best_reward = -float("inf")
    best_eval_sr = 0.0
    last_eval_sr = 0.0
    last_eval_face_sr = {}  # Per-face eval SR dict {1: sr, 2: sr, ...}
    last_eval_min_sr = 0.0

    # Episode tracking — per-phase windows
    phase_ep_successes = []
    phase_ep_count = 0
    phase_start_update = 1  # track when current phase started

    # Global tracking for logging (bounded deques to prevent memory leak)
    ep_rewards = deque(maxlen=500)
    ep_lengths = deque(maxlen=500)
    ep_successes = deque(maxlen=500)
    ep_quat_dists = deque(maxlen=500)
    ep_drop_rates = deque(maxlen=500)
    ep_gait_lifts = deque(maxlen=500)
    ep_gait_replaces = deque(maxlen=500)

    # Per-env episode trackers
    env_ep_rewards = np.zeros(n_envs)
    env_ep_lengths = np.zeros(n_envs, dtype=int)
    env_ep_gait_lifts = np.zeros(n_envs, dtype=int)
    env_ep_gait_replaces = np.zeros(n_envs, dtype=int)

    # Apply initial phase and reset
    apply_phase(current_phase)
    phase = phases[current_phase]
    target_faces = [int(np.random.choice(phase["faces"])) for _ in range(n_envs)]
    obs = vec_env.reset_all(target_faces=target_faces)

    print(f"\nStarting training: {total_updates} updates, {n_envs} envs, "
          f"{rollout_steps} steps/env/update")
    print(f"Native position actuators: action_scale={env_config['action_scale']}")
    print(f"Buffer: ({rollout_steps}, {n_envs}) — per-env GAE")

    pbar = tqdm(range(1, total_updates + 1), desc="Training")
    for update in pbar:
        t_start = time.time()

        # Collect rollout across all envs — one step at a time
        for step in range(rollout_steps):
            obs_norm = agent._normalize_obs(obs)
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs_norm).to(agent.device)
                act_t, lp_t, val_t, _ = agent.ac.get_action_and_value(obs_t)
                actions = act_t.cpu().numpy()
                log_probs = lp_t.cpu().numpy()
                values = val_t.cpu().numpy()

            next_obs, rewards, dones, infos = vec_env.step(actions)

            # Update obs normalization stats (skip if frozen for fine-tuning)
            if agent.normalize_obs and not freeze_obs_norm:
                agent.obs_rms.update(obs)

            # Reward normalization: scale by running std to stabilize value targets
            # Raw rewards used for episode tracking; normalized rewards for PPO buffer
            if agent.normalize_reward:
                agent.reward_rms.update(rewards.reshape(-1, 1))
                reward_std = np.sqrt(agent.reward_rms.var.item()) + 1e-8
                rewards_for_buffer = rewards / reward_std
            else:
                rewards_for_buffer = rewards

            # Store one step for all envs at once
            agent.buffer.add_step(obs_norm, actions, log_probs, rewards_for_buffer, dones, values)

            # Episode tracking (raw rewards, not normalized)
            env_ep_rewards += rewards
            env_ep_lengths += 1
            total_timesteps += n_envs

            # Accumulate gait events per env per step
            for i in range(n_envs):
                env_ep_gait_lifts[i] += infos[i].get("gait_lift_count", 0)
                env_ep_gait_replaces[i] += infos[i].get("gait_replace_count", 0)

            done_indices = np.where(dones)[0]
            for i in done_indices:
                # With continuous episodes, success = achieved at least one goal
                successes_count = infos[i].get("successes", 0)
                success = 1.0 if (successes_count > 0 or infos[i].get("achieved_goal", False)) else 0.0
                ep_rewards.append(env_ep_rewards[i])
                ep_lengths.append(env_ep_lengths[i])
                ep_successes.append(success)
                ep_quat_dists.append(infos[i].get("quat_dist", 1.0))
                ep_drop_rates.append(1.0 if infos[i].get("dropped", False) else 0.0)
                ep_gait_lifts.append(env_ep_gait_lifts[i])
                ep_gait_replaces.append(env_ep_gait_replaces[i])

                phase_ep_successes.append(success)
                phase_ep_count += 1

            env_ep_rewards[done_indices] = 0.0
            env_ep_lengths[done_indices] = 0
            env_ep_gait_lifts[done_indices] = 0
            env_ep_gait_replaces[done_indices] = 0

            obs = next_obs

        # Anneal LR
        frac = 1.0 - (update - 1) / total_updates
        lr = lr_end + frac * (lr_start - lr_end)
        for pg in agent.optimizer.param_groups:
            pg["lr"] = lr

        # Compute per-env GAE with per-env last values
        obs_norm_last = agent._normalize_obs(obs)
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_norm_last).to(agent.device)
            last_values = agent.ac.critic(obs_t).cpu().numpy()  # (n_envs,)
        agent.buffer.compute_gae(last_values)

        stats = agent.update()

        # Logging
        if len(ep_rewards) > 0:
            mean_reward = np.mean(list(ep_rewards)[-100:])
            mean_ep_len = np.mean(list(ep_lengths)[-100:])
            success_rate = np.mean(list(ep_successes)[-100:])
        else:
            mean_reward = 0.0
            mean_ep_len = 0.0
            success_rate = 0.0

        phase_name = phases[current_phase]["name"]
        phase_sr = np.mean(phase_ep_successes[-200:]) if len(phase_ep_successes) > 0 else 0.0
        # Show clamped std (what the policy actually uses), not raw parameter
        clamped_log_std = torch.clamp(agent.ac.actor.log_std,
                                      agent.ac.actor.log_std_lo,
                                      agent.ac.actor.log_std_hi)
        action_std = clamped_log_std.exp().mean().item()
        mean_quat_dist = np.mean(list(ep_quat_dists)[-100:]) if ep_quat_dists else 1.0
        drop_rate = np.mean(list(ep_drop_rates)[-100:]) if ep_drop_rates else 0.0

        t_update = time.time() - t_start

        pbar.set_postfix({
            "R": f"{mean_reward:.1f}",
            "SR": f"{success_rate:.0%}",
            "Ph": f"{current_phase+1}:{phase_name}",
            "std": f"{action_std:.3f}",
            "s/up": f"{t_update:.1f}",
        })

        # Per-face deterministic evaluation
        if update % eval_interval == 0:
            phase_max_steps = phases[current_phase].get("max_episode_steps", 50) + 10

            face_sr, face_dr, face_dist, eval_mean_sr, eval_min_sr_now, eval_mean_dr = \
                evaluate_per_face(agent, vec_env, max_steps=phase_max_steps)

            last_eval_sr = eval_mean_sr
            last_eval_min_sr = eval_min_sr_now
            last_eval_face_sr = face_sr  # Store per-face results for phase-aware advancement

            face_str = " ".join([f"F{f}:{face_sr[f]:.0%}" for f in range(1, 7)])
            tqdm.write(f"  [EVAL] update={update} mean_SR={eval_mean_sr:.1%} "
                       f"min_SR={last_eval_min_sr:.1%} DR={eval_mean_dr:.1%} | {face_str}")

            if writer:
                writer.add_scalar("eval/mean_success_rate", eval_mean_sr, total_timesteps)
                writer.add_scalar("eval/min_success_rate", last_eval_min_sr, total_timesteps)
                writer.add_scalar("eval/drop_rate", eval_mean_dr, total_timesteps)
                for f in range(1, 7):
                    writer.add_scalar(f"eval/face_{f}_sr", face_sr[f], total_timesteps)

            # Save best eval model (by mean per-face SR)
            if eval_mean_sr > best_eval_sr:
                best_eval_sr = eval_mean_sr
                agent.save(os.path.join(checkpoint_dir, "best_eval_model.pt"))
                tqdm.write(f"  [BEST] New best eval SR: {eval_mean_sr:.1%} (min={last_eval_min_sr:.1%})")

            # Re-reset envs after eval (eval changed available_faces)
            vec_env.set_available_faces(phases[current_phase]["faces"])
            target_faces = [int(np.random.choice(phases[current_phase]["faces"]))
                           for _ in range(n_envs)]
            obs = vec_env.reset_all(target_faces=target_faces)
            env_ep_rewards[:] = 0.0
            env_ep_lengths[:] = 0
            env_ep_gait_lifts[:] = 0
            env_ep_gait_replaces[:] = 0

        if writer and update % log_interval == 0:
            writer.add_scalar("train/mean_reward", mean_reward, total_timesteps)
            writer.add_scalar("train/mean_ep_length", mean_ep_len, total_timesteps)
            writer.add_scalar("train/success_rate", success_rate, total_timesteps)
            writer.add_scalar("train/phase_success_rate", phase_sr, total_timesteps)
            writer.add_scalar("train/mean_quat_dist", mean_quat_dist, total_timesteps)
            writer.add_scalar("train/drop_rate", drop_rate, total_timesteps)
            writer.add_scalar("losses/policy_loss", stats["policy_loss"], total_timesteps)
            writer.add_scalar("losses/value_loss", stats["value_loss"], total_timesteps)
            writer.add_scalar("losses/entropy", stats["entropy"], total_timesteps)
            writer.add_scalar("losses/approx_kl", stats["approx_kl"], total_timesteps)
            writer.add_scalar("train/phase", current_phase + 1, total_timesteps)
            writer.add_scalar("train/action_std", action_std, total_timesteps)
            writer.add_scalar("train/learning_rate", lr, total_timesteps)
            if ep_gait_lifts:
                writer.add_scalar("train/gait_lifts_per_ep",
                                  np.mean(list(ep_gait_lifts)[-100:]), total_timesteps)
                writer.add_scalar("train/gait_replaces_per_ep",
                                  np.mean(list(ep_gait_replaces)[-100:]), total_timesteps)

        if update % log_interval == 0:
            # Per-face eval SR for CSV (use last known values)
            f_srs = [last_eval_face_sr.get(f, 0.0) for f in range(1, 7)]
            with open(csv_path, "a") as f:
                f.write(f"{update},{total_timesteps},{mean_reward:.4f},"
                        f"{mean_ep_len:.1f},{success_rate:.4f},"
                        f"{stats['policy_loss']:.6f},{stats['value_loss']:.6f},"
                        f"{stats['entropy']:.6f},{stats['approx_kl']:.6f},"
                        f"{current_phase+1},{phase_name},"
                        f"{mean_quat_dist:.4f},{drop_rate:.4f},{action_std:.4f},"
                        f"{last_eval_sr:.4f},{last_eval_min_sr:.4f},"
                        f"{','.join(f'{s:.4f}' for s in f_srs)}\n")

        if update % checkpoint_interval == 0:
            agent.save(os.path.join(checkpoint_dir, f"ppo_update_{update}.pt"))

        if mean_reward > best_reward and len(ep_rewards) >= 20:
            best_reward = mean_reward
            agent.save(os.path.join(checkpoint_dir, "best_model.pt"))

        # Phase advancement check
        # v5 FIX: ONLY use eval SR for advancement — training SR is too noisy and caused
        # phases to race through in v4b (all 6 phases done in 595 updates, policy unprepared)
        phase = phases[current_phase]
        updates_in_phase = update - phase_start_update + 1
        min_updates_in_phase = phase.get("min_updates_override",
                                         max(50, phase.get("min_episodes", 100) // 40))
        if last_eval_face_sr:
            phase_faces = phase["faces"]
            phase_eval_sr = np.mean([last_eval_face_sr.get(f, 0.0) for f in phase_faces])
        else:
            phase_eval_sr = 0.0
        # v5: NO fallback to training SR — eval_sr MUST be positive
        advance_sr = phase_eval_sr
        if (advance_sr >= phase["advance_threshold"]
                and phase_ep_count >= phase.get("min_episodes", 100)
                and updates_in_phase >= min_updates_in_phase
                and current_phase < len(phases) - 1):
            tqdm.write(f"\n>>> Phase {current_phase+1} COMPLETE: {phase_name} "
                       f"eval_SR={last_eval_sr:.1%} train_SR={phase_sr:.1%} "
                       f"after {phase_ep_count} episodes, {updates_in_phase} updates")
            agent.save(os.path.join(checkpoint_dir,
                                    f"phase_{current_phase+1}_{phase_name}.pt"))

            current_phase += 1
            apply_phase(current_phase)
            agent.buffer.reset()  # Defensive: ensure no stale data from previous phase

            # Reset phase tracking
            phase_ep_successes = []
            phase_ep_count = 0
            phase_start_update = update + 1
            last_eval_sr = 0.0  # Reset eval SR — stale values from old phase must not carry over
            last_eval_min_sr = 0.0
            last_eval_face_sr = {}  # Reset per-face SR too

            # Reset all envs with new phase settings
            new_phase = phases[current_phase]
            target_faces = [int(np.random.choice(new_phase["faces"]))
                           for _ in range(n_envs)]
            obs = vec_env.reset_all(target_faces=target_faces)
            env_ep_rewards[:] = 0.0
            env_ep_lengths[:] = 0
            env_ep_gait_lifts[:] = 0
            env_ep_gait_replaces[:] = 0

    # Final save
    agent.save(os.path.join(checkpoint_dir, "final_model.pt"))
    print(f"\nTraining complete. Total timesteps: {total_timesteps}")
    print(f"Best mean reward: {best_reward:.4f}")
    print(f"Best eval SR (per-face mean): {best_eval_sr:.2%}")
    print(f"Final phase: {current_phase+1}/{len(phases)} ({phases[current_phase]['name']})")
    print(f"Final success rate: {success_rate:.2%}")

    if writer:
        writer.close()
    vec_env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint (fine-tuning)")
    parser.add_argument("--start-phase", type=int, default=0,
                        help="Start from this phase index (0-based)")
    args = parser.parse_args()
    train(args.config, checkpoint=args.checkpoint, start_phase=args.start_phase)
