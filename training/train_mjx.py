"""PPO training with MJX GPU-accelerated physics, hybrid approach.

GPU physics (~71ms/step @2048 envs) + CPU reward/obs (<1ms).
Per update: 64 steps x 80ms = ~5s with 131K transitions (vs 2K on CPU).
"""

import os
import sys
import time
import yaml
import numpy as np
import torch
import jax
import jax.numpy as jnp
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.mjx_vec_env import MJXVecEnv
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


def evaluate_deterministic(agent, vec_env, target_quats_np, rng,
                           n_episodes=64, max_steps=110):
    """Deterministic eval using hybrid step."""
    n_envs = vec_env.n_envs
    rng, key = jax.random.split(rng)
    mjx_data, tgt_jax, steps_jax, pq_jax, pa_jax = vec_env.reset(key)
    mjx_data = vec_env.forward_step(mjx_data)

    state = vec_env.extract_state_np(mjx_data)
    tgt_np = np.array(tgt_jax)
    prev_actions_np = np.zeros((n_envs, 7), dtype=np.float32)
    prev_quats_np = state["cube_quat"].copy()
    step_counts = np.zeros(n_envs, dtype=np.int32)
    obs_np = vec_env.compute_obs_np(state, tgt_np, prev_actions_np)

    episodes_done = 0
    successes = 0
    drops = 0
    total_dist = 0.0

    for _ in range(max_steps):
        obs_norm = agent._normalize_obs(obs_np)
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_norm).to(agent.device)
            act_t, _, _, _ = agent.ac.get_action_and_value(obs_t, deterministic=True)
            actions_np = act_t.cpu().numpy()

        # GPU physics
        actions_jax = jnp.array(actions_np)
        mjx_data = vec_env.physics_step(mjx_data, actions_jax)
        mjx_data = vec_env.forward_step(mjx_data)

        # CPU reward/obs
        state = vec_env.extract_state_np(mjx_data)
        rewards, infos = vec_env.compute_reward_np(state, prev_quats_np, tgt_np)
        step_counts += 1

        dones = infos["dropped"] | infos["achieved_goal"] | (step_counts >= vec_env.max_episode_steps)

        done_idx = np.where(dones)[0]
        for i in done_idx:
            episodes_done += 1
            if infos["achieved_goal"][i]:
                successes += 1
            if infos["dropped"][i]:
                drops += 1
            total_dist += infos["quat_dist"][i]

        if episodes_done >= n_episodes:
            break

        prev_quats_np = state["cube_quat"].copy()
        prev_actions_np = actions_np.copy()

        # Reset done envs
        if dones.any():
            mjx_data, tgt_np, step_counts, rng = vec_env.reset_done_envs(
                mjx_data, dones, tgt_np, step_counts, rng)
            mjx_data = vec_env.forward_step(mjx_data)
            state = vec_env.extract_state_np(mjx_data)
            prev_quats_np = state["cube_quat"].copy()
            prev_actions_np[dones] = 0.0

        obs_np = vec_env.compute_obs_np(state, tgt_np, prev_actions_np)

    if episodes_done == 0:
        return 0.0, 1.0, 1.0, rng
    return successes/episodes_done, drops/episodes_done, total_dist/episodes_done, rng


def train(config_path=None, resume_path=None, start_phase=0):
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "ppo_mjx_config.yaml"
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

    device_name = config.get("training", {}).get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    print(f"PyTorch: {device_name}, JAX: {jax.devices()}")

    # MJX env
    n_envs = config.get("training", {}).get("n_envs", 2048)
    xml_path = os.path.join(project_root, config["env"]["xml_path"])
    env_config = {
        "frameskip": config["env"].get("frameskip", 10),
        "max_episode_steps": config["env"].get("max_episode_steps", 50),
        "reward": config.get("reward", {}),
        "action_scale": config["env"].get("action_scale", 0.3),
    }

    curriculum_phases = config.get("curriculum", {}).get("phases", [])
    vec_env = MJXVecEnv(xml_path=xml_path, n_envs=n_envs, config=env_config,
                        curriculum_phases=curriculum_phases)
    print(f"MJX env: {n_envs} parallel envs on GPU")

    # Domain randomization
    dr_config = config.get("domain_rand", {})
    dr_enabled = dr_config.get("enabled", False)
    obs_noise_config = None
    if dr_enabled:
        rng_dr = jax.random.PRNGKey(seed + 1000)
        vec_env.apply_domain_randomization(rng_dr, dr_config)
        obs_noise_config = dr_config.get("obs_noise", None)
        if obs_noise_config:
            print(f"  Obs noise: joint_pos={obs_noise_config.get('joint_pos', 0.05)}, "
                  f"cube_pos={obs_noise_config.get('cube_pos', 0.02)}, "
                  f"cube_ori={obs_noise_config.get('cube_ori', 0.1)}")

    # PPO agent
    agent = PPO(vec_env.obs_dim, vec_env.act_dim, config=config, device=device_name)

    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        agent.load(resume_path)

    ppo_cfg = config.get("ppo", {})
    rollout_steps = ppo_cfg.get("rollout_steps", 64)
    agent.buffer = RolloutBuffer(
        rollout_steps, n_envs, vec_env.obs_dim, vec_env.act_dim,
        agent.gamma, agent.gae_lambda
    )
    total_transitions = rollout_steps * n_envs
    print(f"PPO: {sum(p.numel() for p in agent.ac.parameters())} params, "
          f"{total_transitions} transitions/update ({n_envs} x {rollout_steps})")

    # TensorBoard
    writer = None
    if HAS_TB:
        run_name = f"ppo_mjx{n_envs}_{time.strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(os.path.join(log_dir, run_name))
        print(f"TensorBoard: {os.path.join(log_dir, run_name)}")

    # CSV
    csv_path = os.path.join(log_dir, "training_log.csv")
    with open(csv_path, "w") as f:
        f.write("update,timestep,mean_reward,mean_ep_len,success_rate,"
                "policy_loss,value_loss,entropy,approx_kl,phase,phase_name,"
                "mean_quat_dist,drop_rate,action_std,eval_sr\n")

    # Curriculum
    phases = config.get("curriculum", {}).get("phases", [
        {"name": "HOLD", "max_angle": 0.0, "faces": [1],
         "advance_threshold": 0.90, "min_episodes": 500},
    ])
    current_phase = max(0, min(start_phase, len(phases) - 1))
    if start_phase > 0:
        print(f"Starting from phase {current_phase + 1}/{len(phases)}: "
              f"{phases[current_phase]['name']}")

    def apply_phase(phase_idx):
        phase = phases[phase_idx]
        if "max_episode_steps" in phase:
            vec_env.max_episode_steps = phase["max_episode_steps"]
        if "action_scale" in phase:
            vec_env.action_scale = phase["action_scale"]
        vec_env.set_target_faces(phase["faces"])
        vec_env.set_start_faces(phase.get("start_faces", list(range(1, 7))))
        vec_env.set_curriculum_max_angle(phase.get("max_angle", 3.15))
        if "log_std_bounds" in phase:
            agent.ac.actor.set_log_std_bounds(*phase["log_std_bounds"])
        if "entropy_coef" in phase:
            agent.entropy_coef = phase["entropy_coef"]
        if "reward_overrides" in phase:
            vec_env.set_reward_config(phase["reward_overrides"])
            print(f"  Reward overrides: {phase['reward_overrides']}")
        max_angle_str = f"{phase.get('max_angle', 3.15):.1f}rad"
        print(f"\n>>> Phase {phase_idx+1}/{len(phases)}: {phase['name']} "
              f"(faces={phase['faces']}, start={phase.get('start_faces', 'all')}, "
              f"max_angle={max_angle_str}, "
              f"sr>={phase['advance_threshold']:.0%}, "
              f"steps={phase.get('max_episode_steps', 'def')}, "
              f"scale={phase.get('action_scale', 'def')})")

    # Config
    total_updates = ppo_cfg.get("total_updates", 2000)
    checkpoint_interval = config.get("training", {}).get("checkpoint_interval", 100)
    log_interval = config.get("training", {}).get("log_interval", 5)
    eval_interval = config.get("training", {}).get("eval_interval", 50)
    eval_episodes = config.get("training", {}).get("eval_episodes", 20)
    lr_start = ppo_cfg.get("lr", 3e-4)
    lr_end = ppo_cfg.get("lr_end", lr_start)

    total_timesteps = 0
    best_reward = -float("inf")
    last_eval_sr = 0.0

    phase_ep_successes = []
    phase_ep_count = 0
    phase_start_update = 1
    ep_rewards, ep_lengths, ep_successes = [], [], []
    ep_quat_dists, ep_drop_rates = [], []
    env_ep_rewards = np.zeros(n_envs)
    env_ep_lengths = np.zeros(n_envs, dtype=int)

    rng = jax.random.PRNGKey(seed)

    # Initialize
    apply_phase(current_phase)
    rng, key = jax.random.split(rng)
    mjx_data, tgt_jax, _, _, _ = vec_env.reset(key)

    # JIT warmup: physics + forward
    print("Warming up JIT (physics + forward, ~60-90s)...")
    t0 = time.time()
    dummy = jnp.zeros((n_envs, 7))
    mjx_data = vec_env.physics_step(mjx_data, dummy)
    jax.block_until_ready(mjx_data.qpos)
    print(f"  Physics JIT: {time.time()-t0:.1f}s")
    t0 = time.time()
    mjx_data = vec_env.forward_step(mjx_data)
    jax.block_until_ready(mjx_data.site_xpos)
    print(f"  Forward JIT: {time.time()-t0:.1f}s")

    # Re-reset
    rng, key = jax.random.split(rng)
    mjx_data, tgt_jax, _, _, _ = vec_env.reset(key)
    mjx_data = vec_env.forward_step(mjx_data)

    # Extract initial state
    state = vec_env.extract_state_np(mjx_data)
    target_quats_np = np.array(tgt_jax)
    prev_actions_np = np.zeros((n_envs, 7), dtype=np.float32)
    prev_quats_np = state["cube_quat"].copy()
    step_counts = np.zeros(n_envs, dtype=np.int32)

    # Gait state tracking
    tip_dists_init = np.linalg.norm(
        state["tip_pos"] - state["cube_pos"][:, None, :], axis=2)
    prev_per_finger_contact = tip_dists_init < 0.04  # (n_envs, 3)
    gait_cooldown = np.zeros(n_envs, dtype=np.int32)
    gait_cooldown_steps = 5

    obs_np = vec_env.compute_obs_np(state, target_quats_np, prev_actions_np,
                                    obs_noise=obs_noise_config)

    print(f"\nStarting training: {total_updates} updates, {n_envs} envs, "
          f"{rollout_steps} steps/update")
    print(f"Hybrid: GPU physics (~80ms/step) + CPU reward/obs (<1ms/step)")

    pbar = tqdm(range(1, total_updates + 1), desc="Training")
    for update in pbar:
        t_start = time.time()

        for step in range(rollout_steps):
            obs_norm = agent._normalize_obs(obs_np)

            # PyTorch policy
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs_norm).to(agent.device)
                act_t, lp_t, val_t, _ = agent.ac.get_action_and_value(obs_t)
                actions_np = act_t.cpu().numpy()
                log_probs_np = lp_t.cpu().numpy()
                values_np = val_t.cpu().numpy()

            # GPU physics
            actions_jax = jnp.array(actions_np)
            mjx_data = vec_env.physics_step(mjx_data, actions_jax)
            mjx_data = vec_env.forward_step(mjx_data)

            # CPU: extract state, compute reward/obs
            state = vec_env.extract_state_np(mjx_data)
            gait_cooldown = np.maximum(0, gait_cooldown - 1)
            rewards_np, infos = vec_env.compute_reward_np(
                state, prev_quats_np, target_quats_np,
                prev_per_finger_contact=prev_per_finger_contact,
                prev_actions_np=prev_actions_np,
                actions_np=actions_np,
                gait_cooldown=gait_cooldown)
            # Update gait cooldown on events
            gait_cooldown[infos["gait_event"]] = gait_cooldown_steps
            step_counts += 1

            dones_np = (infos["dropped"] | infos["achieved_goal"] |
                       (step_counts >= vec_env.max_episode_steps))

            if agent.normalize_obs:
                agent.obs_rms.update(obs_np)

            agent.buffer.add_step(obs_norm, actions_np, log_probs_np,
                                  rewards_np, dones_np.astype(np.float32), values_np)

            env_ep_rewards += rewards_np
            env_ep_lengths += 1
            total_timesteps += n_envs

            # Episode tracking
            done_indices = np.where(dones_np)[0]
            for i in done_indices:
                success = 1.0 if infos["achieved_goal"][i] else 0.0
                ep_rewards.append(env_ep_rewards[i])
                ep_lengths.append(env_ep_lengths[i])
                ep_successes.append(success)
                ep_quat_dists.append(float(infos["quat_dist"][i]))
                ep_drop_rates.append(1.0 if infos["dropped"][i] else 0.0)
                phase_ep_successes.append(success)
                phase_ep_count += 1

            env_ep_rewards[done_indices] = 0.0
            env_ep_lengths[done_indices] = 0

            # Update prev state
            prev_quats_np = state["cube_quat"].copy()
            prev_actions_np = actions_np.copy()
            prev_per_finger_contact = infos["per_finger_contact"].copy()

            # Reset done envs
            if dones_np.any():
                mjx_data, target_quats_np, step_counts, rng = \
                    vec_env.reset_done_envs(mjx_data, dones_np,
                                            target_quats_np, step_counts, rng)
                mjx_data = vec_env.forward_step(mjx_data)
                state = vec_env.extract_state_np(mjx_data)
                prev_quats_np = state["cube_quat"].copy()
                prev_actions_np[dones_np] = 0.0
                # Reset gait state for done envs
                tip_dists_reset = np.linalg.norm(
                    state["tip_pos"][dones_np] - state["cube_pos"][dones_np, None, :],
                    axis=2)
                prev_per_finger_contact[dones_np] = tip_dists_reset < 0.04
                gait_cooldown[dones_np] = 0

            obs_np = vec_env.compute_obs_np(state, target_quats_np, prev_actions_np,
                                            obs_noise=obs_noise_config)

        # LR anneal
        frac = 1.0 - (update - 1) / total_updates
        lr = lr_end + frac * (lr_start - lr_end)
        for pg in agent.optimizer.param_groups:
            pg["lr"] = lr

        # GAE
        obs_norm_last = agent._normalize_obs(obs_np)
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_norm_last).to(agent.device)
            last_values = agent.ac.critic(obs_t).cpu().numpy()
        agent.buffer.compute_gae(last_values)
        stats = agent.update()

        # Logging
        if len(ep_rewards) > 0:
            mean_reward = np.mean(ep_rewards[-200:])
            mean_ep_len = np.mean(ep_lengths[-200:])
            success_rate = np.mean(ep_successes[-200:])
        else:
            mean_reward = mean_ep_len = success_rate = 0.0

        phase_name = phases[current_phase]["name"]
        phase_sr = np.mean(phase_ep_successes[-500:]) if phase_ep_successes else 0.0
        clamped_log_std = torch.clamp(agent.ac.actor.log_std,
                                      agent.ac.actor.log_std_lo,
                                      agent.ac.actor.log_std_hi)
        action_std = clamped_log_std.exp().mean().item()
        mean_qd = np.mean(ep_quat_dists[-200:]) if ep_quat_dists else 1.0
        drop_rate = np.mean(ep_drop_rates[-200:]) if ep_drop_rates else 0.0
        t_update = time.time() - t_start

        pbar.set_postfix({
            "R": f"{mean_reward:.1f}", "SR": f"{success_rate:.0%}",
            "Ph": f"{current_phase+1}:{phase_name}",
            "std": f"{action_std:.3f}", "s/up": f"{t_update:.1f}",
        })

        # Eval
        if update % eval_interval == 0:
            max_s = phases[current_phase].get("max_episode_steps", 50) + 10
            eval_sr, eval_dr, eval_dist, rng = evaluate_deterministic(
                agent, vec_env, target_quats_np, rng,
                n_episodes=eval_episodes, max_steps=max_s)
            last_eval_sr = eval_sr
            print(f"\n  [EVAL] update={update} SR={eval_sr:.1%} DR={eval_dr:.1%} "
                  f"dist={eval_dist:.3f}")
            if writer:
                writer.add_scalar("eval/success_rate", eval_sr, total_timesteps)
                writer.add_scalar("eval/drop_rate", eval_dr, total_timesteps)

            # Re-reset after eval
            rng, key = jax.random.split(rng)
            mjx_data, tgt_jax, _, _, _ = vec_env.reset(key)
            mjx_data = vec_env.forward_step(mjx_data)
            state = vec_env.extract_state_np(mjx_data)
            target_quats_np = np.array(tgt_jax)
            prev_actions_np[:] = 0.0
            prev_quats_np = state["cube_quat"].copy()
            step_counts[:] = 0
            # Reset gait state after eval
            tip_dists_ev = np.linalg.norm(
                state["tip_pos"] - state["cube_pos"][:, None, :], axis=2)
            prev_per_finger_contact = tip_dists_ev < 0.04
            gait_cooldown[:] = 0
            obs_np = vec_env.compute_obs_np(state, target_quats_np, prev_actions_np,
                                            obs_noise=obs_noise_config)
            env_ep_rewards[:] = 0.0
            env_ep_lengths[:] = 0

        if writer and update % log_interval == 0:
            writer.add_scalar("train/mean_reward", mean_reward, total_timesteps)
            writer.add_scalar("train/success_rate", success_rate, total_timesteps)
            writer.add_scalar("train/phase_sr", phase_sr, total_timesteps)
            writer.add_scalar("train/mean_quat_dist", mean_qd, total_timesteps)
            writer.add_scalar("train/drop_rate", drop_rate, total_timesteps)
            writer.add_scalar("losses/policy_loss", stats["policy_loss"], total_timesteps)
            writer.add_scalar("losses/value_loss", stats["value_loss"], total_timesteps)
            writer.add_scalar("losses/entropy", stats["entropy"], total_timesteps)
            writer.add_scalar("train/phase", current_phase + 1, total_timesteps)
            writer.add_scalar("train/action_std", action_std, total_timesteps)

        if update % log_interval == 0:
            with open(csv_path, "a") as f:
                f.write(f"{update},{total_timesteps},{mean_reward:.4f},"
                        f"{mean_ep_len:.1f},{success_rate:.4f},"
                        f"{stats['policy_loss']:.6f},{stats['value_loss']:.6f},"
                        f"{stats['entropy']:.6f},{stats['approx_kl']:.6f},"
                        f"{current_phase+1},{phase_name},"
                        f"{mean_qd:.4f},{drop_rate:.4f},{action_std:.4f},"
                        f"{last_eval_sr:.4f}\n")

        if update % checkpoint_interval == 0:
            agent.save(os.path.join(checkpoint_dir, f"ppo_update_{update}.pt"))

        if mean_reward > best_reward and len(ep_rewards) >= 20:
            best_reward = mean_reward
            agent.save(os.path.join(checkpoint_dir, "best_model.pt"))

        # Phase advancement — ONLY based on eval SR (requires at least one eval)
        phase = phases[current_phase]
        updates_in_phase = update - phase_start_update + 1
        min_up = phase.get("min_updates_override",
                           max(eval_interval, phase.get("min_episodes", 100) // 40))
        if (last_eval_sr > 0
                and last_eval_sr >= phase["advance_threshold"]
                and phase_ep_count >= phase.get("min_episodes", 100)
                and updates_in_phase >= min_up
                and current_phase < len(phases) - 1):
            print(f"\n>>> Phase {current_phase+1} COMPLETE: {phase_name} "
                  f"eval_SR={last_eval_sr:.1%} train_SR={phase_sr:.1%} "
                  f"after {phase_ep_count} eps, {updates_in_phase} updates")
            agent.save(os.path.join(checkpoint_dir,
                                    f"phase_{current_phase+1}_{phase_name}.pt"))

            current_phase += 1
            apply_phase(current_phase)
            phase_ep_successes = []
            phase_ep_count = 0
            phase_start_update = update + 1
            last_eval_sr = 0.0

            rng, key = jax.random.split(rng)
            mjx_data, tgt_jax, _, _, _ = vec_env.reset(key)
            mjx_data = vec_env.forward_step(mjx_data)
            state = vec_env.extract_state_np(mjx_data)
            target_quats_np = np.array(tgt_jax)
            prev_actions_np[:] = 0.0
            prev_quats_np = state["cube_quat"].copy()
            step_counts[:] = 0
            # Reset gait state
            tip_dists_ph = np.linalg.norm(
                state["tip_pos"] - state["cube_pos"][:, None, :], axis=2)
            prev_per_finger_contact = tip_dists_ph < 0.04
            gait_cooldown[:] = 0
            obs_np = vec_env.compute_obs_np(state, target_quats_np, prev_actions_np,
                                            obs_noise=obs_noise_config)
            env_ep_rewards[:] = 0.0
            env_ep_lengths[:] = 0

            # Obs normalization warmup: collect obs stats for new
            # distribution before policy update (prevents NaN explosion)
            if agent.normalize_obs:
                warmup_steps = 10
                print(f"  Warming up obs normalization ({warmup_steps} steps)...")
                for _ in range(warmup_steps):
                    agent.obs_rms.update(obs_np)
                    with torch.no_grad():
                        obs_norm = agent._normalize_obs(obs_np)
                        obs_t = torch.FloatTensor(obs_norm).to(agent.device)
                        act_t, _, _, _ = agent.ac.get_action_and_value(obs_t)
                        actions_np = act_t.cpu().numpy()
                    actions_jax = jnp.array(actions_np)
                    mjx_data = vec_env.physics_step(mjx_data, actions_jax)
                    mjx_data = vec_env.forward_step(mjx_data)
                    state = vec_env.extract_state_np(mjx_data)
                    step_counts += 1
                    dones_w = (step_counts >= vec_env.max_episode_steps)
                    if dones_w.any():
                        mjx_data, target_quats_np, step_counts, rng = \
                            vec_env.reset_done_envs(mjx_data, dones_w,
                                                    target_quats_np, step_counts, rng)
                        mjx_data = vec_env.forward_step(mjx_data)
                        state = vec_env.extract_state_np(mjx_data)
                        prev_actions_np[dones_w] = 0.0
                    prev_quats_np = state["cube_quat"].copy()
                    prev_actions_np = actions_np.copy()
                    obs_np = vec_env.compute_obs_np(state, target_quats_np,
                                                    prev_actions_np,
                                                    obs_noise=obs_noise_config)
                print(f"  Warmup done. obs_rms count={agent.obs_rms.count:.0f}")

    agent.save(os.path.join(checkpoint_dir, "final_model.pt"))
    print(f"\nTraining complete. Timesteps: {total_timesteps}")
    print(f"Best reward: {best_reward:.4f}")
    print(f"Final phase: {current_phase+1}/{len(phases)} ({phases[current_phase]['name']})")
    print(f"Final SR: {success_rate:.2%}")

    if writer:
        writer.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--start_phase", type=int, default=0,
                        help="Phase index to start from (0-based)")
    args = parser.parse_args()
    train(config_path=args.config, resume_path=args.resume,
          start_phase=args.start_phase)
