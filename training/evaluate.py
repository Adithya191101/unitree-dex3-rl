"""Evaluation script: per-face success rate, mean reward, video recording."""

import os
import sys
import argparse
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.dex_cube_env import DexCubeEnv
from rl.ppo import PPO


def evaluate(agent, env, face, n_episodes=100, max_steps=100):
    """Evaluate agent on a specific target face."""
    rewards = []
    successes = []
    lengths = []
    drops = []

    for ep in range(n_episodes):
        obs = env.reset(target_face=face)
        ep_reward = 0.0
        success = False
        dropped = False

        for step in range(max_steps):
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward

            if info.get("achieved_goal", False):
                success = True
            if info.get("dropped", False):
                dropped = True
            if done:
                break

        rewards.append(ep_reward)
        successes.append(success)
        lengths.append(step + 1)
        drops.append(dropped)

    return {
        "face": face,
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "success_rate": np.mean(successes),
        "mean_length": np.mean(lengths),
        "drop_rate": np.mean(drops),
    }


def record_video(agent, env, face, output_path, max_steps=150):
    """Record a rollout as MP4."""
    try:
        import cv2
    except ImportError:
        print("OpenCV not available, skipping video recording")
        return

    obs = env.reset(target_face=face)
    frames = []

    for step in range(max_steps):
        frame = env.render_camera("top_cam", width=480, height=480)
        frames.append(frame)

        action, _, _ = agent.select_action(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            break

    # Write video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (480, 480))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Video saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agent")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML (default: ppo_parallel_config.yaml)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--record", action="store_true", help="Record videos")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml_path = os.path.join(project_root, "models", "dex3_dice_scene_torque.xml")
    config_path = args.config or os.path.join(
        project_root, "configs", "ppo_parallel_config.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Use ALL_FACES phase overrides to match training conditions
    phases = config.get("curriculum", {}).get("phases", [])
    final_phase = phases[-1] if phases else {}
    reward_config = dict(config.get("reward", {}))
    reward_overrides = final_phase.get("reward_overrides", {})
    reward_config.update(reward_overrides)

    action_scale = final_phase.get("action_scale", config["env"].get("action_scale", 0.3))
    max_steps = final_phase.get("max_episode_steps", config["env"].get("max_episode_steps", 100))

    env_config = {
        "frameskip": config["env"].get("frameskip", 10),
        "max_episode_steps": max_steps,
        "reward": reward_config,
        "action_scale": action_scale,
        "use_proxy_contacts": config["env"].get("use_proxy_contacts", False),
    }
    env = DexCubeEnv(xml_path=xml_path, config=env_config)
    agent = PPO(env.obs_dim, env.act_dim, config=config, device="cpu")
    agent.load(args.checkpoint)
    print(f"Loaded: {args.checkpoint}")
    print(f"action_scale={action_scale}, max_steps={max_steps}, goal_threshold={reward_config.get('goal_threshold', 0.03)}")

    print(f"\n{'Face':>6} {'Success%':>10} {'Mean R':>10} {'Mean Len':>10} {'Drop%':>10}")
    print("-" * 50)

    all_results = []
    for face in range(1, 7):
        result = evaluate(agent, env, face, n_episodes=args.episodes, max_steps=max_steps)
        all_results.append(result)
        print(f"{face:>6} {result['success_rate']:>10.1%} "
              f"{result['mean_reward']:>10.2f} "
              f"{result['mean_length']:>10.1f} "
              f"{result['drop_rate']:>10.1%}")

    overall_success = np.mean([r["success_rate"] for r in all_results])
    overall_reward = np.mean([r["mean_reward"] for r in all_results])
    print("-" * 50)
    print(f"{'ALL':>6} {overall_success:>10.1%} {overall_reward:>10.2f}")

    # Record videos
    if args.record:
        video_dir = os.path.join(project_root, "videos")
        os.makedirs(video_dir, exist_ok=True)
        for face in range(1, 7):
            path = os.path.join(video_dir, f"face_{face}.mp4")
            record_video(agent, env, face, path)

    env.close()


if __name__ == "__main__":
    main()
