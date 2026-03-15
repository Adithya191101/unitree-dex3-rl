"""Verify environment: load model, run random steps, check shapes."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.dex_cube_env import DexCubeEnv
from perception.face_detector import detect_top_face, get_target_quat


def test_env():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml_path = os.path.join(project_root, "models", "dex3_dice_scene.xml")

    print("Creating environment (Unitree Dex3-1 + Dice)...")
    env = DexCubeEnv(xml_path=xml_path)
    print(f"  obs_dim = {env.obs_dim}")
    print(f"  act_dim = {env.act_dim}")
    print(f"  n_hand_joints = {env.n_hand_joints}")
    print(f"  frameskip = {env.frameskip}")
    print(f"  ctrl_ranges:\n{env.ctrl_ranges}")

    # Test reset
    print("\nTesting reset...")
    obs = env.reset(target_face=1)
    print(f"  obs shape = {obs.shape}")
    print(f"  obs range = [{obs.min():.4f}, {obs.max():.4f}]")
    print(f"  target face = {env.target_face}")
    print(f"  current top face = {env.get_current_top_face()}")

    # Test random steps
    print("\nRunning 1000 random steps...")
    total_reward = 0
    n_nans = 0
    n_infs = 0
    n_done = 0

    for i in range(1000):
        action = np.random.uniform(-1, 1, size=env.act_dim).astype(np.float32)
        obs, reward, done, info = env.step(action)

        if np.any(np.isnan(obs)):
            n_nans += 1
        if np.any(np.isinf(obs)):
            n_infs += 1

        total_reward += reward

        if done:
            n_done += 1
            obs = env.reset(target_face=np.random.randint(1, 7))

    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Episodes completed: {n_done}")
    print(f"  NaN observations: {n_nans}")
    print(f"  Inf observations: {n_infs}")

    # Test face detection for all 6 faces
    print("\nTesting face detection (target quats)...")
    for face in range(1, 7):
        quat = get_target_quat(face)
        detected = detect_top_face(quat)
        status = "OK" if detected == face else "MISMATCH"
        print(f"  Face {face}: target quat -> detected face {detected} [{status}]")

    # Test obs shape consistency
    print("\nTesting obs shape consistency across resets...")
    for face in range(1, 7):
        obs = env.reset(target_face=face)
        assert obs.shape == (env.obs_dim,), f"Shape mismatch: {obs.shape}"
    print("  All shapes consistent!")

    # Smoke test training pipeline
    print("\nSmoke testing PPO pipeline...")
    import torch, yaml
    from rl.ppo import PPO
    config_path = os.path.join(project_root, "configs", "ppo_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    agent = PPO(env.obs_dim, env.act_dim, config=config, device="cpu")
    print(f"  Parameters: {sum(p.numel() for p in agent.ac.parameters())}")

    obs = env.reset(target_face=1)
    for step in range(64):
        action, log_prob, value = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        agent.store_transition(obs, action, log_prob, reward, done, value)
        obs = next_obs if not done else env.reset(target_face=1)

    agent.buffer.ptr = 64
    agent.compute_gae(obs)
    stats = agent.update()
    print(f"  PPO update: policy_loss={stats['policy_loss']:.4f}, value_loss={stats['value_loss']:.4f}")

    env.close()
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_env()
