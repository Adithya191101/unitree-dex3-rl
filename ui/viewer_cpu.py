"""CPU-only MuJoCo interactive viewer for dice reorientation.

Uses DexCubeEnv directly so physics matches training exactly.
No JAX/MJX dependency. Input via terminal.

Type 1-6 + Enter to set target face. r = reset. q = quit.
"""

import os
import sys
import time
import threading
import numpy as np
import mujoco
import mujoco.viewer
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.dex_cube_env import DexCubeEnv
from rl.ppo import PPO
from perception.face_detector import detect_top_face


class DexCubeCPUViewer:
    def __init__(self, env, agent, max_steps=150):
        self.env = env
        self.agent = agent
        self.max_steps = max_steps
        self.target_face = None
        self.obs = None
        self.achieved = False
        self.step_count = 0
        self.running = True
        self.command = None

    def _current_face(self):
        cube_quat = self.env._get_cube_quat()
        return detect_top_face(cube_quat)

    def _cube_in_hand(self):
        """Check if cube is still held (near palm and above drop height)."""
        cube_pos = self.env._get_cube_pos()
        palm_pos = self.env.data.site_xpos[self.env.palm_site_id]
        dist = np.linalg.norm(cube_pos - palm_pos)
        return dist < 0.06 and cube_pos[2] > palm_pos[2] - 0.025

    def _start_episode(self, face):
        """Start a new manipulation episode with a fresh grip.

        Uses env.reset() to match training conditions exactly.
        Retries until the starting face differs from the target.
        """
        for attempt in range(10):
            self.obs = self.env.reset(target_face=face)
            start_face = self._current_face()
            if start_face != face:
                break
        self.target_face = face
        self.achieved = False
        self.step_count = 0
        return start_face

    def _idle_reset(self):
        """Reset to idle — hand holds cube, no target."""
        self.obs = self.env.reset()
        self.target_face = None
        self.achieved = False
        self.step_count = 0

    def _input_thread(self):
        while self.running:
            try:
                cmd = input(">>> ").strip().lower()
                self.command = cmd
            except (EOFError, KeyboardInterrupt):
                self.running = False
                break

    def _process_command(self):
        if self.command is None:
            return
        cmd = self.command
        self.command = None

        if cmd in ('1', '2', '3', '4', '5', '6'):
            face = int(cmd)
            start_face = self._start_episode(face)
            print(f"  Target: face {face} | Start: face {start_face} | Manipulating...")
        elif cmd == 'r':
            self._idle_reset()
            current = self._current_face()
            print(f"  Reset — face {current} on top. Type 1-6 for target.")
        elif cmd == 'q':
            self.running = False
        else:
            print("  Commands: 1-6 (target face), r (reset), q (quit)")

    def run(self):
        self._idle_reset()
        current = self._current_face()

        print()
        print("=" * 50)
        print("  Unitree Dex3-1 Dice Reorientation")
        print("=" * 50)
        print("  Type 1-6 + Enter  ->  set target face")
        print("  Type r + Enter    ->  reset grip")
        print("  Type q + Enter    ->  quit")
        print("=" * 50)
        print(f"  Holding cube — face {current} on top")
        print()

        input_thread = threading.Thread(target=self._input_thread, daemon=True)
        input_thread.start()

        with mujoco.viewer.launch_passive(
            self.env.model,
            self.env.data,
        ) as viewer:
            while viewer.is_running() and self.running:
                self._process_command()

                if self.target_face is not None and not self.achieved:
                    # Active episode — run policy
                    action, _, _ = self.agent.select_action(self.obs, deterministic=True)
                    self.obs, reward, done, info = self.env.step(action)
                    self.step_count += 1

                    # Check drop
                    if not self._cube_in_hand():
                        print(f"  Dropped after {self.step_count} steps — resetting...")
                        self._idle_reset()
                        current = self._current_face()
                        print(f"  Face {current} on top. Type 1-6 for new target.")
                        print(">>> ", end="", flush=True)

                    # Check goal (strict: face match + in hand + min steps)
                    elif self.step_count >= 3 and self._current_face() == self.target_face:
                        self.achieved = True
                        print(f"  Goal! Face {self.target_face} on top ({self.step_count} steps)")
                        print(">>> ", end="", flush=True)

                    # Check timeout
                    elif self.step_count >= self.max_steps:
                        print(f"  Timeout ({self.step_count} steps) — resetting...")
                        self._idle_reset()
                        current = self._current_face()
                        print(f"  Face {current} on top. Type 1-6 for new target.")
                        print(">>> ", end="", flush=True)

                else:
                    # Idle — hold at grip_qpos directly (no env overhead)
                    self.env.data.ctrl[:self.env.n_hand_joints] = self.env.grip_qpos
                    for _ in range(self.env.frameskip):
                        mujoco.mj_step(self.env.model, self.env.data)

                viewer.sync()
                time.sleep(0.02)

        self.running = False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dex3-1 Dice Viewer (CPU)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml_path = os.path.join(project_root, "models", "dex3_dice_scene_torque.xml")
    config_path = args.config or os.path.join(
        project_root, "configs", "ppo_v2_runpod.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Use final curriculum phase settings (matches training)
    phases = config.get("curriculum", {}).get("phases", [])
    final_phase = phases[-1] if phases else {}
    action_scale = final_phase.get("action_scale", config["env"].get("action_scale", 0.3))
    max_steps = final_phase.get("max_episode_steps", config["env"].get("max_episode_steps", 150))
    reward_config = dict(config.get("reward", {}))
    reward_config.update(final_phase.get("reward_overrides", {}))

    env_config = {
        "frameskip": config["env"].get("frameskip", 10),
        "max_episode_steps": max_steps,
        "reward": reward_config,
        "action_scale": action_scale,
        "use_proxy_contacts": config["env"].get("use_proxy_contacts", False),
    }
    env = DexCubeEnv(xml_path=xml_path, config=env_config)

    # Set curriculum to ALL_FACES (all start faces, full rotation)
    env.set_curriculum_max_angle(np.pi)
    env.set_curriculum_start_faces([1, 2, 3, 4, 5, 6])

    agent = PPO(env.obs_dim, env.act_dim, config=config, device="cpu")
    agent.load(args.checkpoint)
    print(f"Loaded: {args.checkpoint}")
    print(f"action_scale={action_scale}, max_steps={max_steps}")

    viewer = DexCubeCPUViewer(env, agent, max_steps=max_steps)
    viewer.run()


if __name__ == "__main__":
    main()
