"""CPU MuJoCo interactive viewer for dice reorientation.

Press 1-6 to command dice reorientation. Uses CPU MuJoCo physics directly
(no JAX/MJX dependency). Works with models trained in MJX+DR.
"""

import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.ppo import PPO
from perception.face_detector import detect_top_face, get_target_quat

FACE_NORMALS = np.array([
    [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
])
FACE_TARGET_QUATS = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.7071068, 0.7071068, 0.0, 0.0],
    [0.7071068, 0.0, -0.7071068, 0.0],
    [0.7071068, 0.0, 0.7071068, 0.0],
    [0.7071068, -0.7071068, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
])
GRIP_QPOS = np.array([-0.419, -0.339, -1.047, 1.100, 1.222, 1.100, 1.222])
JOINT_NAMES = ["thumb_0", "thumb_1", "thumb_2",
               "middle_0", "middle_1", "index_0", "index_1"]
CTRL_LO = np.array([-1.0472, -1.0472, -1.74533, 0.0, 0.0, 0.0, 0.0])
CTRL_HI = np.array([1.0472, 0.724312, 0.0, 1.5708, 1.74533, 1.5708, 1.74533])


def detect_face_np(cube_quat):
    """Detect top face from quaternion."""
    w, xyz = cube_quat[0], cube_quat[1:4]
    t = 2.0 * np.cross(xyz[None, :], FACE_NORMALS)
    world_normals = FACE_NORMALS + w * t + np.cross(xyz[None, :], t)
    return int(np.argmax(world_normals[:, 2]) + 1)


def compute_obs(hand_qpos, hand_qvel, cube_pos, cube_quat, cube_angvel,
                palm_pos, tip_pos, target_quat, prev_action):
    """Compute 48-dim obs matching MJX training exactly."""
    rel_cube = cube_pos - palm_pos
    tip_flat = tip_pos.flatten()
    face = detect_face_np(cube_quat) / 6.0
    tip_dists = np.linalg.norm(tip_pos - cube_pos[None, :], axis=1)
    contact_forces = np.exp(-10.0 * tip_dists)
    return np.concatenate([
        hand_qpos, hand_qvel, rel_cube, cube_quat, cube_angvel,
        target_quat, tip_flat, [face], prev_action, contact_forces,
    ]).astype(np.float32)


class DexCubeViewerCPU:
    """CPU-only viewer using standard MuJoCo physics.

    Keys:
        1-6: Set target face
        R: Reset to grip
        P: Pause/Resume
    """

    def __init__(self, agent, xml_path, config):
        self.agent = agent
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frameskip = config.get("frameskip", 10)
        self.action_scale = config.get("action_scale", 0.3)

        # Cache joint/site IDs
        self.hand_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in JOINT_NAMES
        ]
        self.hand_qpos_adr = np.array([
            self.model.jnt_qposadr[j] for j in self.hand_joint_ids
        ])
        self.hand_qvel_adr = np.array([
            self.model.jnt_dofadr[j] for j in self.hand_joint_ids
        ])
        cube_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self.cube_qpos_adr = int(self.model.jnt_qposadr[cube_jid])
        self.cube_qvel_adr = int(self.model.jnt_dofadr[cube_jid])
        self.tip_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)
            for n in ["thumb_tip_site", "middle_tip_site", "index_tip_site"]
        ]
        self.palm_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "palm_site")

        # State
        self.target_face = None
        self.target_quat = FACE_TARGET_QUATS[0].copy()
        self.prev_action = np.zeros(7, dtype=np.float32)
        self.paused = False
        self.achieved = False
        self.holding = True

    def _do_physics(self, action):
        """Run frameskip steps of CPU MuJoCo physics."""
        action = np.clip(action, -1.0, 1.0)
        tgt = GRIP_QPOS + action * self.action_scale
        tgt = np.clip(tgt, CTRL_LO, CTRL_HI)
        self.data.ctrl[:7] = tgt
        for _ in range(self.frameskip):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def _grasp_cube(self, cube_pos, cube_quat):
        """Establish grip."""
        mujoco.mj_resetData(self.model, self.data)
        open_qpos = np.array([-0.2, -0.5, -0.3, 0.5, 0.5, 0.5, 0.5])
        for i, adr in enumerate(self.hand_qpos_adr):
            self.data.qpos[adr] = open_qpos[i]
        cq = self.cube_qpos_adr
        self.data.qpos[cq:cq+3] = cube_pos
        self.data.qpos[cq+3:cq+7] = cube_quat
        self.data.qvel[self.cube_qvel_adr:self.cube_qvel_adr+6] = 0

        # Close fingers
        for step in range(300):
            t = min(step / 200.0, 1.0)
            target = open_qpos + t * (GRIP_QPOS - open_qpos)
            self.data.ctrl[:7] = target
            self.data.qpos[cq:cq+3] = cube_pos
            self.data.qpos[cq+3:cq+7] = cube_quat
            self.data.qvel[self.cube_qvel_adr:self.cube_qvel_adr+6] = 0
            mujoco.mj_step(self.model, self.data)

        # Settle
        self.data.ctrl[:7] = GRIP_QPOS
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)
        self.prev_action[:] = 0.0

    def _extract_obs(self):
        """Extract observation from CPU MuJoCo state."""
        cq = self.cube_qpos_adr
        cv = self.cube_qvel_adr
        return compute_obs(
            hand_qpos=self.data.qpos[self.hand_qpos_adr],
            hand_qvel=self.data.qvel[self.hand_qvel_adr],
            cube_pos=self.data.qpos[cq:cq+3].copy(),
            cube_quat=self.data.qpos[cq+3:cq+7].copy(),
            cube_angvel=self.data.qvel[cv+3:cv+6].copy(),
            palm_pos=self.data.site_xpos[self.palm_site_id].copy(),
            tip_pos=self.data.site_xpos[self.tip_site_ids].copy(),
            target_quat=self.target_quat,
            prev_action=self.prev_action,
        )

    def _get_cube_quat(self):
        return self.data.qpos[self.cube_qpos_adr+3:self.cube_qpos_adr+7].copy()

    def _get_cube_pos(self):
        return self.data.qpos[self.cube_qpos_adr:self.cube_qpos_adr+3].copy()

    def key_callback(self, key):
        if 49 <= key <= 54:
            face = key - 48
            current_top = detect_face_np(self._get_cube_quat())
            if face == current_top:
                print(f"  Face {face} is already on top!")
                return
            self.target_face = face
            self.target_quat = FACE_TARGET_QUATS[face - 1].copy()
            self.achieved = False
            self.holding = False
            self.prev_action[:] = 0.0
            print(f"  Target: face {face} | Current: face {current_top} | Manipulating...")

        elif key == 82:  # R
            cube_pos = np.array([0.09, 0.0, 0.22])
            start_face = np.random.randint(1, 7)
            cube_quat = FACE_TARGET_QUATS[start_face - 1].copy()
            self._grasp_cube(cube_pos, cube_quat)
            current = detect_face_np(self._get_cube_quat())
            self.target_quat = FACE_TARGET_QUATS[current - 1].copy()
            self.target_face = None
            self.achieved = False
            self.holding = True
            print(f"  Reset -- face {current} on top. Press 1-6 to set target.")

        elif key == 80:  # P
            self.paused = not self.paused
            print(f"  {'Paused' if self.paused else 'Resumed'}")

    def run(self):
        # Initial grasp
        cube_pos = np.array([0.09, 0.0, 0.22])
        cube_quat = FACE_TARGET_QUATS[0].copy()
        self._grasp_cube(cube_pos, cube_quat)

        current = detect_face_np(self._get_cube_quat())
        self.target_quat = FACE_TARGET_QUATS[current - 1].copy()

        print()
        print("=" * 50)
        print("  Unitree Dex3-1 Dice Reorientation (CPU)")
        print("=" * 50)
        print("  Press 1-6 to select target face")
        print("  Press R to reset grip")
        print("  Press P to pause/resume")
        print("=" * 50)
        print(f"  Holding cube -- face {current} on top")
        print()

        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self.key_callback,
        ) as viewer:
            while viewer.is_running():
                if not self.paused:
                    obs = self._extract_obs()
                    action, _, _ = self.agent.select_action(obs, deterministic=True)

                    self._do_physics(action)
                    self.prev_action = action.copy()

                    # Check goal / drop
                    cube_quat = self._get_cube_quat()
                    cube_pos = self._get_cube_pos()
                    palm_z = self.data.site_xpos[self.palm_site_id, 2]

                    if not self.holding and not self.achieved:
                        quat_dist = 1.0 - np.abs(np.dot(cube_quat, self.target_quat))
                        if quat_dist < 0.25:
                            self.achieved = True
                            print(f"  Goal achieved! Face {self.target_face} on top!")

                        if cube_pos[2] < palm_z - 0.03:
                            print("  Dropped! Press R to reset.")
                            self.holding = True

                viewer.sync()
                time.sleep(0.02)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dex3-1 Dice Viewer (CPU)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml_path = os.path.join(project_root, "models", "dex3_dice_scene_torque.xml")
    config_path = args.config or os.path.join(
        project_root, "configs", "ppo_mjx_config.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    agent = PPO(48, 7, config=config, device="cpu")
    agent.load(args.checkpoint)
    print(f"Loaded: {args.checkpoint}")

    env_config = {
        "frameskip": config["env"].get("frameskip", 10),
        "action_scale": config["env"].get("action_scale", 0.3),
    }

    viewer = DexCubeViewerCPU(agent, xml_path, env_config)
    viewer.run()


if __name__ == "__main__":
    main()
