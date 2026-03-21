"""MuJoCo interactive viewer: MJX physics + mesh rendering.

Press 1-6 to command dice reorientation. The policy runs on MJX GPU physics
(matching training) while rendering uses the mesh-based CPU model.
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

# Face normals and target quats (matching MJX training)
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
    """Detect top face from quaternion (numpy, matches MJX training)."""
    q = cube_quat
    w, xyz = q[0], q[1:4]
    xyz_exp = xyz[None, :]       # (1, 3)
    n_exp = FACE_NORMALS          # (6, 3)
    t = 2.0 * np.cross(xyz_exp, n_exp)
    w_exp = w
    world_normals = n_exp + w_exp * t + np.cross(xyz_exp, t)
    dots = world_normals[:, 2]
    return int(np.argmax(dots) + 1)


def compute_obs(hand_qpos, hand_qvel, cube_pos, cube_quat, cube_angvel,
                palm_pos, tip_pos, target_quat, prev_action):
    """Compute 48-dim obs matching MJX training exactly."""
    rel_cube = cube_pos - palm_pos
    tip_flat = tip_pos.flatten()
    face = detect_face_np(cube_quat) / 6.0
    # Distance-based proxy contact forces (matches MJX)
    tip_dists = np.linalg.norm(tip_pos - cube_pos[None, :], axis=1)
    contact_forces = np.exp(-10.0 * tip_dists)
    return np.concatenate([
        hand_qpos, hand_qvel, rel_cube, cube_quat, cube_angvel,
        target_quat, tip_flat, [face], prev_action, contact_forces,
    ]).astype(np.float32)


class DexCubeViewer:
    """Hybrid viewer: MJX GPU physics + mesh CPU rendering.

    Keys:
        1-6: Set target face (robot manipulates to show this face)
        R: Reset to grip
        P: Pause/Resume
    """

    def __init__(self, agent, mjx_xml_path, render_xml_path, config):
        import jax
        import jax.numpy as jnp
        from functools import partial
        from mujoco import mjx

        self.agent = agent
        self.jax = jax
        self.jnp = jnp

        # ---- MJX model (primitive collisions — for physics) ----
        self.mj_model = mujoco.MjModel.from_xml_path(mjx_xml_path)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.frameskip = config.get("frameskip", 10)
        self.action_scale = config.get("action_scale", 0.3)

        # ---- Render model (mesh visuals) ----
        self.render_model = mujoco.MjModel.from_xml_path(render_xml_path)
        self.render_data = mujoco.MjData(self.render_model)

        # ---- Cache joint/site IDs (same names in both XMLs) ----
        self.hand_joint_ids = [
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in JOINT_NAMES
        ]
        self.hand_qpos_adr = np.array([
            self.mj_model.jnt_qposadr[j] for j in self.hand_joint_ids
        ])
        self.hand_qvel_adr = np.array([
            self.mj_model.jnt_dofadr[j] for j in self.hand_joint_ids
        ])
        cube_jid = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self.cube_qpos_adr = int(self.mj_model.jnt_qposadr[cube_jid])
        self.cube_qvel_adr = int(self.mj_model.jnt_dofadr[cube_jid])
        self.tip_site_ids = [
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, n)
            for n in ["thumb_tip_site", "middle_tip_site", "index_tip_site"]
        ]
        self.palm_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "palm_site")

        # Render model IDs (for syncing state)
        self.render_hand_qpos_adr = np.array([
            self.render_model.jnt_qposadr[
                mujoco.mj_name2id(self.render_model, mujoco.mjtObj.mjOBJ_JOINT, n)
            ] for n in JOINT_NAMES
        ])
        render_cube_jid = mujoco.mj_name2id(
            self.render_model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self.render_cube_qpos_adr = int(self.render_model.jnt_qposadr[render_cube_jid])

        # ---- JIT compile MJX physics ----
        # Use single-step JIT (avoids fori_loop tracing issues)
        self._jit_step = jax.jit(partial(mjx.step, self.mjx_model))
        self._jit_forward = jax.jit(partial(mjx.forward, self.mjx_model))
        self._grip_jax = jnp.array(GRIP_QPOS)
        self._clo_jax = jnp.array(CTRL_LO)
        self._chi_jax = jnp.array(CTRL_HI)

        # ---- State ----
        self.target_face = None
        self.target_quat = FACE_TARGET_QUATS[0].copy()
        self.prev_action = np.zeros(7, dtype=np.float32)
        self.paused = False
        self.achieved = False
        self.holding = True
        self.mjx_data = None

    def _do_physics(self, action_np):
        """Run frameskip steps of MJX physics with the given action."""
        jnp = self.jnp
        action = jnp.clip(jnp.array(action_np), -1.0, 1.0)
        tgt = self._grip_jax + action * self.action_scale
        tgt = jnp.clip(tgt, self._clo_jax, self._chi_jax)
        self.mjx_data = self.mjx_data.replace(
            ctrl=self.mjx_data.ctrl.at[:7].set(tgt))
        for _ in range(self.frameskip):
            self.mjx_data = self._jit_step(self.mjx_data)
        self.mjx_data = self._jit_forward(self.mjx_data)

    def _grasp_cube(self, cube_pos, cube_quat):
        """Establish grip using CPU MuJoCo, then transfer state to MJX."""
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        open_qpos = np.array([-0.2, -0.5, -0.3, 0.5, 0.5, 0.5, 0.5])
        for i, adr in enumerate(self.hand_qpos_adr):
            self.mj_data.qpos[adr] = open_qpos[i]
        cq = self.cube_qpos_adr
        self.mj_data.qpos[cq:cq+3] = cube_pos
        self.mj_data.qpos[cq+3:cq+7] = cube_quat
        self.mj_data.qvel[self.cube_qvel_adr:self.cube_qvel_adr+6] = 0

        # Close fingers
        for step in range(300):
            t = min(step / 200.0, 1.0)
            target = open_qpos + t * (GRIP_QPOS - open_qpos)
            self.mj_data.ctrl[:7] = target
            self.mj_data.qpos[cq:cq+3] = cube_pos
            self.mj_data.qpos[cq+3:cq+7] = cube_quat
            self.mj_data.qvel[self.cube_qvel_adr:self.cube_qvel_adr+6] = 0
            mujoco.mj_step(self.mj_model, self.mj_data)

        # Settle
        self.mj_data.ctrl[:7] = GRIP_QPOS
        for _ in range(100):
            mujoco.mj_step(self.mj_model, self.mj_data)

        # Transfer state to MJX
        from mujoco import mjx
        base_mjx = mjx.put_data(self.mj_model, self.mj_data)
        self.mjx_data = self._jit_forward(base_mjx)
        self.prev_action[:] = 0.0

    def _extract_obs(self):
        """Extract observation from MJX state."""
        jnp = self.jnp
        qpos = np.array(self.mjx_data.qpos)
        qvel = np.array(self.mjx_data.qvel)
        site_xpos = np.array(self.mjx_data.site_xpos)

        cq = self.cube_qpos_adr
        cv = self.cube_qvel_adr
        return compute_obs(
            hand_qpos=qpos[self.hand_qpos_adr],
            hand_qvel=qvel[self.hand_qvel_adr],
            cube_pos=qpos[cq:cq+3],
            cube_quat=qpos[cq+3:cq+7],
            cube_angvel=qvel[cv+3:cv+6],
            palm_pos=site_xpos[self.palm_site_id],
            tip_pos=site_xpos[self.tip_site_ids],
            target_quat=self.target_quat,
            prev_action=self.prev_action,
        )

    def _sync_render(self):
        """Copy MJX state to render model for visualization."""
        qpos = np.array(self.mjx_data.qpos)
        qvel = np.array(self.mjx_data.qvel)

        # Sync hand joints
        for i, (mjx_adr, rend_adr) in enumerate(
                zip(self.hand_qpos_adr, self.render_hand_qpos_adr)):
            self.render_data.qpos[rend_adr] = qpos[mjx_adr]

        # Sync cube
        cq_m = self.cube_qpos_adr
        cq_r = self.render_cube_qpos_adr
        self.render_data.qpos[cq_r:cq_r+7] = qpos[cq_m:cq_m+7]

        mujoco.mj_forward(self.render_model, self.render_data)

    def _get_cube_quat(self):
        qpos = np.array(self.mjx_data.qpos)
        return qpos[self.cube_qpos_adr+3:self.cube_qpos_adr+7]

    def _get_cube_pos(self):
        qpos = np.array(self.mjx_data.qpos)
        return qpos[self.cube_qpos_adr:self.cube_qpos_adr+3]

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
            self._sync_render()
            current = detect_face_np(self._get_cube_quat())
            self.target_quat = FACE_TARGET_QUATS[current - 1].copy()
            self.target_face = None
            self.achieved = False
            self.holding = True
            print(f"  Reset — face {current} on top. Press 1-6 to set target.")

        elif key == 80:  # P
            self.paused = not self.paused
            print(f"  {'Paused' if self.paused else 'Resumed'}")

    def run(self):
        jnp = self.jnp

        # JIT warmup
        print("JIT compiling MJX physics (one-time, ~30-60s)...")
        cube_pos = np.array([0.09, 0.0, 0.22])
        cube_quat = FACE_TARGET_QUATS[0].copy()
        self._grasp_cube(cube_pos, cube_quat)

        # JIT warmup: single step + forward
        t0 = time.time()
        self.mjx_data = self._jit_step(self.mjx_data)
        self.jax.block_until_ready(self.mjx_data.qpos)
        print(f"  Step JIT: {time.time()-t0:.1f}s")
        t0 = time.time()
        self.mjx_data = self._jit_forward(self.mjx_data)
        self.jax.block_until_ready(self.mjx_data.site_xpos)
        print(f"  Forward JIT: {time.time()-t0:.1f}s")

        # Re-grasp after warmup
        self._grasp_cube(cube_pos, cube_quat)
        self._sync_render()

        current = detect_face_np(self._get_cube_quat())
        self.target_quat = FACE_TARGET_QUATS[current - 1].copy()

        print()
        print("=" * 50)
        print("  Unitree Dex3-1 Dice Reorientation")
        print("=" * 50)
        print("  Press 1-6 to select target face")
        print("  Press R to reset grip")
        print("  Press P to pause/resume")
        print("=" * 50)
        print(f"  Holding cube — face {current} on top")
        print()

        with mujoco.viewer.launch_passive(
            self.render_model,
            self.render_data,
            key_callback=self.key_callback,
        ) as viewer:
            step_count = 0
            while viewer.is_running():
                if not self.paused:
                    obs = self._extract_obs()
                    action, _, _ = self.agent.select_action(obs, deterministic=True)

                    # MJX physics step (frameskip x single steps)
                    self._do_physics(action)
                    self.prev_action = action.copy()
                    step_count += 1

                    # Sync to render model
                    self._sync_render()

                    # Check goal / drop
                    cube_quat = self._get_cube_quat()
                    cube_pos = self._get_cube_pos()
                    palm_z = np.array(self.mjx_data.site_xpos)[self.palm_site_id, 2]

                    if not self.holding and not self.achieved:
                        quat_dist = 1.0 - np.abs(np.dot(cube_quat, self.target_quat))
                        if quat_dist < 0.25:
                            self.achieved = True
                            current = detect_face_np(cube_quat)
                            print(f"  Goal achieved! Face {self.target_face} on top!")

                        if cube_pos[2] < palm_z - 0.03:
                            print("  Dropped! Press R to reset.")
                            self.holding = True

                    if cube_pos[2] < palm_z - 0.03 and self.holding:
                        # Cube fell while holding — auto-reset
                        pass  # User presses R

                viewer.sync()
                time.sleep(0.02)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dex3-1 Dice Interactive Viewer")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML (default: configs/ppo_v2_runpod.yaml)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mjx_xml = os.path.join(project_root, "models", "dex3_dice_scene_mjx.xml")
    render_xml = os.path.join(project_root, "models", "dex3_dice_scene_torque.xml")
    config_path = args.config or os.path.join(
        project_root, "configs", "ppo_v2_runpod.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Use final curriculum phase overrides (matches training conditions)
    phases = config.get("curriculum", {}).get("phases", [])
    final_phase = phases[-1] if phases else {}
    action_scale = final_phase.get("action_scale", config["env"].get("action_scale", 0.3))

    agent = PPO(48, 7, config=config, device="cpu")
    agent.load(args.checkpoint)
    print(f"Loaded: {args.checkpoint}")
    print(f"action_scale={action_scale}")

    env_config = {
        "frameskip": config["env"].get("frameskip", 10),
        "action_scale": action_scale,
    }

    viewer = DexCubeViewer(agent, mjx_xml, render_xml, env_config)
    viewer.run()


if __name__ == "__main__":
    main()
