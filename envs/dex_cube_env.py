"""DexCubeEnv: Unitree Dex3-1 hand + dice reorientation environment.

Uses MuJoCo native position actuators (kp=5, dampratio=1) — action=0 holds the cube.
"""

import os
import numpy as np
import mujoco

from envs.reward import compute_reward, compute_gait_reward, quat_distance
from perception.face_detector import detect_top_face, get_target_quat


# Dex3-1: 7 DOF (3 thumb + 2 middle + 2 index)
JOINT_NAMES = [
    "thumb_0", "thumb_1", "thumb_2",
    "middle_0", "middle_1",
    "index_0", "index_1",
]

ACTUATOR_NAMES = [
    "thumb_0_act", "thumb_1_act", "thumb_2_act",
    "middle_0_act", "middle_1_act",
    "index_0_act", "index_1_act",
]

TIP_SITE_NAMES = ["thumb_tip_site", "middle_tip_site", "index_tip_site"]

# ALL finger body names per finger (for contact detection — not just tips)
FINGER_BODY_NAMES = [
    ["right_hand_thumb_0_link", "right_hand_thumb_1_link", "right_hand_thumb_2_link"],
    ["right_hand_middle_0_link", "right_hand_middle_1_link"],
    ["right_hand_index_0_link", "right_hand_index_1_link"],
]


class DexCubeEnv:
    """Unitree Dex3-1 dice reorientation environment.

    Observation space (48 dims):
        - hand_qpos (7): joint positions
        - hand_qvel (7): joint velocities
        - rel_cube_pos (3): cube position relative to palm
        - cube_quat (4): dice orientation quaternion
        - cube_angvel (3): dice angular velocity
        - target_quat (4): target orientation quaternion
        - fingertip_pos (9): 3 fingertip positions (3x3)
        - current_face (1): current top face / 6
        - prev_action (7): previous action
        - contact_forces (3): normal force per fingertip

    Action space (7 dims):
        - Residual position targets in [-1, 1], applied via native position actuators
    """

    def __init__(self, xml_path=None, config=None):
        if xml_path is None:
            xml_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "models", "dex3_dice_scene.xml"
            )

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.config = config or {}
        self.frameskip = self.config.get("frameskip", 25)
        self.max_episode_steps = self.config.get("max_episode_steps", 50)
        self.use_proxy_contacts = self.config.get("use_proxy_contacts", False)

        # Cache IDs
        self.n_hand_joints = 7
        self.hand_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ]
        self.hand_qpos_adr = [self.model.jnt_qposadr[jid] for jid in self.hand_joint_ids]
        self.hand_qvel_adr = [self.model.jnt_dofadr[jid] for jid in self.hand_joint_ids]

        self.cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
        )
        self.cube_qpos_adr = self.model.jnt_qposadr[self.cube_joint_id]
        self.cube_qvel_adr = self.model.jnt_dofadr[self.cube_joint_id]

        self.cube_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )

        self.tip_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in TIP_SITE_NAMES
        ]
        self.palm_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "palm_site"
        )

        self.cube_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core"
        )

        # Precompute geom IDs for ALL bodies in each finger (for contact detection)
        self._finger_geom_ids = []
        for finger_bodies in FINGER_BODY_NAMES:
            gids = set()
            for body_name in finger_bodies:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                for g in range(self.model.ngeom):
                    if self.model.geom_bodyid[g] == body_id:
                        gids.add(g)
            self._finger_geom_ids.append(gids)

        # Actuator control ranges
        self.ctrl_ranges = np.zeros((self.n_hand_joints, 2))
        for i, name in enumerate(ACTUATOR_NAMES):
            act_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            self.ctrl_ranges[i] = self.model.actuator_ctrlrange[act_id]

        # Joint limits from model
        self.joint_lo = np.array([self.model.jnt_range[jid, 0] for jid in self.hand_joint_ids])
        self.joint_hi = np.array([self.model.jnt_range[jid, 1] for jid in self.hand_joint_ids])

        # Grip configuration
        self.grip_qpos = np.array([-0.419, -0.339, -1.047, 1.100, 1.222, 1.100, 1.222])

        # Action scaling (radians of position offset per action unit)
        self.action_scale = self.config.get("action_scale", 0.3)

        # State
        self.target_face = 1
        self.target_quat = get_target_quat(1)
        self.step_count = 0
        self.prev_action = np.zeros(self.n_hand_joints)
        self.prev_cube_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.prev_quat_dist = None
        self.goal_achieved = False

        # Curriculum
        self.curriculum_max_angle = np.pi
        self.curriculum_start_faces = [1, 2, 3, 4, 5, 6]

        # Reward config
        self.reward_config = self.config.get("reward", {})

        # Finger gaiting state
        self.prev_per_finger_contact = [False, False, False]
        self.gait_cooldown_timer = 0
        self.gait_cooldown_steps = 5

        # Renderer (lazy init)
        self._renderer = None

        # Obs/action dims: 7+7+3+4+3+4+9+1+7+3 = 48
        self.obs_dim = 48
        self.act_dim = 7

    def set_target_face(self, face_num):
        """Set the target dice face to show on top."""
        assert 1 <= face_num <= 6
        self.target_face = face_num
        self.target_quat = get_target_quat(face_num)

    def set_curriculum_max_angle(self, max_angle):
        self.curriculum_max_angle = float(max_angle)

    def set_curriculum_start_faces(self, start_faces):
        self.curriculum_start_faces = list(start_faces)

    def set_action_scale(self, scale):
        self.action_scale = float(scale)

    def set_reward_config(self, overrides):
        """Update reward config with per-phase overrides."""
        self.reward_config.update(overrides)

    def reset(self, target_face=None, seed=None):
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)

        if target_face is not None:
            self.set_target_face(target_face)
        else:
            self.set_target_face(np.random.randint(1, 7))

        palm_z = 0.197
        drop_z = palm_z - 0.05

        for attempt in range(5):
            mujoco.mj_resetData(self.model, self.data)

            # Pick start orientation
            if self.curriculum_max_angle < 2.5:
                rand_quat = self._sample_start_quat(self.target_quat, self.curriculum_max_angle)
            else:
                start_face = int(np.random.choice(self.curriculum_start_faces))
                rand_quat = get_target_quat(start_face)

            cube_grasp_pos = np.array([0.09, 0.0, 0.22])
            self._reset_grip(cube_grasp_pos, rand_quat)

            cube_z = self._get_cube_pos()[2]
            if cube_z > drop_z:
                break

        self.step_count = 0
        self.prev_action = np.zeros(self.n_hand_joints)
        self.prev_cube_quat = self._get_cube_quat().copy()
        self.prev_quat_dist = quat_distance(self.prev_cube_quat, self.target_quat)
        self.goal_achieved = False

        # Init gait state from ACTUAL contact after grip settle (prevents spurious events)
        self.prev_per_finger_contact = self._get_per_finger_contact()
        self.gait_cooldown_timer = 0

        return self._get_obs()

    def _reset_grip(self, cube_pos, cube_quat):
        """Establish grip using position actuators then settle."""
        # Set initial joint positions (open hand)
        open_qpos = np.array([-0.2, -0.5, -0.3, 0.5, 0.5, 0.5, 0.5])

        for i, adr in enumerate(self.hand_qpos_adr):
            self.data.qpos[adr] = open_qpos[i]

        self.data.qpos[self.cube_qpos_adr: self.cube_qpos_adr + 3] = cube_pos
        self.data.qpos[self.cube_qpos_adr + 3: self.cube_qpos_adr + 7] = cube_quat
        self.data.qvel[self.cube_qvel_adr: self.cube_qvel_adr + 6] = 0

        # Phase 1: Close by interpolating target from open_qpos to grip_qpos
        for step in range(300):
            t = min(step / 200.0, 1.0)
            target = open_qpos + t * (self.grip_qpos - open_qpos)
            self.data.ctrl[:self.n_hand_joints] = target

            # Hold cube in place during grasp
            self.data.qpos[self.cube_qpos_adr: self.cube_qpos_adr + 3] = cube_pos
            self.data.qpos[self.cube_qpos_adr + 3: self.cube_qpos_adr + 7] = cube_quat
            self.data.qvel[self.cube_qvel_adr: self.cube_qvel_adr + 6] = 0
            mujoco.mj_step(self.model, self.data)

        # Phase 2: Release cube, hold at grip_qpos and let settle
        self.data.ctrl[:self.n_hand_joints] = self.grip_qpos
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

    def step(self, action):
        """Take action (7-dim, in [-1, 1]) via native position actuators."""
        action = np.clip(action, -1.0, 1.0)

        # Native position control: action=0 → hold at grip_qpos
        # MuJoCo position actuator internally computes: force = kp*(ctrl - qpos) - kd*qvel
        target_qpos = self.grip_qpos + action * self.action_scale
        target_qpos = np.clip(target_qpos, self.joint_lo, self.joint_hi)
        self.data.ctrl[:self.n_hand_joints] = target_qpos

        # Step physics
        for _ in range(self.frameskip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        # Get state
        cube_quat = self._get_cube_quat()
        cube_pos = self._get_cube_pos()
        palm_pos = self.data.site_xpos[self.palm_site_id].copy()

        # Per-finger contact state and forces
        per_finger_contact = self._get_per_finger_contact()
        contact_forces = self._get_contact_forces()
        cube_angvel = self.data.qvel[self.cube_qvel_adr + 3: self.cube_qvel_adr + 6].copy()

        # Decrement gait cooldown timer
        self.gait_cooldown_timer = max(0, self.gait_cooldown_timer - 1)

        # Compute reward with finger gaiting bonuses
        reward, info = compute_gait_reward(
            cube_quat=cube_quat,
            target_quat=self.target_quat,
            cube_pos=cube_pos,
            palm_pos=palm_pos,
            action=action,
            config=self.reward_config,
            prev_quat_dist=self.prev_quat_dist,
            per_finger_contact=per_finger_contact,
            prev_per_finger_contact=self.prev_per_finger_contact,
            prev_action=self.prev_action,
            contact_forces=contact_forces,
            cube_angvel=cube_angvel,
            gait_cooldown_remaining=self.gait_cooldown_timer,
        )

        # Reset cooldown on gait event
        if info["gait_event"]:
            self.gait_cooldown_timer = self.gait_cooldown_steps

        # Update gait contact state
        self.prev_per_finger_contact = list(per_finger_contact)

        # One-time goal bonus
        if info["achieved_goal"] and not self.goal_achieved:
            self.goal_achieved = True
            goal_bonus = self.reward_config.get("goal_bonus", 100.0)
            reward += goal_bonus
            info["r_goal"] = goal_bonus

        # Check termination
        done = False
        if info["dropped"]:
            done = True
        if info["achieved_goal"]:
            done = True  # Early success termination
        if self.step_count >= self.max_episode_steps:
            done = True

        info["step"] = self.step_count
        info["target_face"] = self.target_face
        info["current_face"] = self.get_current_top_face()
        info["truncated"] = self.step_count >= self.max_episode_steps and not info["dropped"] and not info["achieved_goal"]

        self.prev_action = action.copy()
        self.prev_cube_quat = cube_quat.copy()
        self.prev_quat_dist = info["quat_dist"]

        return self._get_obs(), reward, done, info

    def get_current_top_face(self):
        return detect_top_face(self._get_cube_quat())

    def _get_obs(self):
        hand_qpos = np.array([self.data.qpos[adr] for adr in self.hand_qpos_adr])
        hand_qvel = np.array([self.data.qvel[adr] for adr in self.hand_qvel_adr])
        cube_pos = self._get_cube_pos()
        cube_quat = self._get_cube_quat()
        cube_angvel = self.data.qvel[self.cube_qvel_adr + 3: self.cube_qvel_adr + 6].copy()
        target_quat = self.target_quat.copy()
        fingertip_pos = self.data.site_xpos[self.tip_site_ids].flatten()
        current_face = np.array([self.get_current_top_face() / 6.0])

        # Relative cube position (cube - palm)
        palm_pos = self.data.site_xpos[self.palm_site_id].copy()
        rel_cube_pos = cube_pos - palm_pos

        # Previous action
        prev_action = self.prev_action.copy()

        # Contact forces per fingertip
        contact_forces = self._get_contact_forces()

        obs = np.concatenate([
            hand_qpos,       # 7
            hand_qvel,       # 7
            rel_cube_pos,    # 3 (was absolute cube_pos)
            cube_quat,       # 4
            cube_angvel,     # 3
            target_quat,     # 4
            fingertip_pos,   # 9
            current_face,    # 1
            prev_action,     # 7
            contact_forces,  # 3
        ])
        return obs.astype(np.float32)

    def _get_cube_pos(self):
        return self.data.qpos[self.cube_qpos_adr: self.cube_qpos_adr + 3].copy()

    def _get_cube_quat(self):
        return self.data.qpos[self.cube_qpos_adr + 3: self.cube_qpos_adr + 7].copy()

    def _get_contact_forces(self):
        """Get contact force per fingertip (3 values).

        If use_proxy_contacts=True, uses distance-based proxy (matches MJX training).
        Otherwise uses real MuJoCo contact forces.
        """
        if self.use_proxy_contacts:
            # Distance-based proxy (matches MJX compute_obs_np)
            tip_pos = self.data.site_xpos[self.tip_site_ids]  # (3, 3)
            cube_pos = self._get_cube_pos()  # (3,)
            tip_dists = np.linalg.norm(tip_pos - cube_pos[None, :], axis=1)  # (3,)
            return np.exp(-10.0 * tip_dists).astype(np.float32)

        forces = np.zeros(3)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            for j, gids in enumerate(self._finger_geom_ids):
                if (g1 in gids and g2 == self.cube_geom_id) or \
                   (g2 in gids and g1 == self.cube_geom_id):
                    # Extract contact force
                    c_force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, c_force)
                    forces[j] += abs(c_force[0])  # Normal force magnitude
        # Normalize to reasonable range
        forces = np.clip(forces / 5.0, 0.0, 1.0)
        return forces

    def _get_per_finger_contact(self):
        """Return [bool, bool, bool] per-finger contact with cube."""
        in_contact = [False, False, False]
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            for j, gids in enumerate(self._finger_geom_ids):
                if (g1 in gids and g2 == self.cube_geom_id) or \
                   (g2 in gids and g1 == self.cube_geom_id):
                    in_contact[j] = True
        return in_contact

    def _count_finger_contacts(self):
        """Count how many fingers are in contact with the cube."""
        return sum(self._get_per_finger_contact())

    def _sample_start_quat(self, target_quat, max_angle):
        if max_angle < 0.1:
            return target_quat.copy()
        if max_angle > 2.5:
            start_face = np.random.randint(1, 7)
            return get_target_quat(start_face)
        axis = np.random.randn(3)
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        angle = np.random.uniform(0, max_angle)
        half = angle / 2.0
        rot_quat = np.array([
            np.cos(half),
            np.sin(half) * axis[0],
            np.sin(half) * axis[1],
            np.sin(half) * axis[2],
        ])
        result = self._quat_multiply(rot_quat, target_quat)
        return result / (np.linalg.norm(result) + 1e-8)

    @staticmethod
    def _quat_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    def _random_quaternion(self):
        u = np.random.uniform(0, 1, 3)
        q = np.array([
            np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]),
            np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
            np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]),
            np.sqrt(u[0]) * np.cos(2 * np.pi * u[2]),
        ])
        return q / np.linalg.norm(q)

    def render_camera(self, camera_name="top_cam", width=256, height=256):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=camera_name)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
