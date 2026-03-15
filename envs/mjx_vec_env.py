"""GPU-accelerated vectorized environment using MuJoCo MJX.

Runs 1024+ environments in parallel on GPU for massive speedup.
Pre-computes valid grip states on CPU, then runs everything on GPU.

v3 architecture: residual position control, obs_dim=48, distance+progress+contact reward.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import mujoco
from mujoco import mjx


# Face normals in dice local frame (matching face_detector.py)
FACE_NORMALS_NP = np.array([
    [0.0, 0.0, 1.0],   # face 1: +Z
    [0.0, 1.0, 0.0],   # face 2: +Y
    [1.0, 0.0, 0.0],   # face 3: +X
    [-1.0, 0.0, 0.0],  # face 4: -X
    [0.0, -1.0, 0.0],  # face 5: -Y
    [0.0, 0.0, -1.0],  # face 6: -Z
])

# Target quaternions [w,x,y,z] for each face pointing up
FACE_TARGET_QUATS_NP = np.array([
    [1.0, 0.0, 0.0, 0.0],              # face 1
    [0.7071068, 0.7071068, 0.0, 0.0],   # face 2
    [0.7071068, 0.0, -0.7071068, 0.0],  # face 3
    [0.7071068, 0.0, 0.7071068, 0.0],   # face 4
    [0.7071068, -0.7071068, 0.0, 0.0],  # face 5
    [0.0, 1.0, 0.0, 0.0],              # face 6
])

# Grip qpos for v3 residual position control
GRIP_QPOS_NP = np.array([-0.419, -0.339, -1.047, 1.100, 1.222, 1.100, 1.222])

# Joint limits (ctrl_lo / ctrl_hi)
CTRL_LO_NP = np.array([-1.0472, -1.0472, -1.74533, 0.0, 0.0, 0.0, 0.0])
CTRL_HI_NP = np.array([1.0472, 0.724312, 0.0, 1.5708, 1.74533, 1.5708, 1.74533])


def domain_randomize(mj_model, mjx_model, rng, n_envs, config=None):
    """Create per-env randomized physics parameters.

    Follows MuJoCo Playground's pattern (leap_hand/reorient.py).
    Returns (model, in_axes) where model has batched randomized fields
    and in_axes tells jax.vmap which fields are per-env (0) vs shared (None).
    """
    config = config or {}

    # Extract IDs from MuJoCo model (Python-side, not JAX)
    fingertip_geom_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, n)
        for n in ["thumb_tip_col", "middle_tip_col", "index_tip_col"]
    ]
    cube_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core")
    cube_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "cube")

    joint_names = ["thumb_0", "thumb_1", "thumb_2",
                   "middle_0", "middle_1", "index_0", "index_1"]
    hand_joint_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)
        for n in joint_names
    ]
    hand_dof_adr = np.array([mj_model.jnt_dofadr[j] for j in hand_joint_ids])

    hand_body_names = [
        "right_wrist_yaw_link",
        "right_hand_thumb_0_link", "right_hand_thumb_1_link",
        "right_hand_thumb_2_link",
        "right_hand_middle_0_link", "right_hand_middle_1_link",
        "right_hand_index_0_link", "right_hand_index_1_link",
    ]
    hand_body_ids = np.array([
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, n)
        for n in hand_body_names
    ])

    # Get ranges from config
    ft_fric = config.get("fingertip_friction_range", [0.5, 1.0])
    cube_fric = config.get("cube_friction_range", [0.4, 1.2])
    mass_sc = config.get("mass_scale", [0.8, 1.2])
    kp_sc = config.get("kp_scale", [0.8, 1.2])
    damp_sc = config.get("damping_scale", [0.8, 1.2])
    fl_sc = config.get("frictionloss_scale", [0.5, 2.0])
    arm_sc = config.get("armature_scale", [1.0, 1.05])
    link_mass_sc = config.get("link_mass_scale", [0.9, 1.1])

    model = mjx_model

    @jax.vmap
    def rand(rng):
        # Fingertip friction: =U(range)
        rng, key = jax.random.split(rng)
        ft_friction = jax.random.uniform(key, (1,),
                                         minval=ft_fric[0], maxval=ft_fric[1])
        geom_friction = model.geom_friction.at[fingertip_geom_ids, 0].set(
            ft_friction
        )

        # Cube friction: =U(range)
        rng, key = jax.random.split(rng)
        cf = jax.random.uniform(key, (1,),
                                minval=cube_fric[0], maxval=cube_fric[1])
        geom_friction = geom_friction.at[cube_geom_id, 0].set(cf[0])

        # Cube inertia: *U(range)
        rng, key1, key2 = jax.random.split(rng, 3)
        dmass = jax.random.uniform(key1, minval=mass_sc[0], maxval=mass_sc[1])
        body_inertia = model.body_inertia.at[cube_body_id].set(
            model.body_inertia[cube_body_id] * dmass
        )
        # Inertia position offset ±5mm
        dpos = jax.random.uniform(key2, (3,), minval=-5e-3, maxval=5e-3)
        body_ipos = model.body_ipos.at[cube_body_id].set(
            model.body_ipos[cube_body_id] + dpos
        )

        # Joint stiffness (kp): *U(range) — 7 actuators
        rng, key = jax.random.split(rng)
        kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
            key, (model.nu,), minval=kp_sc[0], maxval=kp_sc[1]
        )
        actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
        actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)

        # Joint damping: *U(range)
        rng, key = jax.random.split(rng)
        kd = model.dof_damping[hand_dof_adr] * jax.random.uniform(
            key, (7,), minval=damp_sc[0], maxval=damp_sc[1]
        )
        dof_damping = model.dof_damping.at[hand_dof_adr].set(kd)

        # Joint frictionloss: *U(range)
        rng, key = jax.random.split(rng)
        fl = model.dof_frictionloss[hand_dof_adr] * jax.random.uniform(
            key, (7,), minval=fl_sc[0], maxval=fl_sc[1]
        )
        dof_frictionloss = model.dof_frictionloss.at[hand_dof_adr].set(fl)

        # Joint armature: *U(range)
        rng, key = jax.random.split(rng)
        arm = model.dof_armature[hand_dof_adr] * jax.random.uniform(
            key, (7,), minval=arm_sc[0], maxval=arm_sc[1]
        )
        dof_armature = model.dof_armature.at[hand_dof_adr].set(arm)

        # Link masses: *U(range)
        rng, key = jax.random.split(rng)
        dm = jax.random.uniform(
            key, (len(hand_body_ids),),
            minval=link_mass_sc[0], maxval=link_mass_sc[1]
        )
        body_mass = model.body_mass.at[hand_body_ids].set(
            model.body_mass[hand_body_ids] * dm
        )

        return (geom_friction, body_mass, body_inertia, body_ipos,
                dof_frictionloss, dof_armature, dof_damping,
                actuator_gainprm, actuator_biasprm)

    rngs = jax.random.split(rng, n_envs)
    (geom_friction, body_mass, body_inertia, body_ipos,
     dof_frictionloss, dof_armature, dof_damping,
     actuator_gainprm, actuator_biasprm) = rand(rngs)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({
        "geom_friction": 0,
        "body_mass": 0,
        "body_inertia": 0,
        "body_ipos": 0,
        "dof_frictionloss": 0,
        "dof_armature": 0,
        "dof_damping": 0,
        "actuator_gainprm": 0,
        "actuator_biasprm": 0,
    })

    model = model.tree_replace({
        "geom_friction": geom_friction,
        "body_mass": body_mass,
        "body_inertia": body_inertia,
        "body_ipos": body_ipos,
        "dof_frictionloss": dof_frictionloss,
        "dof_armature": dof_armature,
        "dof_damping": dof_damping,
        "actuator_gainprm": actuator_gainprm,
        "actuator_biasprm": actuator_biasprm,
    })

    return model, in_axes


class MJXVecEnv:
    """GPU-vectorized environment using MJX (v3 architecture)."""

    def __init__(self, xml_path, n_envs=1024, config=None, curriculum_phases=None):
        self.n_envs = n_envs
        self.config = config or {}
        self.config["xml_path_abs"] = xml_path  # Store for precompute
        self.frameskip = self.config.get("frameskip", 10)
        self.action_scale = self.config.get("action_scale", 0.3)
        self.max_episode_steps = self.config.get("max_episode_steps", 50)
        reward_cfg = self.config.get("reward", {})

        # Load MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mjx_model = mjx.put_model(self.mj_model)

        # Cache joint/body/site IDs
        self.n_hand_joints = 7
        joint_names = ["thumb_0", "thumb_1", "thumb_2",
                       "middle_0", "middle_1", "index_0", "index_1"]
        actuator_names = ["thumb_0_act", "thumb_1_act", "thumb_2_act",
                          "middle_0_act", "middle_1_act",
                          "index_0_act", "index_1_act"]

        self.hand_joint_ids = [
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in joint_names
        ]
        self.hand_qpos_adr = np.array([self.mj_model.jnt_qposadr[j] for j in self.hand_joint_ids])
        self.hand_qvel_adr = np.array([self.mj_model.jnt_dofadr[j] for j in self.hand_joint_ids])

        cube_joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self.cube_qpos_adr = int(self.mj_model.jnt_qposadr[cube_joint_id])
        self.cube_qvel_adr = int(self.mj_model.jnt_dofadr[cube_joint_id])

        self.tip_site_ids = np.array([
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, n)
            for n in ["thumb_tip_site", "middle_tip_site", "index_tip_site"]
        ])
        self.palm_site_id = int(mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "palm_site"
        ))

        # Actuator control ranges (read from model for reference, but we use
        # the explicit CTRL_LO/CTRL_HI constants for the v3 joint limits)
        self.ctrl_ranges = np.zeros((7, 2))
        for i, name in enumerate(actuator_names):
            act_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.ctrl_ranges[i] = self.mj_model.actuator_ctrlrange[act_id]

        # Reward params (v3 + gait)
        self.reward_params = {
            "distance_scale": float(reward_cfg.get("distance_scale", 1.0)),
            "progress_scale": float(reward_cfg.get("progress_scale", 15.0)),
            "contact_bonus": float(reward_cfg.get("contact_bonus", 0.1)),
            "contact_penalty_1": float(reward_cfg.get("contact_penalty_1", -0.1)),
            "goal_threshold": float(reward_cfg.get("goal_threshold", 0.25)),
            "drop_penalty": float(reward_cfg.get("drop_penalty", -50.0)),
            "drop_height": float(reward_cfg.get("drop_height", 0.03)),
            "goal_bonus": float(reward_cfg.get("goal_bonus", 100.0)),
            "gait_lift_bonus": float(reward_cfg.get("gait_lift_bonus", 0.0)),
            "gait_replace_bonus": float(reward_cfg.get("gait_replace_bonus", 0.0)),
            "gait_force_threshold": float(reward_cfg.get("gait_force_threshold", 0.3)),
            "hold_bonus": float(reward_cfg.get("hold_bonus", 0.05)),
            "action_smooth_coef": float(reward_cfg.get("action_smooth_coef", 0.02)),
        }

        # Convert constants to JAX
        self.jax_hand_qpos_adr = jnp.array(self.hand_qpos_adr, dtype=jnp.int32)
        self.jax_hand_qvel_adr = jnp.array(self.hand_qvel_adr, dtype=jnp.int32)
        self.jax_tip_site_ids = jnp.array(self.tip_site_ids, dtype=jnp.int32)
        self.jax_ctrl_lo = jnp.array(CTRL_LO_NP)
        self.jax_ctrl_hi = jnp.array(CTRL_HI_NP)
        self.jax_grip_qpos = jnp.array(GRIP_QPOS_NP)
        self.jax_face_normals = jnp.array(FACE_NORMALS_NP)
        self.jax_face_target_quats = jnp.array(FACE_TARGET_QUATS_NP)

        # Obs/action dims
        self.obs_dim = 48
        self.act_dim = 7

        # Domain randomization state
        self._dr_enabled = False
        self._dr_model = None
        self._dr_in_axes = None

        # Curriculum state: which faces to sample on reset
        # Default: all faces allowed for both start and target
        self._target_face_indices = jnp.arange(6, dtype=jnp.int32)  # 0-5 (face 1-6)
        self._target_faces_set = set(range(1, 7))
        self._start_faces_set = set(range(1, 7))

        # Curriculum max_angle state
        self._current_max_angle = 3.15
        self._current_max_level = None  # set after precompute
        self._use_level_filter = False  # True when max_angle < 2.5

        # Pre-compute valid initial states at multiple difficulty levels
        print("Pre-computing valid grip states (multi-level)...")
        (self.init_qpos_bank, self.init_qvel_bank,
         self._init_target_faces, self._init_start_faces,
         self._init_levels, self._level_angles) = \
            self._precompute_init_states(curriculum_phases=curriculum_phases)
        self._current_max_level = len(self._level_angles) - 1
        print(f"  {len(self.init_qpos_bank)} valid initial states across "
              f"{len(self._level_angles)} difficulty levels")

        # Build init state pool
        self._update_pool()

        # Prev actions state (for obs and step_envs)
        self._prev_actions = jnp.zeros((n_envs, 7))

        # JIT compile physics and forward pass separately (hybrid approach)
        print("JIT compiling MJX functions...")
        self._jit_forward = jax.jit(jax.vmap(
            partial(mjx.forward, self.mjx_model)
        ))

        # Physics step: action → fori_loop of mjx.step
        mjx_m = self.mjx_model
        fs = self.frameskip
        grip = self.jax_grip_qpos
        clo = self.jax_ctrl_lo
        chi = self.jax_ctrl_hi

        def _single_physics(data, action, action_scale):
            action = jnp.clip(action, -1.0, 1.0)
            tgt = grip + action * action_scale
            tgt = jnp.clip(tgt, clo, chi)
            data = data.replace(ctrl=data.ctrl.at[:7].set(tgt))
            def body(_, d):
                return mjx.step(mjx_m, d)
            return jax.lax.fori_loop(0, fs, body, data)

        self._jit_physics = jax.jit(jax.vmap(_single_physics))

    def _precompute_init_states(self, n_per_combo=15, curriculum_phases=None):
        """Pre-compute valid grip states at multiple difficulty levels.

        For each unique max_angle in the curriculum, generates init states
        where the cube starts at that distance from the target orientation.
        This enables proper graduated curriculum in MJX.

        Returns:
            qpos_bank, qvel_bank: JAX arrays of physical states
            target_faces: numpy array — which face was the TARGET during precompute
            start_faces: numpy array — which face is actually on TOP after settle
            levels: numpy array — which max_angle level (0 = easiest)
            level_angles: sorted list of max_angle values
        """
        from envs.dex_cube_env import DexCubeEnv
        from perception.face_detector import detect_top_face

        xml_path = self.config.get("xml_path_abs")
        env = DexCubeEnv(xml_path=xml_path, config=self.config)

        # Extract unique max_angle values from curriculum phases
        angles = set()
        if curriculum_phases:
            for p in curriculum_phases:
                angles.add(p.get("max_angle", 3.15))
        angles.add(3.15)  # always include random starts
        sorted_angles = sorted(angles)

        all_qpos, all_qvel = [], []
        all_target_faces, all_start_faces, all_levels = [], [], []

        for level_idx, max_angle in enumerate(sorted_angles):
            env.set_curriculum_max_angle(max_angle)
            level_count = 0
            for target_face in range(1, 7):
                successes = 0
                for _ in range(n_per_combo * 3):
                    if successes >= n_per_combo:
                        break
                    env.reset(target_face=target_face)
                    cube_z = env._get_cube_pos()[2]
                    palm_z = env.data.site_xpos[env.palm_site_id][2]
                    if cube_z > palm_z - 0.05:
                        cube_quat = env._get_cube_quat()
                        start_face = detect_top_face(cube_quat)

                        all_qpos.append(env.data.qpos.copy())
                        all_qvel.append(env.data.qvel.copy())
                        all_target_faces.append(target_face)
                        all_start_faces.append(start_face)
                        all_levels.append(level_idx)
                        successes += 1
                        level_count += 1
            print(f"  Level {level_idx} (max_angle={max_angle:.2f}): "
                  f"{level_count} states")

        qpos_bank = jnp.array(np.stack(all_qpos))
        qvel_bank = jnp.array(np.stack(all_qvel))
        target_faces = np.array(all_target_faces, dtype=np.int32)
        start_faces = np.array(all_start_faces, dtype=np.int32)
        levels = np.array(all_levels, dtype=np.int32)

        return qpos_bank, qvel_bank, target_faces, start_faces, levels, sorted_angles

    def set_curriculum_max_angle(self, max_angle):
        """Set the max angular distance from target for starting orientations.

        For max_angle < 2.5: filters init states by precomputed difficulty level.
        For max_angle >= 2.5: uses start_faces filtering instead.
        """
        self._current_max_angle = max_angle
        if max_angle < 2.5:
            # Find the highest level whose angle <= max_angle
            level = 0
            for i, a in enumerate(self._level_angles):
                if a <= max_angle + 0.01:
                    level = i
            self._current_max_level = level
            self._use_level_filter = True
        else:
            self._current_max_level = len(self._level_angles) - 1
            self._use_level_filter = False
        self._update_pool()

    def _update_pool(self):
        """Rebuild init state pool based on current max_angle, target_faces, start_faces."""
        n_states = len(self.init_qpos_bank)
        valid = np.ones(n_states, dtype=bool)

        if self._use_level_filter:
            # Low max_angle: use states at matching levels, for matching target faces.
            # Include all levels up to current (easier + current difficulty).
            valid &= (self._init_levels <= self._current_max_level)
            valid &= np.isin(self._init_target_faces, list(self._target_faces_set))
        else:
            # High max_angle: use only max-level (random) states,
            # filtered by actual starting face orientation.
            max_level = len(self._level_angles) - 1
            valid &= (self._init_levels == max_level)
            valid &= np.isin(self._init_start_faces, list(self._start_faces_set))

        indices = np.where(valid)[0]
        if len(indices) == 0:
            print(f"  WARNING: No matching init states (level_filter={self._use_level_filter}, "
                  f"max_level={self._current_max_level}, targets={self._target_faces_set}, "
                  f"starts={self._start_faces_set}), using all {n_states}")
            indices = np.arange(n_states)

        self._start_pool = jnp.array(indices, dtype=jnp.int32)
        print(f"  Init pool: {len(indices)} states")

    def apply_domain_randomization(self, rng, dr_config):
        """Randomize physics params per-env and rebuild JIT functions.

        Args:
            rng: JAX random key
            dr_config: dict with randomization ranges
        """
        self._dr_model, self._dr_in_axes = domain_randomize(
            self.mj_model, self.mjx_model, rng, self.n_envs, dr_config
        )
        self._dr_enabled = True

        # Rebuild JIT physics with per-env model (model as first arg)
        fs = self.frameskip
        grip = self.jax_grip_qpos
        clo = self.jax_ctrl_lo
        chi = self.jax_ctrl_hi

        def _single_physics_dr(model, data, action, action_scale):
            action = jnp.clip(action, -1.0, 1.0)
            tgt = grip + action * action_scale
            tgt = jnp.clip(tgt, clo, chi)
            data = data.replace(ctrl=data.ctrl.at[:7].set(tgt))
            def body(_, d):
                return mjx.step(model, d)
            return jax.lax.fori_loop(0, fs, body, data)

        self._jit_physics_dr = jax.jit(
            jax.vmap(_single_physics_dr,
                     in_axes=(self._dr_in_axes, 0, 0, 0))
        )
        self._jit_forward_dr = jax.jit(
            jax.vmap(mjx.forward, in_axes=(self._dr_in_axes, 0))
        )

        print(f"Domain randomization enabled: {self.n_envs} envs with randomized physics")

    def set_target_faces(self, face_indices):
        """Set which target faces are sampled on reset.

        Args:
            face_indices: list of face numbers (1-6), e.g. [1, 6]
        """
        self._target_face_indices = jnp.array(
            [f - 1 for f in face_indices], dtype=jnp.int32
        )
        self._target_faces_set = set(face_indices)
        self._update_pool()

    def set_start_faces(self, face_indices):
        """Set which starting orientations are used on reset.

        Args:
            face_indices: list of face numbers (1-6), e.g. [2, 3, 4, 5]
        """
        self._start_faces_set = set(face_indices)
        self._update_pool()

    def set_reward_config(self, overrides):
        """Update reward params (for per-phase reward overrides)."""
        for k, v in overrides.items():
            if k in self.reward_params:
                self.reward_params[k] = float(v)

    def physics_step(self, mjx_data, actions):
        """Run physics on GPU. Returns updated mjx_data.

        GPU-only, ~71ms at 2048 envs. Actions in [-1, 1].
        Uses per-env randomized model when domain randomization is enabled.
        """
        action_scales = jnp.broadcast_to(
            jnp.float32(self.action_scale), (self.n_envs,)
        )
        if self._dr_enabled:
            return self._jit_physics_dr(
                self._dr_model, mjx_data, actions, action_scales)
        return self._jit_physics(mjx_data, actions, action_scales)

    def forward_step(self, mjx_data):
        """Run forward kinematics on GPU. Updates site_xpos etc. ~8ms."""
        if self._dr_enabled:
            return self._jit_forward_dr(self._dr_model, mjx_data)
        return self._jit_forward(mjx_data)

    def extract_state_np(self, mjx_data):
        """Transfer relevant state from GPU to CPU numpy arrays.

        Returns dict with: hand_qpos, hand_qvel, cube_pos, cube_quat,
        cube_angvel, palm_pos, tip_pos, qpos, qvel, site_xpos
        """
        cq = self.cube_qpos_adr
        cv = self.cube_qvel_adr
        qpos = np.array(mjx_data.qpos)
        qvel = np.array(mjx_data.qvel)
        site_xpos = np.array(mjx_data.site_xpos)

        return {
            "hand_qpos": qpos[:, self.hand_qpos_adr],
            "hand_qvel": qvel[:, self.hand_qvel_adr],
            "cube_pos": qpos[:, cq:cq+3],
            "cube_quat": qpos[:, cq+3:cq+7],
            "cube_angvel": qvel[:, cv+3:cv+6],
            "palm_pos": site_xpos[:, self.palm_site_id],
            "tip_pos": site_xpos[:, self.tip_site_ids],  # (n, 3, 3)
            "qpos": qpos,
            "qvel": qvel,
        }

    def compute_obs_np(self, state, target_quats_np, prev_actions_np,
                       obs_noise=None):
        """Compute 48-dim obs in numpy from extracted state.

        Args:
            obs_noise: optional dict with keys 'level', 'joint_pos',
                       'cube_pos', 'cube_ori'. When provided, adds noise
                       to observations for domain randomization robustness.
        """
        n = state["hand_qpos"].shape[0]

        # Apply observation noise (training only, not eval)
        hand_qpos = state["hand_qpos"]
        cube_pos = state["cube_pos"]
        cube_quat = state["cube_quat"]
        if obs_noise is not None:
            level = obs_noise.get("level", 1.0)
            hand_qpos = hand_qpos + (
                np.random.uniform(-1, 1, size=hand_qpos.shape)
                * obs_noise.get("joint_pos", 0.05) * level
            )
            cube_pos = cube_pos + (
                np.random.uniform(-1, 1, size=cube_pos.shape)
                * obs_noise.get("cube_pos", 0.02) * level
            )
            cube_quat = cube_quat + (
                np.random.normal(0, 1, size=cube_quat.shape)
                * obs_noise.get("cube_ori", 0.1) * level
            )
            cube_quat = cube_quat / np.linalg.norm(
                cube_quat, axis=-1, keepdims=True)

        rel_cube = cube_pos - state["palm_pos"]
        tip_flat = state["tip_pos"].reshape(n, 9)

        # Face detection (vectorized numpy — no Python loops)
        q = cube_quat  # (n, 4)
        w = q[:, 0]             # (n,)
        xyz = q[:, 1:4]         # (n, 3)
        normals = FACE_NORMALS_NP  # (6, 3)
        # Broadcast: xyz->(n,1,3), normals->(1,6,3)
        xyz_exp = xyz[:, None, :]       # (n, 1, 3)
        n_exp = normals[None, :, :]     # (1, 6, 3)
        t = 2.0 * np.cross(xyz_exp, n_exp)  # (n, 6, 3)
        w_exp = w[:, None, None]             # (n, 1, 1)
        world_normals = n_exp + w_exp * t + np.cross(xyz_exp, t)  # (n, 6, 3)
        dots = world_normals[:, :, 2]   # (n, 6) — dot with [0,0,1] = z component
        face = ((np.argmax(dots, axis=1) + 1) / 6.0)[:, None]  # (n, 1)

        # Contact forces (distance proxy)
        tip_dists = np.linalg.norm(
            state["tip_pos"] - cube_pos[:, None, :], axis=2
        )  # (n, 3)
        contact_forces = np.exp(-10.0 * tip_dists)

        return np.concatenate([
            hand_qpos,                 # 7
            state["hand_qvel"],        # 7
            rel_cube,                  # 3
            cube_quat,                 # 4
            state["cube_angvel"],      # 3
            target_quats_np,           # 4
            tip_flat,                  # 9
            face,                      # 1
            prev_actions_np,           # 7
            contact_forces,            # 3
        ], axis=1).astype(np.float32)  # total: 48

    def compute_reward_np(self, state, prev_cube_quats_np, target_quats_np,
                          prev_per_finger_contact=None, prev_actions_np=None,
                          actions_np=None, gait_cooldown=None):
        """Compute reward with finger gaiting in vectorized numpy.

        Args:
            state: dict from extract_state_np
            prev_cube_quats_np: (n, 4)
            target_quats_np: (n, 4)
            prev_per_finger_contact: (n, 3) bool, None on first step
            prev_actions_np: (n, 7), actions from previous step
            actions_np: (n, 7), current actions (for smoothing penalty)
            gait_cooldown: (n,) int, steps remaining before gait bonus allowed

        Returns:
            rewards: (n,) float32
            infos: dict with per-env arrays
        """
        p = self.reward_params
        n = state["cube_quat"].shape[0]

        # Quat distance
        curr_dot = np.abs(np.sum(state["cube_quat"] * target_quats_np, axis=1))
        curr_dot = np.clip(curr_dot, 0.0, 1.0)
        curr_dist = 1.0 - curr_dot

        prev_dot = np.abs(np.sum(prev_cube_quats_np * target_quats_np, axis=1))
        prev_dot = np.clip(prev_dot, 0.0, 1.0)
        prev_dist = 1.0 - prev_dot

        r_distance = -curr_dist * p["distance_scale"]
        r_progress = (prev_dist - curr_dist) * p["progress_scale"]

        # Per-finger contact (distance proxy, same as obs)
        tip_dists = np.linalg.norm(
            state["tip_pos"] - state["cube_pos"][:, None, :], axis=2
        )  # (n, 3)
        per_finger_contact = tip_dists < 0.04  # (n, 3) bool
        contact_forces = np.exp(-10.0 * tip_dists)  # (n, 3)
        n_contacts = np.sum(per_finger_contact, axis=1)  # (n,)

        # Tiered contact reward
        r_contact = np.where(
            n_contacts >= 3, p["contact_bonus"],
            np.where(n_contacts >= 2, p["contact_bonus"] * 0.5,
                     p["contact_penalty_1"]))

        # Finger gaiting events (vectorized)
        r_gait = np.zeros(n, dtype=np.float32)
        gait_event = np.zeros(n, dtype=bool)
        if prev_per_finger_contact is not None and gait_cooldown is not None:
            cooldown_ok = gait_cooldown <= 0  # (n,) bool
            for i in range(3):
                others = [j for j in range(3) if j != i]
                others_in = per_finger_contact[:, others[0]] & per_finger_contact[:, others[1]]
                other_force = contact_forces[:, others[0]] + contact_forces[:, others[1]]
                force_ok = other_force > p["gait_force_threshold"]

                # Lift: was contact, now not, others holding with force
                lift = (prev_per_finger_contact[:, i] & ~per_finger_contact[:, i]
                        & others_in & force_ok & cooldown_ok)
                r_gait += np.where(lift, p["gait_lift_bonus"], 0.0)
                gait_event |= lift

                # Replace: was not contact, now contact, others holding
                replace = (~prev_per_finger_contact[:, i] & per_finger_contact[:, i]
                           & others_in & cooldown_ok)
                r_gait += np.where(replace, p["gait_replace_bonus"], 0.0)
                gait_event |= replace

        # Hold bonus (all 3 in contact + low angvel)
        r_hold = np.zeros(n, dtype=np.float32)
        all_contact = n_contacts >= 3
        angvel_mag = np.linalg.norm(state["cube_angvel"], axis=1)
        r_hold = np.where(all_contact & (angvel_mag < 0.5), p["hold_bonus"], 0.0)

        # Action smoothing
        r_smooth = np.zeros(n, dtype=np.float32)
        if prev_actions_np is not None and actions_np is not None:
            action_diff_sq = np.sum((actions_np - prev_actions_np) ** 2, axis=1)
            r_smooth = -p["action_smooth_coef"] * action_diff_sq

        # Drop
        drop_z = state["palm_pos"][:, 2] - p["drop_height"]
        dropped = state["cube_pos"][:, 2] < drop_z
        r_drop = np.where(dropped, p["drop_penalty"], 0.0)

        # Goal
        achieved = curr_dist < p["goal_threshold"]

        rewards = (r_distance + r_progress + r_contact + r_gait
                   + r_hold + r_smooth + r_drop)
        rewards += np.where(achieved, p["goal_bonus"], 0.0)

        infos = {
            "achieved_goal": achieved,
            "dropped": dropped,
            "quat_dist": curr_dist,
            "per_finger_contact": per_finger_contact,
            "gait_event": gait_event,
            "n_contacts": n_contacts,
        }
        return rewards.astype(np.float32), infos

    def reset_done_envs(self, mjx_data, done_mask_np, target_quats_np, step_counts_np, rng):
        """Reset done envs on GPU. Returns updated (mjx_data, target_quats_np, step_counts_np, rng)."""
        n_done = int(done_mask_np.sum())
        if n_done == 0:
            return mjx_data, target_quats_np, step_counts_np, rng

        rng, k1, k2 = jax.random.split(rng, 3)

        # Sample new init states for done envs
        pool = self._start_pool
        pool_size = pool.shape[0]
        picks = jax.random.randint(k1, (self.n_envs,), 0, pool_size)
        ri = pool[picks]
        r_qpos = self.init_qpos_bank[ri]
        r_qvel = self.init_qvel_bank[ri]

        # Sample new target faces
        tfi = self._target_face_indices
        n_tgt = tfi.shape[0]
        tpicks = jax.random.randint(k2, (self.n_envs,), 0, n_tgt)
        r_tgt = np.array(self.jax_face_target_quats[tfi[tpicks]])

        # Apply reset using jnp.where on GPU
        dm = jnp.array(done_mask_np)[:, None]
        mjx_data = mjx_data.replace(
            qpos=jnp.where(dm, r_qpos, mjx_data.qpos),
            qvel=jnp.where(dm, r_qvel, mjx_data.qvel),
        )

        # Update CPU arrays
        dm_np = done_mask_np[:, None]
        target_quats_np = np.where(dm_np, r_tgt, target_quats_np)
        step_counts_np[done_mask_np] = 0

        return mjx_data, target_quats_np, step_counts_np, rng

    def reset(self, rng):
        """Reset all environments.

        Args:
            rng: JAX random key

        Returns:
            mjx_data_batch: batched MJX data
            target_quats: (n_envs, 4) target quaternions
            step_counts: (n_envs,) zeros
            prev_cube_quats: (n_envs, 4)
            prev_actions: (n_envs, 7) zeros
        """
        rng, key1, key2 = jax.random.split(rng, 3)

        # Sample init states from allowed start faces
        start_pool = self._start_pool
        pool_size = start_pool.shape[0]
        pool_indices = jax.random.randint(key1, (self.n_envs,), 0, pool_size)
        state_indices = start_pool[pool_indices]

        # Sample target faces from allowed target faces
        n_targets = self._target_face_indices.shape[0]
        target_face_picks = jax.random.randint(key2, (self.n_envs,), 0, n_targets)
        target_face_indices = self._target_face_indices[target_face_picks]
        target_quats = self.jax_face_target_quats[target_face_indices]

        # Create batched initial data
        qpos_batch = self.init_qpos_bank[state_indices]
        qvel_batch = self.init_qvel_bank[state_indices]

        # Create base mjx_data
        mj_data = mujoco.MjData(self.mj_model)
        base_mjx_data = mjx.put_data(self.mj_model, mj_data)

        # Batch it
        batch_data = jax.vmap(
            lambda qp, qv: base_mjx_data.replace(qpos=qp, qvel=qv)
        )(qpos_batch, qvel_batch)

        # Run forward to update derived quantities (xpos, site_xpos, etc.)
        batch_data = self.forward_step(batch_data)

        step_counts = jnp.zeros(self.n_envs, dtype=jnp.int32)

        # Get initial cube quats
        cube_qpos_start = self.cube_qpos_adr + 3
        prev_cube_quats = qpos_batch[:, cube_qpos_start:cube_qpos_start + 4]

        # Zero previous actions
        prev_actions = jnp.zeros((self.n_envs, 7))

        return batch_data, target_quats, step_counts, prev_cube_quats, prev_actions

    def get_obs(self, mjx_data, target_quats, prev_actions):
        """Extract observations from batched MJX data.

        Returns: (n_envs, 48) observation array
        """
        # Hand joint positions and velocities
        hand_qpos = mjx_data.qpos[:, self.jax_hand_qpos_adr]  # (n, 7)
        hand_qvel = mjx_data.qvel[:, self.jax_hand_qvel_adr]  # (n, 7)

        # Cube state
        cq = self.cube_qpos_adr
        cube_pos = mjx_data.qpos[:, cq:cq+3]     # (n, 3)
        cube_quat = mjx_data.qpos[:, cq+3:cq+7]  # (n, 4)

        cv = self.cube_qvel_adr
        cube_angvel = mjx_data.qvel[:, cv+3:cv+6]  # (n, 3)

        # Palm and fingertip positions
        palm_pos = mjx_data.site_xpos[:, self.palm_site_id]  # (n, 3)
        tip_pos = mjx_data.site_xpos[:, self.jax_tip_site_ids]  # (n, 3, 3)
        tip_pos_flat = tip_pos.reshape(self.n_envs, 9)  # (n, 9)

        # Relative cube position (cube - palm)
        rel_cube_pos = cube_pos - palm_pos  # (n, 3)

        # Current face detection (vectorized)
        current_face = jax.vmap(_detect_top_face_fn)(cube_quat) / 6.0  # (n,)

        # Contact forces: distance-based proxy per finger
        tip_dists = jnp.linalg.norm(
            tip_pos - cube_pos[:, None, :], axis=2
        )  # (n, 3)
        contact_forces = jnp.exp(-10.0 * tip_dists)  # (n, 3)

        obs = jnp.concatenate([
            hand_qpos,                  # 7
            hand_qvel,                  # 7
            rel_cube_pos,               # 3
            cube_quat,                  # 4
            cube_angvel,                # 3
            target_quats,               # 4
            tip_pos_flat,               # 9
            current_face[:, None],      # 1
            prev_actions,               # 7
            contact_forces,             # 3
        ], axis=1)  # total: 48

        return obs

    @staticmethod
    def _detect_top_face(cube_quat):
        """Detect which face is on top. Returns float face number (1-6)."""
        return _detect_top_face_fn(cube_quat)

    def compute_rewards(self, mjx_data, actions, target_quats, prev_cube_quats):
        """Compute rewards for all envs. Returns (n_envs,) rewards and info dict."""
        cq = self.cube_qpos_adr
        cube_pos = mjx_data.qpos[:, cq:cq+3]
        cube_quat = mjx_data.qpos[:, cq+3:cq+7]
        palm_pos = mjx_data.site_xpos[:, self.palm_site_id]
        tip_pos = mjx_data.site_xpos[:, self.jax_tip_site_ids]  # (n, 3, 3)

        p = self.reward_params

        # Vectorized reward computation
        rewards, infos = jax.vmap(
            partial(_compute_reward_single, reward_params=p)
        )(cube_quat, prev_cube_quats, target_quats, cube_pos, palm_pos, tip_pos, actions)

        return rewards, infos

    def build_step_fn(self):
        """Return a pure JAX function for the full step (physics + reward + reset + obs).

        NOT JIT'd -- designed to be called inside jax.jit or jax.lax.scan.

        action_scale and max_episode_steps are passed as arguments to the
        returned step_fn so they can change per curriculum phase without
        triggering recompilation.
        """
        mjx_model = self.mjx_model
        n_envs = self.n_envs
        frameskip = self.frameskip  # static: does not change per phase

        init_qpos_bank = self.init_qpos_bank
        init_qvel_bank = self.init_qvel_bank
        face_target_quats = self.jax_face_target_quats

        cq = self.cube_qpos_adr
        cv = self.cube_qvel_adr
        hand_qpos_adr = self.jax_hand_qpos_adr
        hand_qvel_adr = self.jax_hand_qvel_adr
        tip_site_ids = self.jax_tip_site_ids
        palm_site_id = self.palm_site_id

        ctrl_lo = self.jax_ctrl_lo
        ctrl_hi = self.jax_ctrl_hi
        grip_qpos = self.jax_grip_qpos

        rp = self.reward_params

        def _physics_step(data, action, action_scale):
            """Single-env physics step (will be vmapped)."""
            action = jnp.clip(action, -1.0, 1.0)
            target_qpos = grip_qpos + action * action_scale
            target_qpos = jnp.clip(target_qpos, ctrl_lo, ctrl_hi)
            data = data.replace(ctrl=data.ctrl.at[:7].set(target_qpos))

            def body_fn(_, d):
                return mjx.step(mjx_model, d)
            data = jax.lax.fori_loop(0, frameskip, body_fn, data)
            return data

        vmapped_physics = jax.vmap(_physics_step)
        vmapped_forward = jax.vmap(partial(mjx.forward, mjx_model))

        def step_fn(mjx_data, actions, target_quats, prev_cube_quats,
                    prev_actions, step_counts, rng,
                    action_scale, max_episode_steps,
                    start_pool, target_face_indices):
            """Full step: physics + rewards + auto-reset + obs. Pure JAX.

            Args:
                mjx_data: batched MJX data
                actions: (n_envs, 7) actions in [-1, 1]
                target_quats: (n_envs, 4) current target quaternions
                prev_cube_quats: (n_envs, 4) cube quats from previous step
                prev_actions: (n_envs, 7) actions from previous step
                step_counts: (n_envs,) current step counts
                rng: JAX random key
                action_scale: float, radians of position offset per action unit
                max_episode_steps: int, maximum steps before timeout
                start_pool: (M,) indices into init_qpos_bank for allowed start faces
                target_face_indices: (K,) indices (0-5) of allowed target faces
            """
            # Broadcast action_scale to match vmap expectations
            action_scales = jnp.broadcast_to(action_scale, (n_envs,))

            # 1. Physics
            mjx_data = vmapped_physics(mjx_data, actions, action_scales)
            new_step_counts = step_counts + 1

            # 2. Rewards
            cube_pos = mjx_data.qpos[:, cq:cq+3]
            cube_quat = mjx_data.qpos[:, cq+3:cq+7]
            palm_pos = mjx_data.site_xpos[:, palm_site_id]
            tip_pos = mjx_data.site_xpos[:, tip_site_ids]

            rewards, infos = jax.vmap(
                partial(_compute_reward_single, reward_params=rp)
            )(cube_quat, prev_cube_quats, target_quats, cube_pos,
              palm_pos, tip_pos, actions)

            # Goal bonus (one-time, since episode terminates)
            goal_bonus = jnp.where(infos["achieved_goal"], rp["goal_bonus"], 0.0)
            rewards = rewards + goal_bonus

            # 3. Termination
            dones = jnp.logical_or(
                infos["dropped"],
                jnp.logical_or(infos["achieved_goal"],
                               new_step_counts >= max_episode_steps)
            )

            # 4. New prev quats and prev actions
            new_prev_quats = mjx_data.qpos[:, cq+3:cq+7]
            new_prev_actions = actions

            # 5. Auto-reset done envs
            rng, k1, k2 = jax.random.split(rng, 3)

            # Sample from allowed start faces
            pool_size = start_pool.shape[0]
            pool_picks = jax.random.randint(k1, (n_envs,), 0, pool_size)
            ri = start_pool[pool_picks]

            # Sample from allowed target faces
            n_targets = target_face_indices.shape[0]
            target_picks = jax.random.randint(k2, (n_envs,), 0, n_targets)
            rfi = target_face_indices[target_picks]

            r_qpos = init_qpos_bank[ri]
            r_qvel = init_qvel_bank[ri]
            r_tgt = face_target_quats[rfi]

            dm = dones[:, None]
            mjx_data = mjx_data.replace(
                qpos=jnp.where(dm, r_qpos, mjx_data.qpos),
                qvel=jnp.where(dm, r_qvel, mjx_data.qvel),
            )
            new_target_quats = jnp.where(dm, r_tgt, target_quats)
            new_step_counts = jnp.where(dones, 0, new_step_counts)
            r_pq = r_qpos[:, cq+3:cq+7]
            new_prev_quats = jnp.where(dm, r_pq, new_prev_quats)
            # Reset prev_actions to zero for newly reset envs
            new_prev_actions = jnp.where(dm, jnp.zeros_like(new_prev_actions),
                                         new_prev_actions)

            # 6. Forward kinematics (updates xpos, site_xpos, etc.)
            mjx_data = vmapped_forward(mjx_data)

            # 7. Observations (48 dims)
            h_qpos = mjx_data.qpos[:, hand_qpos_adr]
            h_qvel = mjx_data.qvel[:, hand_qvel_adr]
            c_pos = mjx_data.qpos[:, cq:cq+3]
            c_quat = mjx_data.qpos[:, cq+3:cq+7]
            c_angvel = mjx_data.qvel[:, cv+3:cv+6]
            p_pos = mjx_data.site_xpos[:, palm_site_id]
            t_pos = mjx_data.site_xpos[:, tip_site_ids]  # (n, 3, 3)
            t_pos_flat = t_pos.reshape(n_envs, 9)

            # Relative cube position
            rel_c_pos = c_pos - p_pos

            # Current face
            c_face = jax.vmap(_detect_top_face_fn)(c_quat) / 6.0

            # Contact forces (distance-based proxy)
            tip_dists = jnp.linalg.norm(
                t_pos - c_pos[:, None, :], axis=2
            )  # (n, 3)
            contact_forces = jnp.exp(-10.0 * tip_dists)  # (n, 3)

            obs = jnp.concatenate([
                h_qpos,                     # 7
                h_qvel,                     # 7
                rel_c_pos,                  # 3
                c_quat,                     # 4
                c_angvel,                   # 3
                new_target_quats,           # 4
                t_pos_flat,                 # 9
                c_face[:, None],            # 1
                new_prev_actions,           # 7
                contact_forces,             # 3
            ], axis=1)  # total: 48

            return (mjx_data, obs, rewards, dones, new_target_quats,
                    new_step_counts, new_prev_quats, new_prev_actions, infos)

        return step_fn

    def step_envs(self, mjx_data, actions, target_quats, prev_cube_quats,
                   step_counts, rng):
        """Convenience wrapper: runs one step and returns updated state + obs.

        This is the imperative API used by train_mjx.py's Python loop.
        Internally uses the pure-JAX step_fn from build_step_fn().
        JAX will retrace/recompile when start_pool/target_face_indices shapes change.

        Returns:
            (mjx_data, obs, rewards, dones, target_quats, step_counts,
             prev_cube_quats, infos)
        """
        if not hasattr(self, "_step_fn_raw"):
            self._step_fn_raw = self.build_step_fn()
            self._step_fn = jax.jit(self._step_fn_raw)

        action_scale = jnp.float32(self.action_scale)
        max_steps = jnp.int32(self.max_episode_steps)
        start_pool = self._start_pool
        target_face_indices = self._target_face_indices

        (mjx_data, obs, rewards, dones, target_quats,
         step_counts, prev_cube_quats, prev_actions, infos) = self._step_fn(
            mjx_data, actions, target_quats, prev_cube_quats,
            self._prev_actions, step_counts, rng,
            action_scale, max_steps, start_pool, target_face_indices
        )

        self._prev_actions = prev_actions
        return (mjx_data, obs, rewards, dones, target_quats,
                step_counts, prev_cube_quats, infos)

    def build_get_obs_fn(self):
        """Return a pure JAX function for getting observations (48 dims)."""
        cq = self.cube_qpos_adr
        cv = self.cube_qvel_adr
        hand_qpos_adr = self.jax_hand_qpos_adr
        hand_qvel_adr = self.jax_hand_qvel_adr
        tip_site_ids = self.jax_tip_site_ids
        palm_site_id = self.palm_site_id
        n_envs = self.n_envs

        def get_obs_fn(mjx_data, target_quats, prev_actions):
            h_qpos = mjx_data.qpos[:, hand_qpos_adr]
            h_qvel = mjx_data.qvel[:, hand_qvel_adr]
            c_pos = mjx_data.qpos[:, cq:cq+3]
            c_quat = mjx_data.qpos[:, cq+3:cq+7]
            c_angvel = mjx_data.qvel[:, cv+3:cv+6]
            p_pos = mjx_data.site_xpos[:, palm_site_id]
            t_pos = mjx_data.site_xpos[:, tip_site_ids]  # (n, 3, 3)
            t_pos_flat = t_pos.reshape(n_envs, 9)

            # Relative cube position
            rel_c_pos = c_pos - p_pos

            # Current face
            c_face = jax.vmap(_detect_top_face_fn)(c_quat) / 6.0

            # Contact forces (distance-based proxy)
            tip_dists = jnp.linalg.norm(
                t_pos - c_pos[:, None, :], axis=2
            )  # (n, 3)
            contact_forces = jnp.exp(-10.0 * tip_dists)  # (n, 3)

            return jnp.concatenate([
                h_qpos,                     # 7
                h_qvel,                     # 7
                rel_c_pos,                  # 3
                c_quat,                     # 4
                c_angvel,                   # 3
                target_quats,               # 4
                t_pos_flat,                 # 9
                c_face[:, None],            # 1
                prev_actions,               # 7
                contact_forces,             # 3
            ], axis=1)  # total: 48

        return get_obs_fn


def _quat_rotate(q, v):
    """Rotate vector v by quaternion q [w, x, y, z]."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    # q * v * q_conj
    t = 2.0 * jnp.cross(jnp.array([x, y, z]), v)
    return v + w * t + jnp.cross(jnp.array([x, y, z]), t)


def _detect_top_face_fn(cube_quat):
    """Standalone top-face detection for use inside JIT/scan."""
    face_normals = jnp.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])
    world_normals = jax.vmap(lambda n: _quat_rotate(cube_quat, n))(face_normals)
    up = jnp.array([0.0, 0.0, 1.0])
    dots = world_normals @ up
    return (jnp.argmax(dots) + 1).astype(jnp.float32)


def _quat_distance(q1, q2):
    """Quaternion distance: 1 - |dot(q1, q2)|."""
    dot = jnp.abs(jnp.dot(q1, q2))
    dot = jnp.clip(dot, 0.0, 1.0)
    return 1.0 - dot


def _compute_reward_single(cube_quat, prev_cube_quat, target_quat,
                           cube_pos, palm_pos, tip_pos, action,
                           reward_params):
    """Compute v3 reward for a single environment.

    Reward = distance + progress + contact + drop.
    Goal bonus is added externally in step_fn (one-time on episode termination).
    """
    p = reward_params

    curr_dist = _quat_distance(cube_quat, target_quat)
    prev_dist = _quat_distance(prev_cube_quat, target_quat)

    # Core: negative distance
    r_distance = -curr_dist * p["distance_scale"]

    # Progress: delta improvement
    r_progress = (prev_dist - curr_dist) * p["progress_scale"]

    # Contact: distance-based proxy, tiered
    tip_dists = jnp.linalg.norm(tip_pos - cube_pos[None, :], axis=1)  # (3,)
    n_contacts = jnp.sum(tip_dists < 0.04)  # slightly generous threshold
    r_contact = jnp.where(
        n_contacts >= 2, p["contact_bonus"],
        jnp.where(n_contacts >= 1, p["contact_bonus"] * 0.5, 0.0)
    )

    # Drop penalty
    drop_z = palm_pos[2] - p["drop_height"]
    dropped = cube_pos[2] < drop_z
    r_drop = jnp.where(dropped, p["drop_penalty"], 0.0)

    # Goal achieved (for info only; bonus added in step_fn)
    achieved_goal = curr_dist < p["goal_threshold"]

    total = r_distance + r_progress + r_contact + r_drop

    info = {
        "achieved_goal": achieved_goal,
        "dropped": dropped,
        "quat_dist": curr_dist,
        "n_contacts": n_contacts,
    }
    return total, info
