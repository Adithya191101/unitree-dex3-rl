"""Distance + progress reward for cube reorientation task."""

import numpy as np


def quat_distance(q1, q2):
    """Compute quaternion distance: 1 - |dot(q1, q2)|.

    Returns 0 when quaternions represent same orientation, 1 when maximally different.
    Handles quaternion double-cover (q and -q represent same rotation).
    """
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, 0.0, 1.0)
    return 1.0 - dot


def compute_reward(cube_quat, target_quat, cube_pos, palm_pos, action, config,
                   prev_quat_dist=None, contact_count=0):
    """Compute reward with progress signal for rotation.

    Components:
        1. r_distance: Negative distance to target (potential-based)
        2. r_progress: Delta improvement toward goal (directional signal)
        3. r_contact: Bonus for maintaining multi-finger contact during manipulation
        4. r_drop: Penalty for dropping

    Args:
        cube_quat: Current dice quaternion [w, x, y, z]
        target_quat: Target quaternion [w, x, y, z]
        cube_pos: Current dice position [x, y, z]
        palm_pos: Palm site position [x, y, z]
        action: Action taken (unused)
        config: Reward config dict
        prev_quat_dist: Quaternion distance from previous step (for progress)
        contact_count: Number of fingers in contact with cube
    """
    curr_dist = quat_distance(cube_quat, target_quat)

    # Core reward: negative distance to target
    r_distance = -curr_dist * config.get("distance_scale", 1.0)

    # Progress reward: positive when getting closer to goal
    r_progress = 0.0
    if prev_quat_dist is not None:
        delta = prev_quat_dist - curr_dist  # positive = getting closer
        r_progress = delta * config.get("progress_scale", 15.0)

    # Contact bonus: reward maintaining finger contact (needed for manipulation)
    # Tiered: small bonus for 1 finger, full bonus for 2+
    r_contact = 0.0
    if contact_count >= 2:
        r_contact = config.get("contact_bonus", 0.1)
    elif contact_count >= 1:
        r_contact = config.get("contact_bonus", 0.1) * 0.5

    # Drop detection
    drop_height = palm_pos[2] - config.get("drop_height", 0.03)
    dropped = cube_pos[2] < drop_height
    r_drop = config.get("drop_penalty", -50.0) if dropped else 0.0

    # Goal achieved check
    goal_threshold = config.get("goal_threshold", 0.25)
    achieved_goal = curr_dist < goal_threshold

    total = r_distance + r_progress + r_contact + r_drop

    info = {
        "r_distance": r_distance,
        "r_progress": r_progress,
        "r_contact": r_contact,
        "r_drop": r_drop,
        "quat_dist": curr_dist,
        "achieved_goal": achieved_goal,
        "dropped": dropped,
    }
    return total, info


def compute_gait_reward(cube_quat, target_quat, cube_pos, palm_pos, action, config,
                        prev_quat_dist=None, per_finger_contact=None,
                        prev_per_finger_contact=None, prev_action=None,
                        contact_forces=None, cube_angvel=None,
                        gait_cooldown_remaining=0):
    """Reward with finger gaiting bonuses and action smoothing.

    Args:
        cube_quat: (4,) current dice quaternion [w, x, y, z]
        target_quat: (4,) target quaternion
        cube_pos: (3,) current dice position
        palm_pos: (3,) palm site position
        action: (7,) current action [-1, 1]
        config: dict with reward hyperparameters
        prev_quat_dist: float, quaternion distance from previous step
        per_finger_contact: [bool, bool, bool] per-finger contact with cube
        prev_per_finger_contact: [bool, bool, bool] from previous step
        prev_action: (7,) previous action for smoothing
        contact_forces: (3,) per-finger normal force, normalized to [0, 1]
        cube_angvel: (3,) cube angular velocity
        gait_cooldown_remaining: int, steps until next gait bonus allowed

    Returns:
        total_reward: float
        info: dict with component breakdown and metadata
    """
    curr_dist = quat_distance(cube_quat, target_quat)

    # 1. Distance reward: negative distance to target (potential-based shaping)
    r_distance = -curr_dist * config.get("distance_scale", 1.0)

    # 2. Progress reward: positive delta when getting closer
    r_progress = 0.0
    if prev_quat_dist is not None:
        delta = prev_quat_dist - curr_dist  # positive = getting closer
        r_progress = delta * config.get("progress_scale", 15.0)

    # 3. Contact tiered reward
    n_contact = sum(per_finger_contact) if per_finger_contact is not None else 0
    r_contact = 0.0
    if n_contact >= 3:
        r_contact = config.get("contact_bonus", 0.1)
    elif n_contact >= 2:
        r_contact = config.get("contact_bonus", 0.1) * 0.5
    elif n_contact <= 1:
        r_contact = config.get("contact_penalty_1", -0.1)

    # 4. Finger gaiting event rewards (with cooldown)
    r_gait = 0.0
    gait_event = False
    gait_lift_count = 0
    gait_replace_count = 0
    if (prev_per_finger_contact is not None and per_finger_contact is not None
            and gait_cooldown_remaining <= 0):
        for i in range(3):
            others = [j for j in range(3) if j != i]
            # Lift event: finger i lost contact, other 2 still in contact with force
            if prev_per_finger_contact[i] and not per_finger_contact[i]:
                if all(per_finger_contact[j] for j in others):
                    other_force = sum(contact_forces[j] for j in others) if contact_forces is not None else 0
                    if other_force > config.get("gait_force_threshold", 0.3):
                        r_gait += config.get("gait_lift_bonus", 0.5)
                        gait_event = True
                        gait_lift_count += 1
            # Replace event: finger i regained contact, other 2 still in contact
            if not prev_per_finger_contact[i] and per_finger_contact[i]:
                if all(per_finger_contact[j] for j in others):
                    r_gait += config.get("gait_replace_bonus", 0.3)
                    gait_event = True
                    gait_replace_count += 1

    # 5. Stable hold bonus (all 3 contact + low angular velocity)
    r_hold = 0.0
    if per_finger_contact is not None and all(per_finger_contact):
        angvel_mag = np.linalg.norm(cube_angvel) if cube_angvel is not None else 0.0
        if angvel_mag < 0.5:
            r_hold = config.get("hold_bonus", 0.05)

    # 6. Action smoothing penalty (penalize jittery actions)
    r_smooth = 0.0
    if prev_action is not None:
        action_diff_sq = np.sum((action - prev_action) ** 2)
        r_smooth = -config.get("action_smooth_coef", 0.02) * action_diff_sq

    # 7. Drop detection
    drop_height = palm_pos[2] - config.get("drop_height", 0.03)
    dropped = bool(cube_pos[2] < drop_height)
    r_drop = config.get("drop_penalty", -50.0) if dropped else 0.0

    # Goal check
    goal_threshold = config.get("goal_threshold", 0.25)
    achieved_goal = bool(curr_dist < goal_threshold)

    total = r_distance + r_progress + r_contact + r_gait + r_hold + r_smooth + r_drop

    info = {
        "r_distance": r_distance,
        "r_progress": r_progress,
        "r_contact": r_contact,
        "r_gait": r_gait,
        "r_hold": r_hold,
        "r_smooth": r_smooth,
        "r_drop": r_drop,
        "quat_dist": curr_dist,
        "achieved_goal": achieved_goal,
        "dropped": dropped,
        "n_finger_contact": n_contact,
        "gait_event": gait_event,
        "gait_lift_count": gait_lift_count,
        "gait_replace_count": gait_replace_count,
    }
    return total, info
