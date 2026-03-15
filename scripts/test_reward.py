"""Unit test for compute_gait_reward — run BEFORE training."""
import numpy as np
import sys; sys.path.insert(0, '.')
from envs.reward import compute_gait_reward, quat_distance


def test_basic_components():
    """Test each reward component independently."""
    config = {
        'distance_scale': 1.0, 'progress_scale': 15.0,
        'contact_bonus': 0.1, 'contact_penalty_1': -0.1,
        'gait_lift_bonus': 0.5, 'gait_replace_bonus': 0.3,
        'gait_force_threshold': 0.3, 'hold_bonus': 0.05,
        'action_smooth_coef': 0.02, 'drop_penalty': -50.0,
        'drop_height': 0.03, 'goal_threshold': 0.25,
    }
    target = np.array([1, 0, 0, 0], dtype=np.float32)  # face 1 up
    palm = np.array([0.09, 0.0, 0.197])

    # Test 1: At goal -> r_distance ~ 0, achieved_goal = True
    r, info = compute_gait_reward(
        cube_quat=target, target_quat=target, cube_pos=np.array([0.09, 0, 0.22]),
        palm_pos=palm, action=np.zeros(7), config=config,
        per_finger_contact=[True, True, True],
        prev_per_finger_contact=[True, True, True],
        prev_action=np.zeros(7), contact_forces=np.array([0.5, 0.5, 0.5]),
        cube_angvel=np.zeros(3), gait_cooldown_remaining=0)
    assert info['achieved_goal'], "Should achieve goal at target quat"
    assert abs(info['r_distance']) < 0.01, f"Distance should be ~0, got {info['r_distance']}"
    assert info['r_hold'] > 0, "Should get hold bonus (3 contacts + low angvel)"
    print("  Test 1 passed: at-goal reward correct")

    # Test 2: Drop -> r_drop = -50
    r, info = compute_gait_reward(
        cube_quat=target, target_quat=target, cube_pos=np.array([0.09, 0, 0.05]),
        palm_pos=palm, action=np.zeros(7), config=config,
        per_finger_contact=[False, False, False],
        prev_per_finger_contact=[True, True, True],
        prev_action=np.zeros(7), contact_forces=np.zeros(3),
        cube_angvel=np.zeros(3), gait_cooldown_remaining=0)
    assert info['dropped'], "Should detect drop"
    assert info['r_drop'] == -50.0, f"Drop penalty should be -50, got {info['r_drop']}"
    print("  Test 2 passed: drop penalty correct")

    # Test 3: Gait lift event -> r_gait > 0
    r, info = compute_gait_reward(
        cube_quat=target, target_quat=target, cube_pos=np.array([0.09, 0, 0.22]),
        palm_pos=palm, action=np.zeros(7), config=config,
        per_finger_contact=[False, True, True],  # thumb lifted
        prev_per_finger_contact=[True, True, True],  # was in contact
        prev_action=np.zeros(7), contact_forces=np.array([0.0, 0.5, 0.5]),
        cube_angvel=np.zeros(3), gait_cooldown_remaining=0)
    assert info['r_gait'] == 0.5, f"Gait lift bonus should be 0.5, got {info['r_gait']}"
    assert info['gait_lift_count'] == 1
    assert info['gait_event'] == True
    print("  Test 3 passed: gait lift bonus correct")

    # Test 4: Gait lift during cooldown -> r_gait = 0
    r, info = compute_gait_reward(
        cube_quat=target, target_quat=target, cube_pos=np.array([0.09, 0, 0.22]),
        palm_pos=palm, action=np.zeros(7), config=config,
        per_finger_contact=[False, True, True],
        prev_per_finger_contact=[True, True, True],
        prev_action=np.zeros(7), contact_forces=np.array([0.0, 0.5, 0.5]),
        cube_angvel=np.zeros(3), gait_cooldown_remaining=3)  # cooldown active
    assert info['r_gait'] == 0.0, f"Gait should be 0 during cooldown, got {info['r_gait']}"
    print("  Test 4 passed: gait cooldown works")

    # Test 5: Action smoothing penalty
    r1, _ = compute_gait_reward(
        cube_quat=target, target_quat=target, cube_pos=np.array([0.09, 0, 0.22]),
        palm_pos=palm, action=np.ones(7), config=config,
        per_finger_contact=[True, True, True],
        prev_per_finger_contact=[True, True, True],
        prev_action=np.zeros(7),  # big action change
        contact_forces=np.array([0.5, 0.5, 0.5]),
        cube_angvel=np.zeros(3), gait_cooldown_remaining=0)
    r2, _ = compute_gait_reward(
        cube_quat=target, target_quat=target, cube_pos=np.array([0.09, 0, 0.22]),
        palm_pos=palm, action=np.ones(7), config=config,
        per_finger_contact=[True, True, True],
        prev_per_finger_contact=[True, True, True],
        prev_action=np.ones(7),  # no action change
        contact_forces=np.array([0.5, 0.5, 0.5]),
        cube_angvel=np.zeros(3), gait_cooldown_remaining=0)
    assert r2 > r1, f"Smooth action should score higher: {r2} vs {r1}"
    print("  Test 5 passed: action smoothing penalizes jitter")

    # Test 6: Progress reward
    far_quat = np.array([0.707, 0.707, 0, 0], dtype=np.float32)  # 90 deg away
    prev_dist = quat_distance(far_quat, target)
    closer_quat = np.array([0.924, 0.383, 0, 0], dtype=np.float32)  # 45 deg away
    r, info = compute_gait_reward(
        cube_quat=closer_quat, target_quat=target, cube_pos=np.array([0.09, 0, 0.22]),
        palm_pos=palm, action=np.zeros(7), config=config, prev_quat_dist=prev_dist,
        per_finger_contact=[True, True, True],
        prev_per_finger_contact=[True, True, True],
        prev_action=np.zeros(7), contact_forces=np.array([0.5, 0.5, 0.5]),
        cube_angvel=np.zeros(3), gait_cooldown_remaining=0)
    assert info['r_progress'] > 0, f"Progress should be positive, got {info['r_progress']}"
    print(f"  Test 6 passed: progress reward = {info['r_progress']:.3f}")

    # Test 7: Gait replace event
    r, info = compute_gait_reward(
        cube_quat=target, target_quat=target, cube_pos=np.array([0.09, 0, 0.22]),
        palm_pos=palm, action=np.zeros(7), config=config,
        per_finger_contact=[True, True, True],  # thumb regained contact
        prev_per_finger_contact=[False, True, True],  # was lifted
        prev_action=np.zeros(7), contact_forces=np.array([0.5, 0.5, 0.5]),
        cube_angvel=np.zeros(3), gait_cooldown_remaining=0)
    assert info['r_gait'] == 0.3, f"Gait replace bonus should be 0.3, got {info['r_gait']}"
    assert info['gait_replace_count'] == 1
    print("  Test 7 passed: gait replace bonus correct")

    # Test 8: Lift with weak holding force -> no bonus
    r, info = compute_gait_reward(
        cube_quat=target, target_quat=target, cube_pos=np.array([0.09, 0, 0.22]),
        palm_pos=palm, action=np.zeros(7), config=config,
        per_finger_contact=[False, True, True],
        prev_per_finger_contact=[True, True, True],
        prev_action=np.zeros(7), contact_forces=np.array([0.0, 0.1, 0.1]),  # weak
        cube_angvel=np.zeros(3), gait_cooldown_remaining=0)
    assert info['r_gait'] == 0.0, f"Weak force should prevent gait bonus, got {info['r_gait']}"
    print("  Test 8 passed: weak holding force prevents gait bonus")

    print("\n=== ALL REWARD UNIT TESTS PASSED ===")


if __name__ == "__main__":
    test_basic_components()
