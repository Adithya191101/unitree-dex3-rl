"""Scripted manipulation test — verify physics pipeline without RL.

Tests:
1. Grip stability: establish grip, verify cube doesn't drop
2. Control response: apply offsets, verify cube rotates (even slightly)
3. Face detection: reset cube to each of 6 orientations, verify detection
4. Drop detection: release grip, verify cube falls and drop is detected
5. Gaiting demo: loosen-push-regrip sequences with viewer

Usage:
    python scripts/test_manipulation.py
"""

import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perception.face_detector import (
    detect_top_face, get_target_quat, quat_to_rot_matrix, FACE_NORMALS
)

# Grip configuration (must match envs/dex_cube_env.py)
GRIP_QPOS = np.array([-0.5, -0.4, -1.2, 0.85, 0.8, 0.85, 0.8])
CUBE_START_POS = np.array([0.09, 0.0, 0.237])
CUBE_START_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # face 1 up
DROP_HEIGHT = 0.03  # below this z = dropped

JOINT_NAMES = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]


def load_model():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml_path = os.path.join(project_root, "models", "dex3_dice_scene_torque.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data


def get_ids(model):
    ids = {}
    ids["hand_joints"] = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    ids["hand_qpos_adr"] = [model.jnt_qposadr[jid] for jid in ids["hand_joints"]]
    ids["cube_joint"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    ids["cube_qpos_adr"] = model.jnt_qposadr[ids["cube_joint"]]
    ids["cube_qvel_adr"] = model.jnt_dofadr[ids["cube_joint"]]
    ids["cube_body"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    return ids


def establish_grip(model, data, ids, cube_pos=None, cube_quat=None):
    """3-phase grip: close fingers, gradual release, free settle."""
    mujoco.mj_resetData(model, data)
    open_qpos = np.zeros(7)
    adr = ids["cube_qpos_adr"]
    vadr = ids["cube_qvel_adr"]
    if cube_pos is None:
        cube_pos = CUBE_START_POS
    if cube_quat is None:
        cube_quat = CUBE_START_QUAT

    data.qpos[adr:adr+3] = cube_pos
    data.qpos[adr+3:adr+7] = cube_quat
    mujoco.mj_forward(model, data)

    # Phase 1: Close fingers (300 steps, kinematic cube hold)
    for step in range(300):
        t = min(step / 200.0, 1.0)
        data.ctrl[:7] = open_qpos + t * (GRIP_QPOS - open_qpos)
        data.qpos[adr:adr+3] = cube_pos
        data.qpos[adr+3:adr+7] = cube_quat
        data.qvel[vadr:vadr+6] = 0
        mujoco.mj_step(model, data)

    # Phase 2: Gradual release (100 steps)
    cube_weight = 0.05 * 9.81
    data.ctrl[:7] = GRIP_QPOS
    for step in range(100):
        alpha = 1.0 - step / 100.0
        data.xfrc_applied[ids["cube_body"]][2] = alpha * cube_weight
        mujoco.mj_step(model, data)
    data.xfrc_applied[ids["cube_body"]][:] = 0

    # Phase 3: Free settle (300 steps)
    data.ctrl[:7] = GRIP_QPOS
    for _ in range(300):
        mujoco.mj_step(model, data)


def get_cube_state(data, ids):
    adr = ids["cube_qpos_adr"]
    return data.qpos[adr:adr+3].copy(), data.qpos[adr+3:adr+7].copy()


def cube_held(data, ids):
    pos, _ = get_cube_state(data, ids)
    return pos[2] > DROP_HEIGHT


def smooth_ramp(t):
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0, 1)))


def step_viewer(model, data, viewer, n, ctrl=None):
    if ctrl is not None:
        data.ctrl[:7] = ctrl
    for _ in range(n):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)


def ramp_ctrl(model, data, viewer, start, end, steps):
    for step in range(steps):
        alpha = smooth_ramp(step / steps)
        data.ctrl[:7] = start + alpha * (end - start)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)


# ─── Test 1: Grip Stability ─────────────────────────────────────────────
def test_grip_stability(model, data, ids):
    """Hold cube for 1000 steps with action=0. Check it stays held."""
    print("=" * 60)
    print("TEST 1: Grip Stability")
    print("=" * 60)
    establish_grip(model, data, ids)
    pos0, _ = get_cube_state(data, ids)

    data.ctrl[:7] = GRIP_QPOS
    for _ in range(1000):
        mujoco.mj_step(model, data)

    pos1, quat1 = get_cube_state(data, ids)
    face = detect_top_face(quat1)
    z_drift = abs(pos1[2] - pos0[2])
    held = cube_held(data, ids)

    print(f"  Initial z: {pos0[2]:.4f}")
    print(f"  Final z:   {pos1[2]:.4f} (drift: {z_drift:.5f})")
    print(f"  Face: {face}, Held: {held}")
    result = "PASS" if held and z_drift < 0.01 else "FAIL"
    print(f"  Result: {result}\n")
    return result == "PASS"


# ─── Test 2: Control Response ────────────────────────────────────────────
def test_control_response(model, data, ids):
    """Apply asymmetric offsets. Verify cube quaternion changes."""
    print("=" * 60)
    print("TEST 2: Control Response (does cube move when fingers move?)")
    print("=" * 60)
    establish_grip(model, data, ids)
    _, quat_before = get_cube_state(data, ids)

    # Asymmetric push: curl index hard, loosen middle
    offset = np.array([0.0, 0.0, 0.0, -0.4, -0.4, 0.5, 0.6])
    lo = model.jnt_range[:7, 0]
    hi = model.jnt_range[:7, 1]
    target = np.clip(GRIP_QPOS + offset, lo, hi)

    for step in range(300):
        alpha = smooth_ramp(step / 300)
        data.ctrl[:7] = GRIP_QPOS + alpha * (target - GRIP_QPOS)
        mujoco.mj_step(model, data)

    data.ctrl[:7] = target
    for _ in range(200):
        mujoco.mj_step(model, data)

    _, quat_after = get_cube_state(data, ids)
    quat_diff = np.linalg.norm(quat_after - quat_before)
    w = np.clip(abs(np.dot(quat_before, quat_after)), 0, 1)
    angle_deg = np.degrees(2 * np.arccos(w))

    print(f"  Quat before: [{quat_before[0]:.3f}, {quat_before[1]:.3f}, {quat_before[2]:.3f}, {quat_before[3]:.3f}]")
    print(f"  Quat after:  [{quat_after[0]:.3f}, {quat_after[1]:.3f}, {quat_after[2]:.3f}, {quat_after[3]:.3f}]")
    print(f"  Rotation: {angle_deg:.1f} degrees")
    print(f"  Held: {cube_held(data, ids)}")
    result = "PASS" if angle_deg > 1.0 else "MARGINAL (< 1 deg rotation)"
    print(f"  Result: {result}\n")
    return "PASS" in result


# ─── Test 3: Face Detection ─────────────────────────────────────────────
def test_face_detection(model, data, ids):
    """Reset cube to each face orientation, verify detect_top_face."""
    print("=" * 60)
    print("TEST 3: Face Detection (all 6 faces)")
    print("=" * 60)
    all_pass = True
    for target_face in range(1, 7):
        target_quat = get_target_quat(target_face)
        adr = ids["cube_qpos_adr"]
        mujoco.mj_resetData(model, data)
        data.qpos[adr:adr+3] = CUBE_START_POS
        data.qpos[adr+3:adr+7] = target_quat
        mujoco.mj_forward(model, data)

        detected = detect_top_face(target_quat)
        ok = detected == target_face
        if not ok:
            all_pass = False
        print(f"  Face {target_face}: quat=[{target_quat[0]:.3f},{target_quat[1]:.3f},"
              f"{target_quat[2]:.3f},{target_quat[3]:.3f}] -> detected={detected} {'PASS' if ok else 'FAIL'}")

    result = "PASS" if all_pass else "FAIL"
    print(f"  Result: {result}\n")
    return all_pass


# ─── Test 4: Drop Detection ─────────────────────────────────────────────
def test_drop_detection(model, data, ids):
    """Open fingers + lateral push, verify cube falls off hand."""
    print("=" * 60)
    print("TEST 4: Drop Detection")
    print("=" * 60)
    establish_grip(model, data, ids)
    pos0, _ = get_cube_state(data, ids)
    print(f"  Grip z: {pos0[2]:.4f}")

    # Open all fingers
    data.ctrl[:7] = np.zeros(7)
    for _ in range(200):
        mujoco.mj_step(model, data)

    pos1, _ = get_cube_state(data, ids)
    print(f"  After release z: {pos1[2]:.4f} (rests on palm)")

    # Apply lateral impulse to knock cube off palm
    cube_body = ids["cube_body"]
    data.xfrc_applied[cube_body][0] = 2.0  # sideways push
    for _ in range(50):
        mujoco.mj_step(model, data)
    data.xfrc_applied[cube_body][:] = 0

    # Let it fall
    for _ in range(500):
        mujoco.mj_step(model, data)

    pos2, _ = get_cube_state(data, ids)
    dropped = pos2[2] < DROP_HEIGHT
    fell_from_grip = pos2[2] < pos0[2] - 0.1
    print(f"  After push z: {pos2[2]:.4f}")
    print(f"  Dropped below {DROP_HEIGHT}: {dropped}")
    print(f"  Fell >10cm from grip: {fell_from_grip}")
    result = "PASS" if dropped or fell_from_grip else "FAIL"
    print(f"  Result: {result}\n")
    return result == "PASS"


# ─── Test 5: Visual Gaiting Demo ────────────────────────────────────────
def test_gaiting_demo(model, data, ids):
    """Run gaiting maneuvers with viewer for visual inspection."""
    print("=" * 60)
    print("TEST 5: Gaiting Demo (visual)")
    print("=" * 60)

    establish_grip(model, data, ids)
    g = GRIP_QPOS.copy()
    lo = model.jnt_range[:7, 0]
    hi = model.jnt_range[:7, 1]

    maneuvers = [
        ("Loosen thumb + curl fingers", [
            np.clip(g + np.array([0.0, 0.3, 0.5, 0.0, 0.0, 0.0, 0.0]), lo, hi),
            np.clip(g + np.array([0.0, 0.3, 0.5, 0.6, 0.8, 0.6, 0.8]), lo, hi),
            g.copy(),
        ]),
        ("Loosen fingers + push thumb", [
            np.clip(g + np.array([0.0, 0.0, 0.0, -0.5, -0.5, -0.5, -0.5]), lo, hi),
            np.clip(g + np.array([-0.3, -0.4, -0.3, -0.5, -0.5, -0.5, -0.5]), lo, hi),
            g.copy(),
        ]),
        ("Asymmetric index push", [
            np.clip(g + np.array([0.0, 0.0, 0.0, -0.4, -0.4, 0.0, 0.0]), lo, hi),
            np.clip(g + np.array([0.0, 0.0, 0.0, -0.4, -0.4, 0.6, 0.8]), lo, hi),
            g.copy(),
        ]),
        ("Thumb sweep", [
            np.clip(g + np.array([0.8, 0.6, 0.5, 0.1, 0.1, 0.1, 0.1]), lo, hi),
            g.copy(),
        ]),
    ]

    print("  Launching viewer...\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_viewer(model, data, viewer, 200, GRIP_QPOS)

        for i, (name, phases) in enumerate(maneuvers):
            if not cube_held(data, ids):
                print(f"  Cube dropped! Stopping.")
                break

            _, quat_before = get_cube_state(data, ids)
            print(f"  [{i+1}/{len(maneuvers)}] {name}")

            current = data.ctrl[:7].copy()
            for target in phases:
                ramp_ctrl(model, data, viewer, current, target, 200)
                step_viewer(model, data, viewer, 100, target)
                current = target.copy()

            # Settle
            step_viewer(model, data, viewer, 150, GRIP_QPOS)

            pos, quat_after = get_cube_state(data, ids)
            face = detect_top_face(quat_after)
            w = np.clip(abs(np.dot(quat_before, quat_after)), 0, 1)
            angle = np.degrees(2 * np.arccos(w))
            print(f"    -> face={face}, rotation={angle:.1f}deg, z={pos[2]:.4f}, held={cube_held(data, ids)}")

        print("\n  Viewer open. Close window to exit.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


def main():
    model, data = load_model()
    ids = get_ids(model)

    print("Physics Pipeline Verification")
    print("Model: models/dex3_dice_scene_torque.xml")
    print(f"Timestep: {model.opt.timestep}, Actuators: kp=50, position control\n")

    # Run automated tests first (no viewer)
    results = {}
    results["grip_stability"] = test_grip_stability(model, data, ids)
    results["control_response"] = test_control_response(model, data, ids)
    results["face_detection"] = test_face_detection(model, data, ids)
    results["drop_detection"] = test_drop_detection(model, data, ids)

    print("=" * 60)
    print("AUTOMATED TEST SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    all_pass = all(results.values())
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}\n")

    # Visual demo
    test_gaiting_demo(model, data, ids)


if __name__ == "__main__":
    main()
