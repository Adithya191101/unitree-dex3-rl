"""Grip tuning script: find stable tripod grip_qpos for Dex3-1 + dice.

Tests grip configurations by:
1. Closing fingers to candidate grip_qpos
2. Releasing cube and settling
3. Checking contact normals (should be horizontal, not pushing down)
4. Running 1000 steps with action=0 to verify stability
5. Running 1000 steps with noisy actions to test robustness

Usage:
    python scripts/tune_grip.py                  # Run full sweep
    python scripts/tune_grip.py --verify          # Test current grip_qpos only
    python scripts/tune_grip.py --grip "..." --steps 2000  # Test specific grip
"""

import os
import sys
import argparse
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Joint order: thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1
# Joint ranges:
#   thumb_0:  [-1.0472, 1.0472]  (abduction, axis 0 1 0)
#   thumb_1:  [-1.0472, 0.7243]  (curl, axis 0 0 1)
#   thumb_2:  [-1.7453, 0.0]     (distal curl, axis 0 0 1)
#   middle_0: [0.0, 1.5708]      (curl, axis 0 0 1)
#   middle_1: [0.0, 1.7453]      (distal curl, axis 0 0 1)
#   index_0:  [0.0, 1.5708]      (curl, axis 0 0 1)
#   index_1:  [0.0, 1.7453]      (distal curl, axis 0 0 1)

CURRENT_GRIP = np.array([-0.419, -0.339, -1.047, 1.100, 1.222, 1.100, 1.222])

JOINT_NAMES = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]
TIP_SITE_NAMES = ["thumb_tip_site", "middle_tip_site", "index_tip_site"]
FINGER_NAMES = ["thumb", "middle", "index"]


def load_env():
    """Load MuJoCo model and data."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml_path = os.path.join(project_root, "models", "dex3_dice_scene_torque.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data


def get_ids(model):
    """Cache all relevant IDs."""
    ids = {}
    ids["hand_joints"] = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    ids["hand_qpos_adr"] = [model.jnt_qposadr[jid] for jid in ids["hand_joints"]]
    ids["tip_sites"] = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in TIP_SITE_NAMES]
    ids["palm_site"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "palm_site")
    ids["cube_joint"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    ids["cube_qpos_adr"] = model.jnt_qposadr[ids["cube_joint"]]
    ids["cube_qvel_adr"] = model.jnt_dofadr[ids["cube_joint"]]
    ids["cube_body"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    ids["cube_geom"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core")
    return ids


def reset_grip(model, data, ids, grip_qpos, cube_pos, cube_quat,
               close_steps=300, settle_steps=300):
    """Establish grip and settle. Returns True if cube held after settle."""
    mujoco.mj_resetData(model, data)
    n_joints = 7
    open_qpos = np.zeros(n_joints)

    # Set open hand
    for i, adr in enumerate(ids["hand_qpos_adr"]):
        data.qpos[adr] = open_qpos[i]

    adr = ids["cube_qpos_adr"]
    data.qpos[adr:adr+3] = cube_pos
    data.qpos[adr+3:adr+7] = cube_quat
    data.qvel[ids["cube_qvel_adr"]:ids["cube_qvel_adr"]+6] = 0

    # Phase 1: Close fingers while holding cube kinematically
    for step in range(close_steps):
        t = min(step / 200.0, 1.0)
        target = open_qpos + t * (grip_qpos - open_qpos)
        data.ctrl[:n_joints] = target
        # Hold cube in place
        data.qpos[adr:adr+3] = cube_pos
        data.qpos[adr+3:adr+7] = cube_quat
        data.qvel[ids["cube_qvel_adr"]:ids["cube_qvel_adr"]+6] = 0
        mujoco.mj_step(model, data)

    # Phase 2: Gradual release using external force
    cube_weight = 0.05 * 9.81  # mg
    for step in range(100):
        data.ctrl[:n_joints] = grip_qpos
        # Linearly reduce upward support force
        alpha = 1.0 - step / 100.0
        data.xfrc_applied[ids["cube_body"]][2] = alpha * cube_weight
        mujoco.mj_step(model, data)
    data.xfrc_applied[ids["cube_body"]][:] = 0  # Remove external force

    # Phase 3: Free settle
    data.ctrl[:n_joints] = grip_qpos
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    return True


def get_contact_info(model, data, ids):
    """Get per-finger contact normals and forces with the cube."""
    cube_geom = ids["cube_geom"]
    info = {f: {"in_contact": False, "normal": None, "force": 0.0} for f in FINGER_NAMES}

    # Build finger geom sets
    finger_body_names = [
        ["right_hand_thumb_0_link", "right_hand_thumb_1_link", "right_hand_thumb_2_link"],
        ["right_hand_middle_0_link", "right_hand_middle_1_link"],
        ["right_hand_index_0_link", "right_hand_index_1_link"],
    ]
    finger_geom_ids = []
    for bodies in finger_body_names:
        gids = set()
        for bname in bodies:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
            for g in range(model.ngeom):
                if model.geom_bodyid[g] == bid:
                    gids.add(g)
        finger_geom_ids.append(gids)

    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        for j, gids in enumerate(finger_geom_ids):
            if (g1 in gids and g2 == cube_geom) or (g2 in gids and g1 == cube_geom):
                c_force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, c_force)
                force_mag = abs(c_force[0])
                if force_mag > info[FINGER_NAMES[j]]["force"]:
                    info[FINGER_NAMES[j]]["in_contact"] = True
                    info[FINGER_NAMES[j]]["force"] = force_mag
                    # Contact normal in world frame
                    info[FINGER_NAMES[j]]["normal"] = contact.frame[:3].copy()

    return info


def test_grip(model, data, ids, grip_qpos, cube_quat=None, hold_steps=1000,
              noise_steps=1000, noise_std=0.3, verbose=True):
    """Test a grip configuration. Returns dict with results."""
    cube_pos = np.array([0.09, 0.0, 0.22])
    if cube_quat is None:
        cube_quat = np.array([1.0, 0.0, 0.0, 0.0])

    palm_z = data.site_xpos[ids["palm_site"]][2] if data.site_xpos[ids["palm_site"]][2] > 0 else 0.197
    adr = ids["cube_qpos_adr"]

    # Establish grip
    reset_grip(model, data, ids, grip_qpos, cube_pos, cube_quat)

    # Check initial state
    cube_z_after_settle = data.qpos[adr + 2]
    tip_pos = data.site_xpos[ids["tip_sites"]]
    cube_center = data.qpos[adr:adr+3].copy()

    contact_info = get_contact_info(model, data, ids)
    n_contacts = sum(1 for f in FINGER_NAMES if contact_info[f]["in_contact"])

    if verbose:
        print(f"\n  After settle:")
        print(f"    Cube Z: {cube_z_after_settle:.4f} (palm Z ≈ 0.197)")
        print(f"    Contacts: {n_contacts}/3")
        for f in FINGER_NAMES:
            ci = contact_info[f]
            if ci["in_contact"]:
                nz = ci["normal"][2] if ci["normal"] is not None else 0
                print(f"    {f:>8}: force={ci['force']:.2f}N, normal_Z={nz:+.3f} "
                      f"({'UP' if nz > 0.05 else 'DOWN' if nz < -0.05 else 'HORIZ'})")
            else:
                print(f"    {f:>8}: NO CONTACT")

        # Fingertip positions relative to cube
        print(f"    Fingertip positions vs cube center ({cube_center[0]:.3f}, {cube_center[1]:.3f}, {cube_center[2]:.3f}):")
        for i, f in enumerate(FINGER_NAMES):
            rel = tip_pos[i] - cube_center
            print(f"    {f:>8}: dX={rel[0]:+.4f} dY={rel[1]:+.4f} dZ={rel[2]:+.4f}")

    # Phase 1: Hold with action=0
    dropped_at = None
    min_z = cube_z_after_settle
    data.ctrl[:7] = grip_qpos
    for step in range(hold_steps):
        mujoco.mj_step(model, data)
        cz = data.qpos[adr + 2]
        min_z = min(min_z, cz)
        if cz < 0.15:  # Clearly dropped
            dropped_at = step
            break

    hold_result = "HELD" if dropped_at is None else f"DROPPED at step {dropped_at}"
    final_z_hold = data.qpos[adr + 2]

    if verbose:
        print(f"\n  Hold test ({hold_steps} steps, action=0):")
        print(f"    Result: {hold_result}")
        print(f"    Z drift: {cube_z_after_settle:.4f} → {final_z_hold:.4f} (Δ={final_z_hold - cube_z_after_settle:+.4f})")

    if dropped_at is not None:
        return {"held": False, "noisy_held": False, "n_contacts": n_contacts,
                "contact_info": contact_info, "hold_dropped_at": dropped_at,
                "cube_z": cube_z_after_settle}

    # Phase 2: Noisy actions
    reset_grip(model, data, ids, grip_qpos, cube_pos, cube_quat)
    noisy_dropped_at = None
    for step in range(noise_steps):
        noise = np.random.randn(7) * noise_std
        ctrl = grip_qpos + noise * 0.5  # action_scale=0.5
        # Clip to joint limits
        joint_lo = np.array([model.jnt_range[jid, 0] for jid in ids["hand_joints"]])
        joint_hi = np.array([model.jnt_range[jid, 1] for jid in ids["hand_joints"]])
        ctrl = np.clip(ctrl, joint_lo, joint_hi)
        data.ctrl[:7] = ctrl
        for _ in range(10):  # frameskip
            mujoco.mj_step(model, data)
        cz = data.qpos[adr + 2]
        if cz < 0.15:
            noisy_dropped_at = step
            break

    noise_result = "HELD" if noisy_dropped_at is None else f"DROPPED at step {noisy_dropped_at}"
    if verbose:
        print(f"\n  Noise test ({noise_steps} steps, std={noise_std}):")
        print(f"    Result: {noise_result}")

    return {
        "held": True,
        "noisy_held": noisy_dropped_at is None,
        "n_contacts": n_contacts,
        "contact_info": contact_info,
        "cube_z": cube_z_after_settle,
        "z_drift": final_z_hold - cube_z_after_settle,
        "noisy_dropped_at": noisy_dropped_at,
    }


def sweep_grips(model, data, ids):
    """Sweep grip parameters to find stable configurations."""
    # Parameter ranges to sweep
    # Reduce middle/index curl to keep fingers on cube SIDES, not wrapping over top
    thumb_0_vals = [-0.5, -0.4, -0.3]       # abduction
    thumb_1_vals = [-0.4, -0.3, -0.2]       # curl
    thumb_2_vals = [-1.2, -1.0, -0.8]       # distal curl
    mid_0_vals = [0.7, 0.85, 1.0]           # proximal curl (was 1.1)
    mid_1_vals = [0.8, 1.0, 1.2]            # distal curl (was 1.222)

    results = []
    total = len(thumb_0_vals) * len(thumb_1_vals) * len(thumb_2_vals) * len(mid_0_vals) * len(mid_1_vals)
    print(f"Sweeping {total} configurations...")
    count = 0

    for t0 in thumb_0_vals:
        for t1 in thumb_1_vals:
            for t2 in thumb_2_vals:
                for m0 in mid_0_vals:
                    for m1 in mid_1_vals:
                        grip = np.array([t0, t1, t2, m0, m1, m0, m1])  # index = middle
                        count += 1
                        result = test_grip(model, data, ids, grip, verbose=False,
                                         hold_steps=500, noise_steps=500, noise_std=0.3)
                        result["grip"] = grip.tolist()

                        status = "PASS" if result["held"] and result["noisy_held"] else \
                                 "hold_ok" if result["held"] else "FAIL"

                        if count % 20 == 0 or result["held"]:
                            print(f"  [{count}/{total}] {status} "
                                  f"t=[{t0:.1f},{t1:.1f},{t2:.1f}] m=[{m0:.2f},{m1:.2f}] "
                                  f"contacts={result['n_contacts']}")

                        results.append(result)

    # Sort by quality
    passed = [r for r in results if r["held"] and r["noisy_held"]]
    hold_only = [r for r in results if r["held"] and not r["noisy_held"]]

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(passed)} PASS / {len(hold_only)} hold-only / {len(results)} total")
    print(f"{'='*60}")

    if passed:
        # Sort by least Z drift
        passed.sort(key=lambda r: abs(r.get("z_drift", 1.0)))
        print("\nTop 5 stable grips (pass both hold + noise):")
        for i, r in enumerate(passed[:5]):
            g = r["grip"]
            print(f"  #{i+1}: [{g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f}, {g[3]:.3f}, {g[4]:.3f}, {g[5]:.3f}, {g[6]:.3f}]")
            print(f"       contacts={r['n_contacts']}, z_drift={r.get('z_drift', 0):+.5f}")
    elif hold_only:
        hold_only.sort(key=lambda r: -(r.get("noisy_dropped_at", 0) or 0))
        print("\nTop 5 hold-only grips (sorted by noise survival):")
        for i, r in enumerate(hold_only[:5]):
            g = r["grip"]
            print(f"  #{i+1}: [{g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f}, {g[3]:.3f}, {g[4]:.3f}, {g[5]:.3f}, {g[6]:.3f}]")
            print(f"       contacts={r['n_contacts']}, noisy_drop_at={r.get('noisy_dropped_at', 'N/A')}")
    else:
        print("\nNo grips passed! All dropped during hold test.")
        # Show which got closest
        results.sort(key=lambda r: -(r.get("hold_dropped_at", 0) or 0))
        print("Longest surviving grips:")
        for i, r in enumerate(results[:5]):
            g = r["grip"]
            print(f"  #{i+1}: [{g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f}, {g[3]:.3f}, {g[4]:.3f}, {g[5]:.3f}, {g[6]:.3f}]")
            print(f"       dropped_at={r.get('hold_dropped_at', 'N/A')}")

    return passed, hold_only


def main():
    parser = argparse.ArgumentParser(description="Grip tuning for Dex3-1")
    parser.add_argument("--verify", action="store_true", help="Test current grip_qpos only")
    parser.add_argument("--grip", type=str, default=None, help="Test specific grip (comma-separated)")
    parser.add_argument("--steps", type=int, default=1000, help="Hold test steps")
    parser.add_argument("--noise-std", type=float, default=0.3, help="Noise std for robustness test")
    args = parser.parse_args()

    model, data = load_env()
    ids = get_ids(model)

    # Forward step to initialize site positions
    mujoco.mj_forward(model, data)

    if args.grip:
        grip = np.array([float(x) for x in args.grip.split(",")])
        assert len(grip) == 7, "Grip must have 7 values"
        print(f"Testing grip: {grip}")
        test_grip(model, data, ids, grip, hold_steps=args.steps, noise_std=args.noise_std)
    elif args.verify:
        print(f"Testing CURRENT grip_qpos: {CURRENT_GRIP}")
        test_grip(model, data, ids, CURRENT_GRIP, hold_steps=args.steps, noise_std=args.noise_std)
    else:
        print("Sweeping grip configurations...")
        sweep_grips(model, data, ids)


if __name__ == "__main__":
    main()
