"""Sweep grip configurations to find stable + visible grips.
Goal: fingertips at SIDES of dice (not above), dice visible, hold stable."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mujoco

XML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "models", "dex3_dice_scene_torque.xml")
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)

JOINT_NAMES = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]

def jid(name): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
def sid(name): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)

hand_jids = [jid(n) for n in JOINT_NAMES]
hand_qpos_adr = [m.jnt_qposadr[j] for j in hand_jids]
cube_jid = jid("cube_joint")
cube_adr = m.jnt_qposadr[cube_jid]
cube_vel_adr = m.jnt_dofadr[cube_jid]
palm_sid = sid("palm_site")

def test_grip(grip_qpos, cube_pos, cube_quat, label="", hold_steps=1000):
    """Test a grip: slow close + settle + hold. Return stats."""
    mujoco.mj_resetData(m, d)

    # Set cube
    d.qpos[cube_adr:cube_adr+3] = cube_pos
    d.qpos[cube_adr+3:cube_adr+7] = cube_quat

    # Open hand first
    open_qpos = np.zeros(7)
    for i, adr in enumerate(hand_qpos_adr):
        d.qpos[adr] = open_qpos[i]

    # Slow close (300 steps)
    for step in range(300):
        t = min(step / 200.0, 1.0)
        target = open_qpos + t * (grip_qpos - open_qpos)
        d.ctrl[:7] = target
        # Pin cube during close
        d.qpos[cube_adr:cube_adr+3] = cube_pos
        d.qpos[cube_adr+3:cube_adr+7] = cube_quat
        d.qvel[cube_vel_adr:cube_vel_adr+6] = 0
        mujoco.mj_step(m, d)

    # Settle (100 steps, cube free)
    d.ctrl[:7] = grip_qpos
    for _ in range(100):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)

    settle_cube_z = d.qpos[cube_adr + 2]
    palm_z = d.site_xpos[palm_sid][2]

    # Check if already dropped during settle
    if settle_cube_z < palm_z - 0.03:
        return {"label": label, "hold_rate": 0.0, "dropped_at": "settle",
                "tips_above": -1, "max_tip_z_above_cube": -1}

    # Hold test
    dropped = False
    drop_step = hold_steps
    for step in range(hold_steps):
        d.ctrl[:7] = grip_qpos
        mujoco.mj_step(m, d)
        cube_z = d.qpos[cube_adr + 2]
        if cube_z < palm_z - 0.03:
            dropped = True
            drop_step = step
            break

    mujoco.mj_forward(m, d)

    # Analyze tip positions
    cube_pos_now = d.qpos[cube_adr:cube_adr+3].copy()
    tips = {}
    for name in ['thumb_tip_site', 'middle_tip_site', 'index_tip_site']:
        tip = d.site_xpos[sid(name)].copy()
        tips[name] = tip

    tips_above = sum(1 for t in tips.values() if t[2] > cube_pos_now[2])
    max_tip_z_above = max(t[2] - cube_pos_now[2] for t in tips.values())
    min_tip_dist = min(np.linalg.norm(t - cube_pos_now) for t in tips.values())

    return {
        "label": label,
        "grip_qpos": grip_qpos.copy(),
        "hold_rate": drop_step / hold_steps,
        "dropped_at": drop_step if dropped else "never",
        "tips_above": tips_above,
        "max_tip_z_above_cube": max_tip_z_above,
        "min_tip_dist": min_tip_dist,
        "cube_z_final": d.qpos[cube_adr + 2],
        "cube_drift": abs(d.qpos[cube_adr + 2] - settle_cube_z),
    }


# Cube position and default orientation
cube_pos = np.array([0.09, 0.0, 0.237])
cube_quat = np.array([1.0, 0.0, 0.0, 0.0])

# Define grip configs to test
# Format: [thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1]
# Joint ranges:
#   thumb_0: [-60, 60] deg, thumb_1: [-60, 41.5] deg, thumb_2: [-100, 0] deg
#   middle_0: [0, 90] deg, middle_1: [0, 100] deg
#   index_0: [0, 90] deg, index_1: [0, 100] deg

configs = {
    "current (tight)": np.array([-0.419, -0.339, -1.047, 1.100, 1.222, 1.100, 1.222]),

    # More open middle/index (less curl)
    "open_v1 (mid/idx 30deg)": np.array([-0.3, -0.3, -0.8, 0.524, 0.524, 0.524, 0.524]),
    "open_v2 (mid/idx 40deg)": np.array([-0.3, -0.3, -0.9, 0.698, 0.698, 0.698, 0.698]),
    "open_v3 (mid/idx 50deg)": np.array([-0.35, -0.3, -0.9, 0.873, 0.873, 0.873, 0.873]),

    # Very open - fingertips at sides
    "wide_v1 (mid/idx 20deg)": np.array([-0.2, -0.2, -0.6, 0.349, 0.349, 0.349, 0.349]),
    "wide_v2 (mid/idx 25deg)": np.array([-0.25, -0.25, -0.7, 0.436, 0.436, 0.436, 0.436]),

    # Thumb more engaged, mid/idx lighter
    "asym_v1": np.array([-0.35, -0.4, -1.0, 0.6, 0.6, 0.6, 0.6]),
    "asym_v2": np.array([-0.4, -0.35, -0.9, 0.5, 0.7, 0.5, 0.7]),

    # Cradle grip - middle/index proximal open, distal more curled
    "cradle_v1": np.array([-0.3, -0.3, -0.8, 0.4, 0.9, 0.4, 0.9]),
    "cradle_v2": np.array([-0.3, -0.25, -0.7, 0.35, 0.8, 0.35, 0.8]),

    # Minimal grip - just enough to touch
    "minimal_v1": np.array([-0.2, -0.15, -0.5, 0.3, 0.4, 0.3, 0.4]),
    "minimal_v2": np.array([-0.25, -0.2, -0.6, 0.4, 0.5, 0.4, 0.5]),
}

# Also test with different cube heights (palm-up, gravity helps)
cube_heights = [0.237, 0.230, 0.225, 0.220]

print("=" * 90)
print("GRIP CONFIGURATION SWEEP")
print("=" * 90)
print(f"{'Config':<30} {'Hold%':>6} {'Drop':>8} {'Tips>Cube':>10} {'MaxTipZ':>8} {'MinDist':>8}")
print("-" * 90)

results = []
for name, grip in configs.items():
    r = test_grip(grip, cube_pos, cube_quat, label=name)
    results.append(r)
    print(f"{name:<30} {r['hold_rate']*100:>5.0f}% {str(r['dropped_at']):>8} "
          f"{r['tips_above']:>10}/3 {r.get('max_tip_z_above_cube',0):>7.4f}m {r.get('min_tip_dist',0):>7.4f}m")

# Find best: stable hold + tips NOT all above cube
print("\n" + "=" * 90)
print("BEST CANDIDATES (hold >= 100% AND tips_above < 3)")
print("=" * 90)
good = [r for r in results if r['hold_rate'] >= 1.0 and r['tips_above'] < 3]
if good:
    for r in good:
        print(f"  {r['label']}: tips_above={r['tips_above']}, grip={r['grip_qpos']}")
else:
    print("  No config found with 100% hold AND tips not all above cube.")
    print("\n  Relaxing: hold >= 80%:")
    ok = [r for r in results if r['hold_rate'] >= 0.8]
    for r in sorted(ok, key=lambda x: x['tips_above']):
        print(f"  {r['label']}: hold={r['hold_rate']*100:.0f}%, tips_above={r['tips_above']}")

# Also test lower cube positions for best grips that held
print("\n" + "=" * 90)
print("CUBE HEIGHT SWEEP (for configs with 100% hold)")
print("=" * 90)
stable_grips = [(r['label'], r['grip_qpos']) for r in results if r['hold_rate'] >= 1.0]
for name, grip in stable_grips[:5]:
    for h in cube_heights:
        cp = np.array([0.09, 0.0, h])
        r = test_grip(grip, cp, cube_quat, label=f"{name} h={h}")
        print(f"  {name:<30} h={h:.3f}: hold={r['hold_rate']*100:.0f}%, tips_above={r['tips_above']}/3")
