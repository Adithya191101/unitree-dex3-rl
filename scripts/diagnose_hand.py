"""Diagnose hand orientation, grip, and action space limits.
Prints world-frame positions/orientations to understand the geometry."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mujoco

XML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "models", "dex3_dice_scene_torque.xml")
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)

JOINT_NAMES = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]
GRIP_QPOS = np.array([-0.419, -0.339, -1.047, 1.100, 1.222, 1.100, 1.222])

# Helper to get IDs
def jid(name): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
def sid(name): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
def bid(name): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)

hand_jids = [jid(n) for n in JOINT_NAMES]
hand_qpos_adr = [m.jnt_qposadr[j] for j in hand_jids]

print("=" * 70)
print("HAND ORIENTATION DIAGNOSTIC")
print("=" * 70)

# Test 1: Default pose (all joints = 0)
mujoco.mj_resetData(m, d)
mujoco.mj_forward(m, d)
print("\n--- TEST 1: All joints at 0 (fingers straight) ---")
print(f"  Palm site (world):  {d.site_xpos[sid('palm_site')]}")
print(f"  Thumb tip (world):  {d.site_xpos[sid('thumb_tip_site')]}")
print(f"  Middle tip (world): {d.site_xpos[sid('middle_tip_site')]}")
print(f"  Index tip (world):  {d.site_xpos[sid('index_tip_site')]}")
palm_z = d.site_xpos[sid('palm_site')][2]
tips_z = np.mean([d.site_xpos[sid(s)][2] for s in ['thumb_tip_site', 'middle_tip_site', 'index_tip_site']])
print(f"  Palm Z={palm_z:.4f}, Tips mean Z={tips_z:.4f}")
if tips_z > palm_z:
    print(f"  -> Tips are ABOVE palm by {tips_z-palm_z:.4f}m => fingers curl UPWARD (palm-up config)")
else:
    print(f"  -> Tips are BELOW palm by {palm_z-tips_z:.4f}m => fingers curl DOWNWARD")

# Test 2: Grip pose
mujoco.mj_resetData(m, d)
for i, adr in enumerate(hand_qpos_adr):
    d.qpos[adr] = GRIP_QPOS[i]
mujoco.mj_forward(m, d)
print(f"\n--- TEST 2: Grip pose (current grip_qpos) ---")
print(f"  Palm site (world):  {d.site_xpos[sid('palm_site')]}")
print(f"  Thumb tip (world):  {d.site_xpos[sid('thumb_tip_site')]}")
print(f"  Middle tip (world): {d.site_xpos[sid('middle_tip_site')]}")
print(f"  Index tip (world):  {d.site_xpos[sid('index_tip_site')]}")
for name in JOINT_NAMES:
    j = jid(name)
    lo, hi = m.jnt_range[j]
    val = GRIP_QPOS[JOINT_NAMES.index(name)]
    pct = (val - lo) / (hi - lo) * 100
    print(f"  {name:12s}: {np.degrees(val):7.1f} deg  (range [{np.degrees(lo):.1f}, {np.degrees(hi):.1f}])  {pct:.0f}% of range")

# Test 3: Where does the cube sit?
cube_jid = jid("cube_joint")
cube_adr = m.jnt_qposadr[cube_jid]
print(f"\n--- TEST 3: Cube default position ---")
print(f"  Cube spawn pos: {d.qpos[cube_adr:cube_adr+3]}")
print(f"  Cube spawn quat: {d.qpos[cube_adr+3:cube_adr+7]}")

# Test 4: Action space analysis
print(f"\n--- TEST 4: Action space reach analysis ---")
print(f"  Current: ctrl = grip_qpos + action * 0.3, action in [-1, 1]")
for i, name in enumerate(JOINT_NAMES):
    j = jid(name)
    lo, hi = m.jnt_range[j]
    grip = GRIP_QPOS[i]
    ctrl_lo = grip - 0.3
    ctrl_hi = grip + 0.3
    ctrl_lo_clipped = max(ctrl_lo, lo)
    ctrl_hi_clipped = min(ctrl_hi, hi)
    range_accessible = ctrl_hi_clipped - ctrl_lo_clipped
    range_total = hi - lo
    pct = range_accessible / range_total * 100
    print(f"  {name:12s}: can reach [{np.degrees(ctrl_lo_clipped):6.1f}, {np.degrees(ctrl_hi_clipped):6.1f}] deg"
          f"  = {pct:.0f}% of full range [{np.degrees(lo):.1f}, {np.degrees(hi):.1f}]")

# Test 5: Velocity-integrated action reach
print(f"\n--- TEST 5: Velocity-integrated reach (IsaacGym style) ---")
print(f"  target += action * 20.0 * 0.02 = action * 0.4 rad/step")
print(f"  After 10 steps at action=1: +4.0 rad = full range accessible")
print(f"  After 10 steps at action=-1: -4.0 rad = full range accessible")
print(f"  -> FULL joint range reachable within 5-10 steps for any joint")

# Test 6: Palm normal direction
print(f"\n--- TEST 6: Palm orientation (body frame analysis) ---")
wrist_bid = bid("right_wrist_yaw_link")
wrist_xmat = d.xmat[wrist_bid].reshape(3, 3)
print(f"  Wrist body rotation matrix:\n{wrist_xmat}")
# The palm normal in wrist local frame is approximately +Y (fingers curl toward +Y)
palm_normal_local = np.array([0, 1, 0])
palm_normal_world = wrist_xmat @ palm_normal_local
print(f"  Palm normal (local +Y -> world): {palm_normal_world}")
if palm_normal_world[2] > 0.5:
    print(f"  -> Palm faces UP (Z component = {palm_normal_world[2]:.3f})")
elif palm_normal_world[2] < -0.5:
    print(f"  -> Palm faces DOWN (Z component = {palm_normal_world[2]:.3f})")
else:
    print(f"  -> Palm faces SIDEWAYS (Z component = {palm_normal_world[2]:.3f})")

# Test 7: Hold test with grip_qpos
print(f"\n--- TEST 7: Hold stability test (1000 steps, action=0) ---")
mujoco.mj_resetData(m, d)
# Place cube
d.qpos[cube_adr:cube_adr+3] = [0.09, 0.0, 0.237]
d.qpos[cube_adr+3:cube_adr+7] = [1, 0, 0, 0]
# Close grip slowly
open_qpos = np.zeros(7)
for step in range(300):
    t = min(step / 200.0, 1.0)
    target = open_qpos + t * (GRIP_QPOS - open_qpos)
    d.ctrl[:7] = target
    mujoco.mj_step(m, d)
d.ctrl[:7] = GRIP_QPOS
for _ in range(100):
    mujoco.mj_step(m, d)
mujoco.mj_forward(m, d)

start_cube_z = d.qpos[cube_adr + 2]
palm_z = d.site_xpos[sid('palm_site')][2]
print(f"  After grip settle: cube Z={start_cube_z:.4f}, palm Z={palm_z:.4f}")

drops = 0
for step in range(1000):
    d.ctrl[:7] = GRIP_QPOS  # action=0 equivalent
    mujoco.mj_step(m, d)
    cube_z = d.qpos[cube_adr + 2]
    if cube_z < palm_z - 0.03:
        drops += 1
        break

mujoco.mj_forward(m, d)
final_cube_z = d.qpos[cube_adr + 2]
print(f"  After 1000 steps: cube Z={final_cube_z:.4f}, dropped={drops > 0}")
print(f"  Cube drift: {final_cube_z - start_cube_z:.4f}m")

# Test 8: Tips enclosing analysis
print(f"\n--- TEST 8: Do fingertips enclose the dice? ---")
cube_pos = d.qpos[cube_adr:cube_adr+3].copy()
for name in ['thumb_tip_site', 'middle_tip_site', 'index_tip_site']:
    tip = d.site_xpos[sid(name)]
    diff = tip - cube_pos
    dist = np.linalg.norm(diff)
    print(f"  {name:20s}: pos={tip}, dist_to_cube={dist:.4f}m, offset={diff}")

# Check if all tips are above cube (enclosing from above)
tips = [d.site_xpos[sid(s)] for s in ['thumb_tip_site', 'middle_tip_site', 'index_tip_site']]
tips_above = sum(1 for t in tips if t[2] > cube_pos[2])
print(f"  Tips above cube center: {tips_above}/3")
if tips_above == 3:
    print(f"  -> ALL tips are above dice = fingers curl OVER the dice (enclosing)")
elif tips_above == 0:
    print(f"  -> ALL tips below dice = fingers support from below")
else:
    print(f"  -> Mixed: {tips_above} above, {3-tips_above} below")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
