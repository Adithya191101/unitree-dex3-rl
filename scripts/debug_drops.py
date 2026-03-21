"""Debug script: test what happens to the cube under different action regimes.

Tests:
1. Reset env, print grip state (joint positions, cube pos, contacts)
2. 50 steps of zero actions — does the cube stay?
3. 50 steps of small random actions (std=0.1) — does the cube stay?
4. 50 steps of normal random actions (std=0.5) — when does it drop?

Also diagnoses: position_target drift under velocity-integrated control.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.dex_cube_env import DexCubeEnv

XML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "dex3_dice_scene_torque.xml"
)


def print_state(env, label=""):
    """Print detailed environment state."""
    hand_qpos = np.array([env.data.qpos[adr] for adr in env.hand_qpos_adr])
    cube_pos = env._get_cube_pos()
    cube_quat = env._get_cube_quat()
    palm_pos = env.data.site_xpos[env.palm_site_id].copy()
    contacts = env._get_per_finger_contact()
    forces = env._get_contact_forces()
    n_contact = sum(contacts)

    print(f"  [{label}]")
    print(f"    hand_qpos:        {np.array2string(hand_qpos, precision=4, separator=', ')}")
    print(f"    position_targets: {np.array2string(env.position_targets, precision=4, separator=', ')}")
    print(f"    target - actual:  {np.array2string(env.position_targets - hand_qpos, precision=4, separator=', ')}")
    print(f"    grip_qpos:        {np.array2string(env.grip_qpos, precision=4, separator=', ')}")
    print(f"    cube_pos:         {np.array2string(cube_pos, precision=5, separator=', ')}")
    print(f"    palm_pos:         {np.array2string(palm_pos, precision=5, separator=', ')}")
    print(f"    cube_z:           {cube_pos[2]:.5f}  (drop threshold: {palm_pos[2] - 0.03:.5f})")
    print(f"    cube_quat:        {np.array2string(cube_quat, precision=4, separator=', ')}")
    print(f"    contacts:         {contacts} ({n_contact}/3 fingers)")
    print(f"    contact_forces:   {np.array2string(forces, precision=4, separator=', ')}")
    print()


def run_test(env, label, n_steps, action_fn, seed=42):
    """Run n_steps with given action function, tracking drops and state.

    Returns: step at which cube dropped (-1 if never dropped)
    """
    np.random.seed(seed)
    env.reset(target_face=1, seed=seed)

    print(f"\n{'='*70}")
    print(f"TEST: {label} ({n_steps} steps, action_scale={env.action_scale})")
    print(f"{'='*70}")
    print_state(env, "After reset")

    drop_step = -1
    # Track position target drift
    initial_targets = env.position_targets.copy()

    for step in range(n_steps):
        action = action_fn(step)
        obs, reward, done, info = env.step(action)

        # Print at key steps
        if step in [0, 4, 9, 19, 29, 39, 49, 99, 149, 199, 249, 299] and step < n_steps:
            cube_pos = env._get_cube_pos()
            palm_pos = env.data.site_xpos[env.palm_site_id].copy()
            target_drift = env.position_targets - initial_targets
            hand_qpos = np.array([env.data.qpos[adr] for adr in env.hand_qpos_adr])
            contacts = env._get_per_finger_contact()
            print(f"  Step {step+1:3d}: cube_z={cube_pos[2]:.4f}, "
                  f"contacts={sum(contacts)}/3, "
                  f"reward={reward:+.3f}, "
                  f"quat_dist={info['quat_dist']:.4f}, "
                  f"dropped={info['dropped']}")
            print(f"           pos_target_drift: {np.array2string(target_drift, precision=3, separator=', ')}")
            print(f"           pos_targets:      {np.array2string(env.position_targets, precision=3, separator=', ')}")
            print(f"           actual_qpos:      {np.array2string(hand_qpos, precision=3, separator=', ')}")

        if info["dropped"] and drop_step < 0:
            drop_step = step + 1
            print(f"\n  *** DROPPED at step {drop_step} ***")
            print_state(env, f"Drop state (step {drop_step})")
            # Continue running to see what happens after drop

        if done:
            print(f"  Episode done at step {step+1}: reward={reward:+.3f}, "
                  f"dropped={info['dropped']}, achieved={info['achieved_goal']}")
            break

    if drop_step < 0:
        print(f"\n  Cube survived all {n_steps} steps!")

    # Final state
    print_state(env, f"Final state (step {step+1})")

    # Summary of position target drift
    total_drift = env.position_targets - initial_targets
    print(f"  Position target drift summary:")
    print(f"    Total drift:     {np.array2string(total_drift, precision=4, separator=', ')}")
    print(f"    Max abs drift:   {np.max(np.abs(total_drift)):.4f} rad")
    print(f"    Joint limits lo: {np.array2string(env.joint_lo, precision=4, separator=', ')}")
    print(f"    Joint limits hi: {np.array2string(env.joint_hi, precision=4, separator=', ')}")
    print(f"    Final targets:   {np.array2string(env.position_targets, precision=4, separator=', ')}")
    print(f"    Grip qpos:       {np.array2string(env.grip_qpos, precision=4, separator=', ')}")

    return drop_step


def main():
    config = {
        "frameskip": 10,
        "max_episode_steps": 500,  # Long episodes to see drift
        "action_scale": 0.3,
        "reward": {
            "distance_scale": 1.0,
            "progress_scale": 15.0,
            "contact_bonus": 0.1,
            "contact_penalty_1": -0.1,
            "goal_bonus": 100.0,
            "goal_threshold": 0.03,
            "drop_penalty": -50.0,
            "drop_height": 0.03,
            "gait_lift_bonus": 0.0,
            "gait_replace_bonus": 0.0,
            "hold_bonus": 0.05,
            "action_smooth_coef": 0.02,
            "gait_force_threshold": 0.3,
        },
    }

    env = DexCubeEnv(xml_path=XML_PATH, config=config)

    # Print joint limits and control ranges
    print("=" * 70)
    print("ENVIRONMENT INFO")
    print("=" * 70)
    print(f"XML: {XML_PATH}")
    print(f"Joint limits (lo): {np.array2string(env.joint_lo, precision=4, separator=', ')}")
    print(f"Joint limits (hi): {np.array2string(env.joint_hi, precision=4, separator=', ')}")
    print(f"Joint ranges:      {np.array2string(env.joint_hi - env.joint_lo, precision=4, separator=', ')}")
    print(f"Ctrl ranges:\n{env.ctrl_ranges}")
    print(f"Grip qpos:         {np.array2string(env.grip_qpos, precision=4, separator=', ')}")
    print(f"Action scale:      {env.action_scale}")
    print(f"Frameskip:         {env.frameskip}")
    print(f"Timestep:          {env.model.opt.timestep}")
    print(f"Effective dt:      {env.model.opt.timestep * env.frameskip}")

    # Compute max possible drift per step
    max_drift_per_step = 1.0 * env.action_scale  # action clipped to [-1, 1]
    print(f"\nMax drift per step: {max_drift_per_step:.3f} rad")
    print(f"Max drift in 50 steps:  {50 * max_drift_per_step:.3f} rad ({np.degrees(50 * max_drift_per_step):.1f} deg)")
    print(f"Max drift in 100 steps: {100 * max_drift_per_step:.3f} rad ({np.degrees(100 * max_drift_per_step):.1f} deg)")
    print(f"Max drift in 300 steps: {300 * max_drift_per_step:.3f} rad ({np.degrees(300 * max_drift_per_step):.1f} deg)")

    # Test 1: Zero actions (hold)
    drop1 = run_test(
        env, "ZERO ACTIONS (should hold perfectly)", 50,
        action_fn=lambda step: np.zeros(7),
    )

    # Test 2: Small random actions (std=0.1)
    drop2 = run_test(
        env, "SMALL RANDOM ACTIONS (std=0.1)", 50,
        action_fn=lambda step: np.random.randn(7) * 0.1,
    )

    # Test 3: Normal random actions (std=0.5, like policy with action_std=0.528)
    drop3 = run_test(
        env, "NORMAL RANDOM ACTIONS (std=0.5, simulating policy exploration)", 50,
        action_fn=lambda step: np.random.randn(7) * 0.5,
    )

    # Test 4: Normal random actions extended (std=0.5, 300 steps to see full drift)
    drop4 = run_test(
        env, "EXTENDED RANDOM ACTIONS (std=0.5, 300 steps for full drift)", 300,
        action_fn=lambda step: np.random.randn(7) * 0.5,
    )

    # Test 5: Constant action (all +1) to see worst-case drift
    drop5 = run_test(
        env, "WORST CASE: constant action=+1.0 (max drift)", 50,
        action_fn=lambda step: np.ones(7),
    )

    # Test 6: Biased random (mean=0.3, std=0.5) — what a poorly trained policy might do
    drop6 = run_test(
        env, "BIASED RANDOM (mean=0.3, std=0.5, poorly trained policy)", 100,
        action_fn=lambda step: np.random.randn(7) * 0.5 + 0.3,
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    results = [
        ("Zero actions (50 steps)", drop1),
        ("Small random std=0.1 (50 steps)", drop2),
        ("Normal random std=0.5 (50 steps)", drop3),
        ("Extended random std=0.5 (300 steps)", drop4),
        ("Worst case all +1.0 (50 steps)", drop5),
        ("Biased random mean=0.3 std=0.5 (100 steps)", drop6),
    ]

    for label, drop_step in results:
        if drop_step < 0:
            print(f"  {label}: SURVIVED")
        else:
            print(f"  {label}: DROPPED at step {drop_step}")

    # Analysis of velocity integration drift
    print("\n" + "=" * 70)
    print("VELOCITY-INTEGRATION DRIFT ANALYSIS")
    print("=" * 70)
    print("""
Key insight: position_targets += action * action_scale EVERY step.
With random actions (mean~0), the targets do a random walk:
  - After N steps, expected drift ~ std * action_scale * sqrt(N)
  - std=0.5, scale=0.3: drift after 50 steps ~ 0.5 * 0.3 * sqrt(50) = 1.06 rad
  - std=0.5, scale=0.3: drift after 300 steps ~ 0.5 * 0.3 * sqrt(300) = 2.60 rad

The joints will hit their limits and STAY there (clipping prevents return).
This means after enough steps, the fingers are fully open or fully closed.
With biased actions (mean != 0), drift is even faster: mean * scale * N.
  - mean=0.3, scale=0.3: after 100 steps, drift = 0.3 * 0.3 * 100 = 9.0 rad (clipped to limits)

The policy MUST learn to output near-zero MEAN actions to avoid drift,
but with action_std=0.528, exploration noise alone causes massive drift.
""")

    env.close()


if __name__ == "__main__":
    main()
