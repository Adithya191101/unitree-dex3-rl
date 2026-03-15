# Dex_Uni: In-Hand Dice Reorientation with Reinforcement Learning

Train a **Unitree Dex3-1** robotic hand (3-finger, 7 DOF) to reorient a dice in-hand to show any target face (1-6) on top, using Proximal Policy Optimization (PPO) in MuJoCo simulation.

<p align="center">
  <img src="assets/final_front.png" width="32%" />
  <img src="assets/final_closeup.png" width="32%" />
  <img src="assets/final_top.png" width="32%" />
</p>
<p align="center"><em>Dex3-1 hand grasping a dice with three-finger precision grip</em></p>

---

## Overview

This project implements a complete RL pipeline for dexterous in-hand manipulation:

- **Hand Model**: Unitree Dex3-1 right hand extracted from [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie), with 3 fingers (thumb, middle, index) and 7 actuated joints
- **Task**: Reorient a standard dice (opposite faces sum to 7) to show a user-specified face on top
- **Algorithm**: Custom PPO implementation in PyTorch with multi-phase curriculum learning
- **Simulation**: MuJoCo (CPU) and MJX (GPU-accelerated) physics backends
- **Control**: Native position actuators with residual action space (action = 0 holds the grip)

### The Hand

<p align="center">
  <img src="assets/front_view.png" width="24%" />
  <img src="assets/side_view.png" width="24%" />
  <img src="assets/top_view.png" width="24%" />
  <img src="assets/closeup_view.png" width="24%" />
</p>
<p align="center"><em>Dex3-1 hand: front, side, top, and scene overview</em></p>

### Grasping the Cube

<p align="center">
  <img src="assets/palmup_front.png" width="32%" />
  <img src="assets/palmup_side.png" width="32%" />
  <img src="assets/palmup_top.png" width="32%" />
</p>
<p align="center"><em>Palm-up grasp configuration from multiple angles</em></p>

<p align="center">
  <img src="assets/fixed_front.png" width="45%" />
  <img src="assets/fixed_side.png" width="45%" />
</p>
<p align="center"><em>Fixed grasp with cube held between three fingers</em></p>

---

## Project Structure

```
Dex_Uni/
├── configs/                    # Training configurations (YAML)
│   ├── ppo_mjx_gait.yaml      # MJX GPU training + domain randomization
│   ├── ppo_cpu_gait.yaml       # CPU training with finger gaiting
│   ├── ppo_cpu_finetune.yaml   # CPU fine-tune from MJX checkpoint
│   ├── ppo_mjx_config.yaml     # Base MJX config (11-phase curriculum)
│   ├── ppo_parallel_config.yaml# Base CPU parallel config
│   └── ppo_cpu_config.yaml     # Single-target CPU config
│
├── envs/                       # Environment implementations
│   ├── dex_cube_env.py         # Single-env MuJoCo environment (obs_dim=48, act_dim=7)
│   ├── mjx_vec_env.py          # GPU-vectorized env using MJX (2048 parallel envs)
│   ├── vec_env.py              # CPU SubprocVecEnv (multiprocessing)
│   └── reward.py               # Reward functions (distance + progress + gait)
│
├── rl/                         # Reinforcement learning components
│   ├── ppo.py                  # PPO with value clipping, obs normalization
│   ├── actor_critic.py         # Actor-Critic network (512-256-128, tanh)
│   └── buffer.py               # Rollout buffer with per-env GAE
│
├── training/                   # Training scripts
│   ├── train_mjx.py            # Hybrid GPU physics + CPU reward training
│   ├── train_parallel.py       # CPU parallel training with curriculum
│   └── evaluate.py             # Per-face evaluation + video recording
│
├── perception/                 # Dice state detection
│   └── face_detector.py        # Geometric face detection + target quaternions
│
├── ui/                         # Visualization
│   ├── viewer.py               # Interactive MuJoCo viewer (press 1-6 for faces)
│   └── viewer_cpu.py           # CPU-only viewer
│
├── models/                     # MuJoCo XML models
│   ├── dex3_dice_scene_torque.xml  # CPU model (mesh collisions, position actuators)
│   ├── dex3_dice_scene_mjx.xml    # MJX model (primitive collisions for GPU)
│   ├── dex3_dice_scene.xml        # Base scene
│   └── dex3_assets/               # STL meshes + dice OBJ
│
├── scripts/                    # Utility scripts
│   ├── test_env.py             # Environment integration test
│   ├── test_reward.py          # Reward function unit tests
│   ├── generate_textures.py    # Dice texture generator
│   └── decimate_meshes.py      # Mesh simplification
│
└── assets/                     # Images and resources
    └── *.png                   # Screenshots for documentation
```

---

## Approach

### Observation Space (48 dimensions)

| Component | Dims | Description |
|-----------|------|-------------|
| Joint positions | 7 | Hand joint angles |
| Joint velocities | 7 | Hand joint angular velocities |
| Relative cube position | 3 | Cube position relative to palm |
| Cube quaternion | 4 | Current dice orientation [w,x,y,z] |
| Cube angular velocity | 3 | Dice rotation rate |
| Target quaternion | 4 | Desired orientation for target face |
| Fingertip positions | 9 | 3D position of each fingertip (3x3) |
| Current face | 1 | Which face is currently on top (normalized) |
| Previous action | 7 | Last action taken (for smoothing) |
| Contact forces | 3 | Per-finger contact strength with dice |

### Action Space (7 dimensions)

Residual position offsets applied to a pre-tuned grip configuration:

```
ctrl = grip_qpos + action * action_scale
```

- `action = 0` holds the dice at the default grip position
- `action_scale = 0.3 rad` (reduced in early curriculum phases)
- Native MuJoCo position actuators (`kp=5, dampratio=1, forcerange=[-1.5, 1.5]`)

### Reward Design

The reward function combines continuous shaping with sparse task completion signals:

| Component | Formula | Purpose |
|-----------|---------|---------|
| Distance | `-quat_distance * scale` | Potential-based shaping toward target |
| Progress | `(prev_dist - curr_dist) * 15.0` | Directional signal for improvement |
| Goal bonus | `+100.0` (one-time) | Sparse reward for reaching target |
| Drop penalty | `-50.0` | Penalize losing the cube |
| Contact bonus | `+0.1` (3 fingers), `+0.05` (2) | Encourage stable grip |
| Gait lift | `+0.5` when finger lifts with 2 holding | Encourage finger gaiting |
| Gait replace | `+0.3` when finger re-contacts | Complete the gait cycle |
| Action smooth | `-0.02 * \|\|a - a_prev\|\|^2` | Penalize jittery actions |
| Hold bonus | `+0.05` (3 contact + low angvel) | Reward stable states |

**Finger gaiting**: Bioinspired approach where 2 fingers hold the object while 1 lifts, repositions, and re-engages. A cooldown timer (5 steps) prevents reward hacking through rapid cycling.

### Curriculum Learning (7 Phases)

Training progresses through graduated difficulty levels:

| Phase | Task | Max Angle | Advance SR |
|-------|------|-----------|------------|
| 1. GRIP | Hold cube stable | 0.0 rad | 95% |
| 2. MICRO_ROTATE | Tiny rotations | 0.3 rad (~17 deg) | 70% |
| 3. GAIT_EMERGE | Medium rotations, gaiting bonus | 0.8 rad (~46 deg) | 45% |
| 4. MED_ROTATE | ~90 degree rotations | 1.57 rad (~90 deg) | 40% |
| 5. SINGLE_FULL | Full rotation, one start face | 3.15 rad | 35% |
| 6. MULTI_FULL | Full rotation, multiple starts | 3.15 rad | 30% |
| 7. ALL_FACES | Any face to any face | 3.15 rad | 80% |

Per-phase control of: learning rate, exploration noise (log_std bounds), entropy bonus, episode length, action scale, and reward weights.

### Dice Face Mapping

```
Face 1: +Z (top)      Face 6: -Z (bottom)     (opposite faces sum to 7)
Face 2: +Y (front)    Face 5: -Y (back)
Face 3: +X (right)    Face 4: -X (left)
```

<p align="center">
  <img src="assets/dice_texture.png" width="25%" />
</p>
<p align="center"><em>UV-mapped dice texture (standard layout)</em></p>

---

## Training

### MJX GPU Training (Recommended)

Trains on 2048 parallel environments using GPU-accelerated MuJoCo XLA physics with domain randomization:

```bash
python training/train_mjx.py --config configs/ppo_mjx_gait.yaml
```

- **Hardware**: RTX 4090, ~18s/update, ~20 hours for 4000 updates
- **Data**: 131K transitions per update (2048 envs x 64 steps)
- **Domain Randomization**: Friction, mass, damping, stiffness, observation noise
- **Result**: 100% eval success rate on all 6 faces in MJX physics

### CPU Parallel Training

Trains on CPU MuJoCo with SubprocVecEnv (16-64 parallel environments):

```bash
python training/train_parallel.py --config configs/ppo_cpu_gait.yaml
```

### CPU Fine-Tuning (from MJX checkpoint)

Adapts an MJX-trained policy to CPU MuJoCo physics:

```bash
python training/train_parallel.py --config configs/ppo_cpu_finetune.yaml --resume checkpoints/final_model.pt
```

### Resume Training

```bash
python training/train_mjx.py --config configs/ppo_mjx_gait.yaml --resume checkpoints/ppo_update_1000.pt --start_phase 3
```

---

## Evaluation

Run per-face evaluation on CPU MuJoCo:

```bash
python training/evaluate.py --checkpoint checkpoints/final_model.pt --config configs/ppo_mjx_gait.yaml --episodes 100
```

Record videos:

```bash
python training/evaluate.py --checkpoint checkpoints/final_model.pt --record
```

Output:
```
  Face   Success%     Mean R   Mean Len      Drop%
--------------------------------------------------
     1      100.0%      95.20       12.3       0.0%
     2       98.0%      89.50       18.1       0.0%
     ...
   ALL       99.2%      91.30
```

---

## Interactive Viewer

Launch the MuJoCo viewer with a trained policy:

```bash
python ui/viewer.py --checkpoint checkpoints/final_model.pt
```

**Controls**:
- Press **1-6** to command dice reorientation to that face
- The policy runs in real-time, rotating the dice to show the target face on top

---

## Key Technical Details

### Position Actuators

MuJoCo's native position actuators handle PD control internally, providing stable control for RL:

```xml
<actuator>
  <position name="thumb_0_act" joint="thumb_0" kp="5" dampratio="1" forcerange="-1.5 1.5"/>
</actuator>
```

### Quaternion Distance

Orientation error uses quaternion distance with double-cover handling:

```python
def quat_distance(q1, q2):
    return 1.0 - abs(dot(q1, q2))  # range [0, 1]
```

### Grip Configuration

Pre-tuned joint positions that form a stable three-finger cradle:

```python
grip_qpos = [-0.419, -0.339, -1.047, 1.100, 1.222, 1.100, 1.222]
```

Reset uses a 2-phase sequence: 300-step interpolation to grip + 100-step settle.

---

## Results

### MJX Training (GPU, 2048 envs, domain randomization)

- All 7 curriculum phases completed
- **100% eval success rate** on all 6 target faces in MJX physics
- 524M timesteps, ~20 hours on RTX 4090
- 0% drop rate throughout training

### Sim-to-Sim Transfer Gap

The MJX-trained policy achieves ~24% success rate when transferred to CPU MuJoCo without fine-tuning. This gap is primarily due to differences in collision geometry (MJX uses primitive shapes, CPU uses mesh collisions), not physics parameters. Domain randomization on friction/mass/damping did not significantly close this gap.

**Possible solutions** (not yet implemented):
- CPU fine-tuning from MJX checkpoint (lower learning rate, same task)
- Unified collision geometry between MJX and CPU models
- Asymmetric domain randomization targeting collision response

---

## Installation

```bash
# Clone
git clone https://github.com/<your-username>/Dex_Uni.git
cd Dex_Uni

# Install dependencies
pip install mujoco torch numpy pyyaml tensorboard tqdm

# For MJX GPU training (requires CUDA)
pip install mujoco-mjx jax[cuda12]
```

### Requirements

- Python 3.10+
- MuJoCo 3.5+
- PyTorch 2.0+
- JAX with CUDA (for MJX training only)
- NumPy, PyYAML, TensorBoard, tqdm

---

## References

- [MuJoCo](https://mujoco.readthedocs.io/) - Physics simulation
- [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) - GPU-accelerated MuJoCo via JAX
- [Unitree G1 / Dex3-1](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1) - Hand model from MuJoCo Menagerie
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) - Reference for MJX training patterns
