"""Face detection from dice orientation — geometric method.

Standard dice layout (opposite faces sum to 7):
    Face 1: +Z (front)    Face 6: -Z (back)
    Face 2: +Y (top)      Face 5: -Y (bottom)
    Face 3: +X (right)    Face 4: -X (left)

This matches the UV mapping of the dex_cube.obj mesh + dice_texture.png.
"""

import numpy as np


# Face normal vectors in dice local frame
FACE_NORMALS = {
    1: np.array([0.0, 0.0, 1.0]),   # +Z (front)
    2: np.array([0.0, 1.0, 0.0]),   # +Y (top)
    3: np.array([1.0, 0.0, 0.0]),   # +X (right)
    4: np.array([-1.0, 0.0, 0.0]),  # -X (left)
    5: np.array([0.0, -1.0, 0.0]),  # -Y (bottom)
    6: np.array([0.0, 0.0, -1.0]),  # -Z (back)
}

# Target quaternions [w, x, y, z] that place each face pointing upward (+Z world)
FACE_TARGET_QUATS = {
    1: np.array([1.0, 0.0, 0.0, 0.0]),               # +Z already up (identity)
    2: np.array([0.7071068, 0.7071068, 0.0, 0.0]),   # +Y -> +Z: -90° around X
    3: np.array([0.7071068, 0.0, -0.7071068, 0.0]),  # +X -> +Z: -90° around Y
    4: np.array([0.7071068, 0.0, 0.7071068, 0.0]),   # -X -> +Z: +90° around Y
    5: np.array([0.7071068, -0.7071068, 0.0, 0.0]),  # -Y -> +Z: +90° around X
    6: np.array([0.0, 1.0, 0.0, 0.0]),               # -Z -> +Z: 180° around X
}


def quat_to_rot_matrix(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def detect_top_face(cube_quat):
    """Determine which dice face is currently on top (+Z in world frame).

    Args:
        cube_quat: Dice quaternion [w, x, y, z] from MuJoCo

    Returns:
        face_num: int 1-6
    """
    R = quat_to_rot_matrix(cube_quat)
    world_up = np.array([0.0, 0.0, 1.0])

    best_face = 1
    best_alignment = -2.0

    for face_num, local_normal in FACE_NORMALS.items():
        world_normal = R @ local_normal
        alignment = np.dot(world_normal, world_up)
        if alignment > best_alignment:
            best_alignment = alignment
            best_face = face_num

    return best_face


def get_target_quat(face_num):
    """Get target quaternion that places face_num on top."""
    return FACE_TARGET_QUATS[face_num].copy()
