"""Decimate STL meshes for faster MJX collision.

Reduces all meshes to <= 200 faces (palm especially: 5990 → 200).
"""

import fast_simplification
import trimesh
import os
import shutil

# Use project-relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

mesh_dir = os.path.join(project_root, 'models', 'dex3_assets')
out_dir = os.path.join(project_root, 'models', 'dex3_assets_decimated')
os.makedirs(out_dir, exist_ok=True)

TARGET_FACES = 200

for fname in sorted(os.listdir(mesh_dir)):
    fpath = os.path.join(mesh_dir, fname)
    if fname.endswith('.STL'):
        mesh = trimesh.load(fpath)
        n_faces = len(mesh.faces)
        if n_faces > TARGET_FACES:
            ratio = 1.0 - (float(TARGET_FACES) / n_faces)
            verts, faces = fast_simplification.simplify(
                mesh.vertices, mesh.faces, target_reduction=ratio
            )
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.export(os.path.join(out_dir, fname))
        print(f'{fname}: {n_faces} -> {len(mesh.faces)} faces')
    else:
        shutil.copy2(fpath, os.path.join(out_dir, fname))
        print(f'{fname}: copied')

print('Done!')
