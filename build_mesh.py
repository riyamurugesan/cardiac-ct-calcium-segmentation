import numpy as np
from skimage import measure
import trimesh

def build_mesh(npy_path, spacing_path):
    """Utilizes Marching Squares algorithm to create mesh from .npy array."""
    mask = np.load(npy_path)
    spacing = np.load(spacing_path)
    #perform marching cubes algorithm - level = 0.5 so the triangle in marching cubes is exactly halfway before 0 and 1 (calcium)
    verts, faces, normals, values = measure.marching_cubes(mask, level=0.5, spacing=spacing)
    #create a mesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    return mesh