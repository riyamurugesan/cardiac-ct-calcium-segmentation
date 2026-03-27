import numpy as np
from skimage import measure
import trimesh

def export_mask_as_obj(npy_path, spacing_path, output_path):
    mask = np.load(npy_path)
    spacing = np.load(spacing_path)
    #perform marching cubes algorithm - level = 0.5 so that the triangle in marching cubes is exactly halfway before 0 and 1 (calcium)
    verts, faces, normals = measure.marching_cubes(mask, level=0.5, spacing=spacing)
    #create a mesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    mesh.export(output_path)