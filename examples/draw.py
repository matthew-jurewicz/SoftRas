import sys
import soft_renderer as sr
import os
import torch
import numpy as np

file = sys.argv[1]
renderer = sr.SoftRenderer(
    image_size=64, camera_mode='look_at', viewing_angle=15)
parent = os.path.dirname(os.path.abspath(__file__))
cam_info = torch.from_numpy(
    np.load(parent + '/../data/camera.npy'))
renderer.transform.set_eyes_from_angles(
    distances=cam_info[:,0], 
    elevations=cam_info[:,1], 
    azimuths=cam_info[:,2]
)
object = sr.Mesh.from_obj(file, normalization=True)
object = sr.Mesh(
    object.vertices.repeat(len(cam_info), 1, 1), 
    object.faces.repeat(len(cam_info), 1, 1)
)
images = renderer.render_mesh(object) * 255
np.save(
    images.detach().cpu().numpy().astype(np.uint8), 
    parent + '/../data/output.npy'
)