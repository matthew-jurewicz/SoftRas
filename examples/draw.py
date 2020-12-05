import sys
import soft_renderer as sr
import os
import torch
import numpy as np

file = sys.argv[1]
renderer = sr.SoftRenderer(
    image_size=64, camera_mode='look_at', viewing_angle=15)
parent = os.path.dirname(os.path.abspath(__file__))
cam_info = np.load(parent + '/../data/camera.npy')
cam_info = torch.from_numpy(cam_info.astype(np.float32))
object = sr.Mesh.from_obj(file, normalization=True)
object = sr.Mesh(
    object.vertices.repeat(10, 1, 1), 
    object.faces.repeat(10, 1, 1)
)
images = np.empty((len(cam_info), 4, 64, 64), dtype=np.uint8)
for i in range(0, len(cam_info), 10):
    renderer.transform.set_eyes_from_angles(
        distances=cam_info[i:i + 10,0], 
        elevations=cam_info[i:i + 10,1], 
        azimuths=cam_info[i:i + 10,2]
    )
    tmp =  renderer.render_mesh(object) * 255
    images[i:i + 10] = tmp.detach().cpu().numpy().astype(np.uint8)
np.save(images, parent + '/../data/output.npy')