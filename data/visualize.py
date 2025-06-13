# visualize.py
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def project_lidar_to_image(point_cloud, transform, calib_tensor, image):
    """
    Projects LiDAR points to image using the predicted transformation.

    point_cloud: [N, 4]
    transform: [6] → rotation (3) + translation (3)
    calib_tensor: [6] → fx, fy, cx, cy, tx, ty
    image: [H, W, 3] as numpy (RGB)
    """
    pc = point_cloud[:, :3].cpu().numpy()  # [N, 3]
    rot = transform[:3].cpu().numpy()
    trans = transform[3:].cpu().numpy()

    # Apply transformation
    pc = R.from_rotvec(rot).apply(pc) + trans

    # Calibration values from tensor
    fx = calib_tensor[0].item()
    fy = calib_tensor[1].item()
    cx = calib_tensor[2].item()
    cy = calib_tensor[3].item()
    tx = calib_tensor[4].item()
    ty = calib_tensor[5].item()

    # Project to 2D
    u = (pc[:, 0] * fx / pc[:, 2]) + cx
    v = (pc[:, 1] * fy / pc[:, 2]) + cy

    img = image.copy()
    for i in range(len(pc)):
        if 0 < u[i] < img.shape[1] and 0 < v[i] < img.shape[0] and pc[i, 2] > 0:
            cv2.circle(img, (int(u[i]), int(v[i])), 1, (0, 255, 0), -1)

    return img