import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from scipy.spatial.transform import Rotation as R  # put this at the top of file


class KITTIRawDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "image_02", "data")
        self.lidar_dir = os.path.join(root_dir, "velodyne_points", "data")
        self.calib_file = self.calib_file = r"C:\Users\radit\Downloads\2011_09_26_calib\2011_09_26\calib_velo_to_cam.txt"

        self.image_files = sorted(os.listdir(self.image_dir))
        self.lidar_files = sorted(os.listdir(self.lidar_dir))

        # Define default image transform (resize + normalization)
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 832)),  # Resizing to smaller than KITTI 375x1242 for speed
            transforms.ToTensor(),  # Converts to [C, H, W] with values in [0, 1]
        ])

        self.calib = self.load_calib_file(self.calib_file)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load and preprocess image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        # Load and convert LiDAR
        lidar_path = os.path.join(self.lidar_dir, self.lidar_files[idx])
        point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        point_cloud = torch.from_numpy(point_cloud).float()

        return {
            "image": image,
            "point_cloud": point_cloud,
            "calibration": torch.tensor(self.calib['6dof'], dtype=torch.float32)
        }

    def load_calib_file(self, calib_path):
        calib = {}
        with open(calib_path, 'r') as f:
            for line in f:
                if ':' not in line:
                    continue
                key, value = line.strip().split(':', 1)
                try:
                    calib[key] = np.array([float(x) for x in value.strip().split()])
                except ValueError:
                    continue

        # Convert rotation matrix to rotation vector (3 values)
        R_mat = calib['R'].reshape(3, 3)
        T_vec = calib['T']
        r = R.from_matrix(R_mat).as_rotvec()  # (rx, ry, rz)
        calib['6dof'] = np.concatenate([r, T_vec])  # [rx, ry, rz, tx, ty, tz]
        return calib