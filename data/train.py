import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm  # optional progress bar

from dataset.kitti_loader import KITTIRawDataset
from calibrefine.model import CalibRefineNet
from custom_collate import custom_collate_fn

# ====== Configuration ======
DATA_ROOT = r"C:\Users\radit\Downloads\2011_09_26_drive_0001_sync\2011_09_26\2011_09_26_drive_0001_sync"
BATCH_SIZE = 4
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Transforms ======
image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 832)),
    transforms.ToTensor()
])

# ====== Dataset & DataLoader ======
dataset = KITTIRawDataset(root_dir=DATA_ROOT, transform=image_transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)

# ====== Sanity Check ======
print("✅ Running sanity check...")
sample = dataset[0]
print("Image shape:", sample['image'].shape)
print("Point cloud shape:", sample['point_cloud'].shape)
print("Calibration keys:", sample['calibration'].keys())

# ====== Models ======
image_encoder = CalibRefineNet().to(DEVICE)
model = nn.Linear(524, 6).to(DEVICE)  # 512 (image) + 6 (dummy lidar) + 6 (dummy init pose)

# ====== Optimizer & Loss ======
optimizer = torch.optim.Adam(list(image_encoder.parameters()) + list(model.parameters()), lr=1e-3)
criterion = nn.MSELoss()

# ====== Training Loop ======
for epoch in range(1, EPOCHS + 1):
    model.train()
    image_encoder.train()
    epoch_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}]")  # progress bar

    for batch in loop:
        images = batch['image'].to(DEVICE)
        B = images.shape[0]

        image_feat = image_encoder(images)  # [B, 512]
        lidar_feat = torch.zeros((B, 6)).to(DEVICE)       # Dummy lidar features
        init_pose = torch.zeros((B, 6)).to(DEVICE)        # Dummy initial transform

        fused_feat = torch.cat([image_feat, lidar_feat, init_pose], dim=1)  # [B, 524]
        refined = model(fused_feat)  # [B, 6]

        target = torch.zeros_like(refined).to(DEVICE)     # Dummy ground truth
        loss = criterion(refined, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"✅ Epoch [{epoch}/{EPOCHS}] completed — Avg Loss: {epoch_loss / len(train_loader):.6f}")