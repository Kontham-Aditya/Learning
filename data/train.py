import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# === Local imports ===
from dataset.kitti_loader import KITTIRawDataset
from calibrefine.model import CalibRefineNet
from custom_collate import custom_collate_fn
from calibrefine.lidar_encoder import SimpleLidarEncoder
from calibrefine.iterative import IterativeRefinement
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
print("Calibration:", sample['calibration'])

# ====== Models ======
image_encoder = CalibRefineNet().to(DEVICE)
lidar_encoder = SimpleLidarEncoder().to(DEVICE)
model = IterativeRefinement(feature_dim=512, num_iterations=3).to(DEVICE)

# ====== Optimizer & Loss ======
optimizer = torch.optim.Adam(
    list(image_encoder.parameters()) +
    list(lidar_encoder.parameters()) +
    list(model.parameters()),
    lr=1e-3
)
criterion = nn.MSELoss()

# ====== Training Loop ======
for epoch in range(1, EPOCHS + 1):
    model.train()
    image_encoder.train()
    lidar_encoder.train()
    epoch_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}]"):
        images = batch['image'].to(DEVICE)                      # [B, 3, H, W]
        point_clouds = batch['point_cloud']                     # list of [N, 4] tensors

        # Extract features
        image_feat = image_encoder(images)                      # [B, 512]
        lidar_feat = lidar_encoder(point_clouds).to(DEVICE)     # [B, 512]

        # Predict transformation
        refined = model(image_feat, lidar_feat)                 # [B, 6]
        target = batch['calibration'].to(DEVICE)  # shape [B, 6]

        # Loss + backprop
        loss = criterion(refined, target)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"✅ Epoch [{epoch}/{EPOCHS}] Loss: {epoch_loss:.4f}")

    # =========================
    # 🔍 Visualization (after training)
    # =========================
    from visualize import project_lidar_to_image
    import matplotlib.pyplot as plt
    import numpy as np

    # Set models to evaluation mode
    image_encoder.eval()
    model.eval()

    # Get a batch from the dataloader
    batch = next(iter(train_loader))
    images = batch['image'].to(DEVICE)
    point_clouds = batch['point_cloud']
    calibs = batch['calibration']

    with torch.no_grad():
        image_feat = image_encoder(images)
        lidar_feat = lidar_encoder(point_clouds).to(DEVICE)
        preds = model(image_feat, lidar_feat)

    # Visualize the first sample
    img_np = (images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    vis_img = project_lidar_to_image(point_clouds[0], preds[0], calibs[0], img_np)

    # Show the projected LiDAR points
    plt.figure(figsize=(10, 6))
    plt.imshow(vis_img)
    plt.title("LiDAR projected onto Image (Predicted Pose)")
    plt.axis("off")
    plt.show()