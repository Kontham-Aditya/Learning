# calibrefine/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CalibRefineNet(nn.Module):
    def __init__(self):
        super(CalibRefineNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Assuming input image size is [3, 256, 832] — downsampled to [256, 832] via transforms
        self.feature_dim = 256 * 8 * 26  # (H/8 = 32, W/8 = 104)
        self.fc = nn.Linear(self.feature_dim, 512)  # Final image feature vector

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, 128, 416]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 128, 64, 208]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 256, 32, 104]
        x = x.view(x.size(0), -1)            # Flatten
        x = F.relu(self.fc(x))               # [B, 512]
        return x