import torch
import torch.nn as nn
import torch.nn.functional as F

class IterativeRefinement(nn.Module):
    def __init__(self, feature_dim=512, num_iterations=3):
        super(IterativeRefinement, self).__init__()
        self.num_iterations = num_iterations

        self.refine_layer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim * 2 + 6, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 6)  # 3 rotation + 3 translation
            ) for _ in range(num_iterations)
        ])

    def forward(self, image_feat, lidar_feat, init_transform=None):
        """
        image_feat, lidar_feat: [B, feature_dim]
        init_transform: [B, 6] initial pose (optional, defaults to 0s)
        """
        B = image_feat.size(0)
        if init_transform is None:
            transform = torch.zeros((B, 6), device=image_feat.device)
        else:
            transform = init_transform

        for i in range(self.num_iterations):
            fused = torch.cat([image_feat, lidar_feat, transform], dim=1)
            delta = self.refine_layer[i](fused)
            transform = transform + delta  # refine

        return transform  # Final transformation prediction