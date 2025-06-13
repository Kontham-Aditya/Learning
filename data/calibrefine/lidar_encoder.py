import torch
import torch.nn as nn

class SimpleLidarEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super(SimpleLidarEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 64),       # Each point has 4 features: x, y, z, intensity
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.ReLU()
        )

    def forward(self, batch_point_clouds):
        """
        batch_point_clouds: list of [N_i, 4] tensors (B elements)
        Returns: tensor [B, output_dim]
        """
        features = []
        for pc in batch_point_clouds:
            # Encode each point, then average over points
            point_feat = self.encoder(pc)       # [N_i, output_dim]
            pooled_feat = point_feat.mean(dim=0)  # [output_dim]
            features.append(pooled_feat)
        return torch.stack(features)  # [B, output_dim]