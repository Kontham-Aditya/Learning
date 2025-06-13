import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformationLoss(nn.Module):
    def __init__(self, rotation_weight=1.0, translation_weight=1.0):
        super(TransformationLoss, self).__init__()
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight

    def forward(self, pred, target):
        """
        pred and target shape: [B, 6] → [rx, ry, rz, tx, ty, tz]
        """
        # Split rotation and translation
        pred_rot, pred_trans = pred[:, :3], pred[:, 3:]
        target_rot, target_trans = target[:, :3], target[:, 3:]

        # Use L2 loss for both (can be modified for better performance)
        loss_rot = F.mse_loss(pred_rot, target_rot)
        loss_trans = F.mse_loss(pred_trans, target_trans)

        total_loss = self.rotation_weight * loss_rot + self.translation_weight * loss_trans
        return total_loss