import torch
from torch import nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Multi-class Soft Dice Loss for training segmentation models."""

    def __init__(self, smooth: float = 1e-6, include_background: bool = False):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: (batch, num_classes, H, W) - outputs from net
        target: (batch, H, W) - class labels (0, 1, 2, 3)
        """
        num_classes = pred.shape[1]
        probs = F.softmax(pred, dim=1)

        target_one_hot = F.one_hot(target, num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        if not self.include_background:
            probs = probs[:, 1:]
            target_one_hot = target_one_hot[:, 1:]

        dims = (2, 3)
        intersection = torch.sum(probs * target_one_hot, dims)
        cardinality = torch.sum(probs + target_one_hot, dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score.mean()

class DiceCELoss(nn.Module):
    """Cross Entropy + Dice Loss"""
    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(pred, target)
        loss_dice = self.dice(pred, target)
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice


