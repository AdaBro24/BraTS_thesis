import torch
from torch import nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Multi-class Soft Dice Loss for training segmentation models."""

    def __init__(self, smooth: float = 1.0, include_background: bool = False):
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

        # Batch Dice: sum over the whole batch, so a class absent from one slice
        # does not yield a ~1.0 loss term for that slice
        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_one_hot, dims)
        cardinality = torch.sum(probs + target_one_hot, dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score.mean()

def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]  # (1, 1, size, size)


def masked_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    data_range: float = 6.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Mean SSIM over brain pixels. pred/target/mask: (B, 1, H, W). Returns a scalar in [-1, 1]."""
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    w = _gaussian_window(window_size, sigma, pred.device, pred.dtype)
    pad = window_size // 2

    def blur(x):
        return F.conv2d(x, w, padding=pad)

    mu_p, mu_t = blur(pred), blur(target)
    var_p = blur(pred * pred) - mu_p ** 2
    var_t = blur(target * target) - mu_t ** 2
    cov = blur(pred * target) - mu_p * mu_t

    ssim_map = ((2 * mu_p * mu_t + c1) * (2 * cov + c2)) / ((mu_p ** 2 + mu_t ** 2 + c1) * (var_p + var_t + c2))
    return (ssim_map * mask).sum() / mask.sum().clamp(min=1.0)


class MaskedL1SSIMLoss(nn.Module):
    """L1 + (1 - SSIM), both computed only inside the brain mask, for modality synthesis."""

    def __init__(self, l1_weight: float = 1.0, ssim_weight: float = 1.0, data_range: float = 6.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.data_range = data_range

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred, target = pred.float(), target.float()
        l1 = ((pred - target).abs() * mask).sum() / mask.sum().clamp(min=1.0)
        ssim = masked_ssim(pred, target, mask, data_range=self.data_range)
        return self.l1_weight * l1 + self.ssim_weight * (1.0 - ssim)


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


