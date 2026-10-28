from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class BratsDataset(Dataset):
    """PyTorch Dataset for loading 2D BraTS MRI slices (.pt files)."""
    def __init__(self, split_dir: Path | str, transform=None):
        self.split_dir = Path(split_dir)
        self.files = sorted(self.split_dir.rglob("*.pt"))
        self.transform = transform

        if len(self.files) == 0:
            raise FileNotFoundError(f"No .pt files found in {self.split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Fast get method
        data = torch.load(self.files[idx], weights_only=True)
        image = data["image"]
        mask = data["mask"]

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask


MODALITIES = ("T1", "T1CE", "T2", "FLAIR")  # channel order of the "image" tensor


class ModalitySynthesisDataset(BratsDataset):
    """Variant for missing-modality synthesis.

    Returns (input, target, brain_mask): `target` is the channel `target_modality` (1, H, W),
    `input` is the remaining 3 channels (3, H, W) in their original order, `brain_mask`
    (1, H, W) float is 1 where any input channel is non-zero (background is exactly 0.0
    after z-scoring, so this recovers the brain without needing the raw volume).
    """
    def __init__(self, split_dir: Path | str, target_modality: str | int, transform=None):
        super().__init__(split_dir, transform)
        if isinstance(target_modality, str):
            target_modality = MODALITIES.index(target_modality.upper())
        self.target_idx = target_modality
        self.input_idx = [i for i in range(len(MODALITIES)) if i != target_modality]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = torch.load(self.files[idx], weights_only=True)
        image = data["image"]

        if self.transform is not None:
            image, _ = self.transform(image, data["mask"])

        inputs = image[self.input_idx]
        brain_mask = (inputs != 0).any(dim=0, keepdim=True).float()
        return inputs, image[self.target_idx : self.target_idx + 1], brain_mask


def create_dataloader(
        split_dir: Path | str,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        prefetch_factor: int = 4,
        target_modality: str | int | None = None
        ) -> DataLoader:

    """Creates a PyTorch DataLoader for given dataset split.

    With `target_modality` (e.g. "T1CE") yields (3-channel input, 1-channel target,
    brain mask) for modality synthesis instead of (image, mask) for segmentation.
    """
    if target_modality is None:
        dataset = BratsDataset(split_dir)
    else:
        dataset = ModalitySynthesisDataset(split_dir, target_modality)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )