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


def create_dataloader(
        split_dir: Path | str,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 2,
        pin_memory: bool = True
        ) -> DataLoader:
    
    """Creates a PyTorch DataLoader for given dataset split"""
    dataset = BratsDataset(split_dir)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )