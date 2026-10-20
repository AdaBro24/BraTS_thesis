from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from brain_mri.models.metrics import compute_dice_stats, dice_from_stats

class Trainer:
    """Manages the training, validation and checkpointing"""

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            device: torch.device | str = 'cuda' if torch.cuda.is_available() else 'cpu',
            checkpoint_dir: Path | str = 'checkpoints',
            checkpoint_name: str = 'best_unet.pt',
            scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
            ):
                
                self.device = torch.device(device)
                self.model = model.to(self.device, memory_format=torch.channels_last)
                # bf16 autocast on GPU: tensor cores, ~half the activation memory, no GradScaler needed
                self.use_amp = self.device.type == "cuda"
                torch.backends.cudnn.benchmark = True
                self.train_loader = train_loader
                self.val_loader = val_loader
                self.criterion = criterion
                self.optimizer = optimizer
                self.scheduler = scheduler
                self.checkpoint_dir = Path(checkpoint_dir)
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                self.checkpoint_name = checkpoint_name

                self.best_val_dice = 0.0
                self.history: dict[str, list[float]] = {
                    "train_loss": [],
                    "val_loss": [],
                    "val_mean_dice": [],
                    "lr": [],
                }        

    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, masks in pbar:
            images = images.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            masks = masks.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss


    @torch.no_grad()
    def validate_epoch(self, loader: DataLoader | None = None) -> tuple[float, dict[str, float]]:
        """Runs evaluation without tracking gradients; defaults to the validation loader."""
        loader = loader if loader is not None else self.val_loader
        self.model.eval()
        running_loss = 0.0
        total_intersection = None
        total_voxels = None

        pbar = tqdm(loader, desc="Evaluation", leave=False)
        for images, masks in pbar:
            images = images.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            masks = masks.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            running_loss += loss.item()

            # Accumulate counts over the whole set (global Dice), not per-batch averages
            intersection, voxels = compute_dice_stats(outputs, masks)
            if total_intersection is None:
                total_intersection, total_voxels = intersection, voxels
            else:
                total_intersection += intersection
                total_voxels += voxels

        val_loss = running_loss / len(loader)
        val_dice = dice_from_stats(total_intersection, total_voxels)

        return val_loss, val_dice

    def fit(self, epochs: int = 30) -> dict[str, list[float]]:
        """Runs the full training loop across the specified number of epochs."""
        print(f"Starting training on device: {self.device} for {epochs} epochs\n")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_dice = self.validate_epoch()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_mean_dice"].append(val_dice["mean"])
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            dice_details = ", ".join(f"{k}: {v:.2f}" for k, v in val_dice.items() if k != "mean")
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Mean Dice: {val_dice['mean']:.4f} | "
                f"LR: {self.history['lr'][-1]:.2e} "
                f"({dice_details})"
            )

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_dice["mean"])
                else:
                    self.scheduler.step()

            if val_dice["mean"] > self.best_val_dice:
                self.best_val_dice = val_dice["mean"]
                best_model_path = self.checkpoint_dir / self.checkpoint_name
                torch.save(
                    {
                        "epoch" : epoch,
                        "model_state_dict" : self.model.state_dict(),
                        "optimizer_state_dict" : self.optimizer.state_dict(),
                        "best_dice" : self.best_val_dice,
                    },
                    best_model_path,
                )
                print(f"Saved new best model checkpoint, Dice : {self.best_val_dice:.4f}")

        print(f"Training completed, Best Validation Mean Dice : {self.best_val_dice:.4f}")
        return self.history