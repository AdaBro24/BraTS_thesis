from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from brain_mri.models.metrics import compute_dice_score

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
            checkpoint_dir: Path | str = 'checkpoints'
            ):
                
                self.device = torch.device(device)
                self.model = model.to(self.device)
                self.train_loader = train_loader
                self.val_loader = val_loader
                self.criterion = criterion
                self.optimizer = optimizer
                self.checkpoint_dir = Path(checkpoint_dir)
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

                self.best_val_dice = 0.0
                self.history: dict[str, list[float]] = {
                    "train_loss": [],
                    "val_loss": [],
                    "val_mean_dice": [],
                }        

    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss


    @torch.no_grad()
    def validate_epoch(self) -> tuple[float, dict[str, float]]:
        """Runs validation over the validation dataset without tracking gradients."""
        self.model.eval()
        running_loss = 0.0
        total_dice_scores : dict[str, float] = {"NCR": 0.0, "ED": 0.0, "ET": 0.0, "mean": 0.0}

        pbar = tqdm(self.val_loader, desc="Validation", leave=False)
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            running_loss += loss.item()

            batch_dice = compute_dice_score(outputs, masks)
            for key in total_dice_scores:
                total_dice_scores[key] += batch_dice[key]

        val_loss = running_loss / len(self.val_loader)

        avg_dice = {k: round(v / len(self.val_loader), 4) for k, v in total_dice_scores.items()}
            
        return val_loss, avg_dice

    def fit(self, epochs: int = 30) -> dict[str, list[float]]:
        """Runs the full training loop across the specified number of epochs."""
        print(f"Starting training on device: {self.device} for {epochs} epochs\n")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_dice = self.validate_epoch()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_mean_dice"].append(val_dice["mean"])

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Mean Dice: {val_dice['mean']:.4f} "
                f"(NCR: {val_dice['NCR']:.2f}, ED: {val_dice['ED']:.2f}, ET: {val_dice['ET']:.2f})"
            )

            if val_dice["mean"] > self.best_val_dice:
                self.best_val_dice = val_dice["mean"]
                best_model_path = self.checkpoint_dir / "best_unet.pt"
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