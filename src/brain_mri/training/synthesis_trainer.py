from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from brain_mri.models.losses import masked_ssim


class SynthesisTrainer:
    """Training, validation and checkpointing for missing-modality synthesis (input, target, brain_mask)."""

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            device: torch.device | str = 'cuda' if torch.cuda.is_available() else 'cpu',
            checkpoint_dir: Path | str = 'checkpoints',
            checkpoint_name: str = 'best_generator.pt',
            scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
            data_range: float = 6.0,
            ):
        self.device = torch.device(device)
        self.model = model.to(self.device, memory_format=torch.channels_last)
        self.use_amp = self.device.type == "cuda"
        torch.backends.cudnn.benchmark = True
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_range = data_range
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_name = checkpoint_name

        self.best_val_ssim = -1.0
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_ssim": [],
            "val_psnr": [],
            "lr": [],
        }

    def _forward(self, inputs, target, mask):
        inputs = inputs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
        target = target.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            output = self.model(inputs)
        output = output.float()
        return output, target, mask, self.criterion(output, target, mask)

    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for inputs, target, mask in pbar:
            self.optimizer.zero_grad(set_to_none=True)
            _, _, _, loss = self._forward(inputs, target, mask)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def validate_epoch(self, loader: DataLoader | None = None) -> tuple[float, dict[str, float]]:
        """Returns (loss, {"ssim", "psnr", "mae"}), all computed inside the brain mask."""
        loader = loader if loader is not None else self.val_loader
        self.model.eval()
        running_loss = 0.0
        sum_sq_err = sum_abs_err = n_pixels = 0.0
        ssim_sum = 0.0

        for inputs, target, mask in tqdm(loader, desc="Evaluation", leave=False):
            output, target, mask, loss = self._forward(inputs, target, mask)
            running_loss += loss.item()

            diff = (output - target) * mask
            sum_sq_err += (diff ** 2).sum().item()
            sum_abs_err += diff.abs().sum().item()
            n_pixels += mask.sum().item()
            ssim_sum += masked_ssim(output, target, mask, data_range=self.data_range).item()

        mse = sum_sq_err / max(n_pixels, 1.0)
        psnr = 10.0 * torch.log10(torch.tensor(self.data_range ** 2 / max(mse, 1e-12))).item()
        metrics = {"ssim": ssim_sum / len(loader), "psnr": psnr, "mae": sum_abs_err / max(n_pixels, 1.0)}
        return running_loss / len(loader), metrics

    def fit(self, epochs: int = 30) -> dict[str, list[float]]:
        print(f"Starting training on device: {self.device} for {epochs} epochs\n")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss, m = self.validate_epoch()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_ssim"].append(m["ssim"])
            self.history["val_psnr"].append(m["psnr"])
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"SSIM: {m['ssim']:.4f} | PSNR: {m['psnr']:.2f} dB | MAE: {m['mae']:.4f} | "
                f"LR: {self.history['lr'][-1]:.2e}"
            )

            if self.scheduler is not None:
                self.scheduler.step()

            if m["ssim"] > self.best_val_ssim:
                self.best_val_ssim = m["ssim"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_ssim": self.best_val_ssim,
                    },
                    self.checkpoint_dir / self.checkpoint_name,
                )
                print(f"Saved new best model checkpoint, SSIM : {self.best_val_ssim:.4f}")

        print(f"Training completed, Best Validation SSIM : {self.best_val_ssim:.4f}")
        return self.history
