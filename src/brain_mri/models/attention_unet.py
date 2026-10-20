import torch
import torch.nn.functional as F
from torch import nn

from brain_mri.models.unet import DoubleConv


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018, "Attention U-Net").

    The gating signal comes from the coarser decoder level (half the resolution of the skip):
        alpha = sigmoid(psi(relu(W_x(x) + W_g(g))))
    W_x downsamples the skip to the gate's resolution, and alpha is upsampled back
    to weight the skip before it is concatenated in the decoder.
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        # W_g: 1x1 conv on the gating signal (coarse level)
        self.w_gate = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True)
        # W_x: strided conv on the skip, brings it down to the gate's resolution
        self.w_skip = nn.Conv2d(skip_channels, inter_channels, kernel_size=2, stride=2, bias=False)
        # psi: 1 attention coefficient per pixel
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # gate: [batch, C_g, H/2, W/2], skip: [batch, C_x, H, W]
        attention = self.psi(self.relu(self.w_gate(gate) + self.w_skip(skip)))  # [batch, 1, H/2, W/2]
        attention = F.interpolate(attention, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return skip * attention  # [batch, C_x, H, W]


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 5, base_features: int = 64):
        super().__init__()
        f = base_features  # 64

        # ENCODER (4 -> 64 -> 128 -> 256 -> 512):
        self.inc = DoubleConv(in_channels, f)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f, f * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 2, f * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 4, f * 8))

        # BOTTLENECK (512 -> 1024)
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 8, f * 16))

        # DECODER (each level: attention gate on skip, gated by the coarser level -> upsample -> concat -> DoubleConv)
        self.up1 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.att1 = AttentionGate(gate_channels=f * 16, skip_channels=f * 8, inter_channels=f * 4)
        self.conv1 = DoubleConv(f * 16, f * 8)

        self.up2 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.att2 = AttentionGate(gate_channels=f * 8, skip_channels=f * 4, inter_channels=f * 2)
        self.conv2 = DoubleConv(f * 8, f * 4)

        self.up3 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.att3 = AttentionGate(gate_channels=f * 4, skip_channels=f * 2, inter_channels=f)
        self.conv3 = DoubleConv(f * 4, f * 2)

        self.up4 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.att4 = AttentionGate(gate_channels=f * 2, skip_channels=f, inter_channels=f // 2)
        self.conv4 = DoubleConv(f * 2, f)

        # END LAYER
        self.outc = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder (saving x1, x2, x3, x4 copies as skip connections)
        x1 = self.inc(x)        # [batch, 64, 224, 224]
        x2 = self.down1(x1)     # [batch, 128, 112, 112]
        x3 = self.down2(x2)     # [batch, 256, 56, 56]
        x4 = self.down3(x3)     # [batch, 512, 28, 28]

        x5 = self.bottleneck(x4)  # [batch, 1024, 14, 14]

        # Decoder (skip gated by the coarser level, then upsampling + concatenation)
        a4 = self.att1(x5, x4)                 # gate x5 [batch, 1024, 14, 14] -> [batch, 512, 28, 28]
        d1 = torch.cat([a4, self.up1(x5)], dim=1)  # -> [batch, 1024, 28, 28]
        d1 = self.conv1(d1)                    # -> [batch, 512, 28, 28]

        a3 = self.att2(d1, x3)                 # gate d1 [batch, 512, 28, 28] -> [batch, 256, 56, 56]
        d2 = torch.cat([a3, self.up2(d1)], dim=1)  # -> [batch, 512, 56, 56]
        d2 = self.conv2(d2)                    # -> [batch, 256, 56, 56]

        a2 = self.att3(d2, x2)                 # gate d2 [batch, 256, 56, 56] -> [batch, 128, 112, 112]
        d3 = torch.cat([a2, self.up3(d2)], dim=1)  # -> [batch, 256, 112, 112]
        d3 = self.conv3(d3)                    # -> [batch, 128, 112, 112]

        a1 = self.att4(d3, x1)                 # gate d3 [batch, 128, 112, 112] -> [batch, 64, 224, 224]
        d4 = torch.cat([a1, self.up4(d3)], dim=1)  # -> [batch, 128, 224, 224]
        d4 = self.conv4(d4)                    # -> [batch, 64, 224, 224]

        outputs = self.outc(d4)                         # -> [batch, 5, 224, 224]
        return outputs
