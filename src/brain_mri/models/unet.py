import torch
from torch import nn


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm2d -> ReLU) x 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            # First round
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Second round
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 4, base_features: int = 64):
        super().__init__()
        f = base_features  # 64

        # ENCODER (4 -> 64 -> 128 -> 256 -> 512):
        self.inc = DoubleConv(in_channels, f)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f, f * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 2, f * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 4, f * 8))

        # BOTTLENECK (512 -> 1024)
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 8, f * 16))

        # DECODER
        self.up1 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(f * 16, f * 8)

        self.up2 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(f * 8, f * 4)

        self.up3 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(f * 4, f * 2)

        self.up4 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
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

        # Decoder (upsampling + concatenating with skip connections)
        d1 = self.up1(x5)                    # -> [batch, 512, 28, 28]
        d1 = torch.cat([x4, d1], dim=1)      # -> [batch, 1024, 28, 28]
        d1 = self.conv1(d1)                  # -> [batch, 512, 28, 28]

        d2 = self.up2(d1)                    # -> [batch, 256, 56, 56]
        d2 = torch.cat([x3, d2], dim=1)      # -> [batch, 512, 56, 56]
        d2 = self.conv2(d2)                  # -> [batch, 256, 56, 56]

        d3 = self.up3(d2)                    # -> [batch, 128, 112, 112]
        d3 = torch.cat([x2, d3], dim=1)      # -> [batch, 256, 112, 112]
        d3 = self.conv3(d3)                  # -> [batch, 128, 112, 112]

        d4 = self.up4(d3)                    # -> [batch, 64, 224, 224]
        d4 = torch.cat([x1, d4], dim=1)      # -> [batch, 128, 224, 224]
        d4 = self.conv4(d4)                  # -> [batch, 64, 224, 224]

        outputs = self.outc(d4)               # -> [batch, 4, 224, 224]
        return outputs