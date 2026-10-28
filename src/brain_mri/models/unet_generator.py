from brain_mri.models.unet import UNet


class Unet_Generator(UNet):
    """U-Net for missing-modality synthesis: 3 available MRI modalities -> 1 missing one.

    Same architecture as UNet; the output is a linear 1x1 conv (no activation) because
    z-scored intensities are unbounded (roughly -3..+3).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_features: int = 64):
        super().__init__(in_channels=in_channels, num_classes=out_channels, base_features=base_features)
