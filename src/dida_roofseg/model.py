# expected interface (you will implement)

import torch
import torch.nn as nn

class EncoderWrapper(nn.Module):
    def __init__(self, name: str = "resnet18", pretrained: bool = True, in_channels: int = 3):
        ...

    def freeze(self) -> None:
        ...

    def unfreeze(self) -> None:
        ...

    def forward(self, x: torch.Tensor):
        """
        Return feature maps (for UNet-style) OR a single bottleneck.
        You decide; just make sure Decoder matches this contract.
        """
        ...

class DecoderUNetSmall(nn.Module):
    def __init__(self, encoder_channels: ...):
        ...

    def forward(self, features_or_bottleneck):
        """
        Return logits of shape (B,1,H,W). No sigmoid here.
        """
        ...

class SegmentationModel(nn.Module):
    def __init__(self, encoder: EncoderWrapper, decoder: DecoderUNetSmall):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits (B,1,H,W)."""
        ...
