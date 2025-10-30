# src/dida_roofseg/model.py
"""
Model definitions for roof segmentation.
Description: This module defines the segmentation model architecture,
 including an encoder wrapper (e.g., ResNet) and a small UNet-style decoder.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights # assumes torchvision>=0.13

class EncoderWrapper(nn.Module):
    """
    ResNet encoder wrapper that returns UNet-style multi-scale features.
    Returns (in forward):
        features: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
            (c1, c2, c3, c4, c5)
            c1 = output of stem (after relu, BEFORE maxpool)
            c2 = layer1
            c3 = layer2
            c4 = layer3
            c5 = layer4  (deepest, smallest spatial resolution)

    Attributes:
        feature_channels: list[int]
            Number of channels for each feature map (len == 5).
            Useful for configuring your decoder.

    Notes:
        - For ResNet-18/34 (BasicBlock), channels are [64, 64, 128, 256, 512].
        - For ResNet-50 (Bottleneck), channels are [64, 256, 512, 1024, 2048].
        """
    
    def __init__(
            self,
            name: str = "resnet18",
            pretrained: bool = True # if True, use ImageNet weights
        ):
        super().__init__()

        name = name.lower() # ensures case-insensitive
        supported = {"resnet18", "resnet34", "resnet50"}
        if name not in supported:
            raise ValueError(f"Unsupported encoder '{name}'. Supported: {supported}")
        
        # load backbone
        if name == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet18(weights=weights)
            block_type = "basic"
        elif name == "resnet34":
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet34(weights=weights)
            block_type = "basic"
        else: # resnet50
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet50(weights=weights)
            block_type = "bottleneck"

        # ---- record channel dims for decoder ----
        if block_type == "basic":
            self.feature_channels = [64, 64, 128, 256, 512] # resnet18/34
        else: # bottleneck
            self.feature_channels = [64, 256, 512, 1024, 2048] # resnet50

        # ---- expose the prats needed for UNet-style ----
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        ) # NOTE: no maxpool here (we return pre-maxpool featurs as c1)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # keep a handle to the whole thing (not used directly)
        self.backbone = backbone

    def freeze(self) -> None:
        """
        Freeze all encoder parameters (used for warm-up).
        """
        for p in self.parameters():
            p.requires_grad = False
        self.eval() # set to eval mode

    def unfreeze(self) -> None:
        """
        Unfreeze all encoder parameters (fine-tuning).
        """
        for p in self.parameters():
            p.requires_grad = True
        # we do NOT force train() here; Trainer will set model.train()/eval()

    def forward(self, x: Tensor):
        """
        Returns multi-scale features for a UNet-style decoder.

        c1: output of stem (H/2 if conv1 stride=2, but BEFORE maxpool)
        c2: layer1 (after maxpool -> H/4)
        c3: layer2 (H/8)
        c4: layer3 (H/16)
        c5: layer4 (H/32)
        """
        c1 = self.stem(x)      # (B,64,H/2,W/2) for standard ResNet
        x = self.maxpool(c1) # (B,64,H/4,W/4)
        c2 = self.layer1(x)   # (B,64|256,...)
        c3 = self.layer2(c2)  # (B,128|512,...)
        c4 = self.layer3(c3)  # (B,256|1024,...)
        c5 = self.layer4(c4)  # (B,512|2048,...)
        return (c1, c2, c3, c4, c5)


# -------------------------
# Small building-blocks
# -------------------------
class ConvBNReLU(nn.Module):
    """(Conv → BN → ReLU) x 2, a tiny head/body used throughout the decoder."""
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int | None = None):
        super().__init__()
        mid = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """
    Upsample by 2 (bilinear) -> reduce channels -> concat skip ->ConvBNReLU.
    Using bilinear+conv (instead of ConvTranspose2d) to keep it lightweight.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.fuse = ConvBNReLU(out_ch + skip_ch, out_ch)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.upsample(x)      # upsample by 2
        x = self.reduce(x)        # reduce channels
        # assumes encoder/decoder spatial sizes match;
        # we resize inputs to a multiple of 32 in the pipeline
        x = torch.cat([x, skip], dim=1) # concat skip connection
        x = self.fuse(x)          # fuse
        return x

# -------------------------
# Decoder (small UNet-style)
# -------------------------
class DecoderUNetSmall(nn.Module):
    """
    A compact UNet decoder that expects the 5 feature maps from the encoder:
      (c1, c2, c3, c4, c5) with channels given by encoder.feature_channels.

    It upsamples step-by-step:
      c5 → (up with c4) → (up with c3) → (up with c2) → (up with c1) → 1×H×W logits
    """
    def __init__(
        self,
        encoder_channels: list[int] | None,
        out_channels: int = 1,
        base_dec: int = 64
    ):
        """
        encoder_channels: e.g.
          - ResNet-18/34: [64, 64, 128, 256, 512]
          - ResNet-50:    [64, 256, 512, 1024, 2048]
        base_dec: controls decoder width (keep small for CPU).
        """
        super().__init__()
        if len(encoder_channels) != 5:
            raise ValueError("Expected encoder_channels to have length 5 for (c1..c5).")
        
        c1, c2, c3, c4, c5 = encoder_channels # from shallow to deep

        # choose small widths for faster training
        d5 = base_dec * 8  # deepest, from c5
        d4 = base_dec * 4
        d3 = base_dec * 2
        d2 = base_dec
        d1 = base_dec

        # “Bridge” to map c5 to a manageable width before first upsample
        self.bridge = ConvBNReLU(in_ch=c5, out_ch=d5)

        # Up blocks with skips from encoder
        self.up4 = UpBlock(in_ch=d5, skip_ch=c4, out_ch=d4)  # c5→c4
        self.up3 = UpBlock(in_ch=d4, skip_ch=c3, out_ch=d3)  # c4→c3
        self.up2 = UpBlock(in_ch=d3, skip_ch=c2, out_ch=d2)  # c3→c2
        self.up1 = UpBlock(in_ch=d2, skip_ch=c1, out_ch=d1)  # c2→c1

        # Final 1×1 conv to logits (no sigmoid here)
        self.head = nn.Conv2d(d1, out_channels, kernel_size=1)

        # NEW: final upsample to go from H/2 → H
        self.final_upsample = nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False)

        

    def forward(self, features: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> Tensor | None:
        """
        features: expect tuple (c1, c2, c3, c4, c5)
        Return logits of shape (B,1,H,W). No sigmoid here.
        """
        if len(features) != 5:
            raise ValueError("Expected features to be a tuple of length 5: (c1..c5).")
        c1, c2, c3, c4, c5 = features

        x = self.bridge(c5)
        x = self.up4(x, c4)
        x = self.up3(x, c3)
        x = self.up2(x, c2)
        x = self.up1(x, c1)
        x = self.final_upsample(x)          # ← bring back to input size
        logits = self.head(x)               # (B,1,H,W)
        return logits

# -------------------------
# Full model wrapper
# -------------------------
class SegmentationModel(nn.Module):
    """
    Wraps an EncoderWrapper and a DecoderUNetSmall into one model.
    Forward: x -> encoder(x) -> decoder(features) -> logits (B,1,H,W)
    """
    def __init__(self, encoder: EncoderWrapper, decoder: DecoderUNetSmall):
        super().__init__()
        # keep encoder as attribute for Trainer to access
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: Tensor) -> Tensor | None:
        """Return logits (B,1,H,W)."""
        feats = self.encoder(x) # tuple (c1..c5)
        logits = self.decoder(feats) # (B,1,H,W)
        return logits