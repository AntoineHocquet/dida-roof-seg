import torch
import torch.nn as nn
import pytest

from dida_roofseg.model import (
    EncoderWrapper,
    DecoderUNetSmall,
    SegmentationModel,
)


@pytest.mark.parametrize("name,channels", [
    ("resnet18", [64, 64, 128, 256, 512]),
    ("resnet34", [64, 64, 128, 256, 512]),
    # Uncomment to test it; heavier and slower:
    # ("resnet50", [64, 256, 512, 1024, 2048]),
])
def test_encoder_feature_maps_and_channels(name, channels):
    encoder = EncoderWrapper(name=name, pretrained=False)
    assert hasattr(encoder, "feature_channels")
    assert list(encoder.feature_channels) == channels

    x = torch.randn(2, 3, 256, 256)  # B,C,H,W
    feats = encoder(x)
    assert isinstance(feats, tuple) and len(feats) == 5

    # shapes: (B, Ck, H/2^k, W/2^k) with c1 at H/2 (pre-maxpool), c5 at H/32
    c1, c2, c3, c4, c5 = feats
    B = x.size(0)
    assert c1.shape[0] == B and c1.shape[1] == channels[0]
    assert c2.shape[1] == channels[1]
    assert c3.shape[1] == channels[2]
    assert c4.shape[1] == channels[3]
    assert c5.shape[1] == channels[4]

    # Spatial scale checks
    H, W = x.shape[-2:]
    assert c1.shape[-2:] == (H // 2,  W // 2)
    assert c2.shape[-2:] == (H // 4,  W // 4)
    assert c3.shape[-2:] == (H // 8,  W // 8)
    assert c4.shape[-2:] == (H // 16, W // 16)
    assert c5.shape[-2:] == (H // 32, W // 32)


def test_encoder_freeze_unfreeze_toggles_requires_grad():
    enc = EncoderWrapper(name="resnet18", pretrained=False)

    # unfreeze by default
    assert any(p.requires_grad for p in enc.parameters())

    enc.freeze()
    assert all(p.requires_grad is False for p in enc.parameters())

    enc.unfreeze()
    assert all(p.requires_grad is True for p in enc.parameters())


def test_decoder_and_full_model_output_shape_matches_input():
    """
    End-to-end smoke test:
    - Build encoder/decoder/model
    - Ensure logits are (B,1,H,W)
    NOTE: If this fails because logits are (H/2,W/2), add a final upsample in your decoder
          to bring it back to the input resolution.
    """
    device = torch.device("cpu")
    x = torch.randn(2, 3, 256, 256, device=device)

    encoder = EncoderWrapper(name="resnet18", pretrained=False).to(device)
    decoder = DecoderUNetSmall(encoder_channels=encoder.feature_channels, out_channels=1, base_dec=32).to(device)
    model = SegmentationModel(encoder=encoder, decoder=decoder).to(device)

    logits = model(x)
    assert isinstance(logits, torch.Tensor)
    assert logits.dtype == torch.float32
    assert logits.shape[0] == x.shape[0]           # batch
    assert logits.shape[1] == 1                    # single-class logits
    # IMPORTANT: expect same spatial size as input
    assert logits.shape[-2:] == x.shape[-2:], (
        "Decoder output does not match input spatial size. "
        "If you're using c1 (H/2) as the final skip, add one more upsample "
        "stage (or a final upsample layer) to reach H×W."
    )


def test_backward_pass_runs():
    """
    Quick gradient check: forward + loss + backward should run without error.
    """
    x = torch.randn(2, 3, 256, 256)
    y = torch.randint(low=0, high=2, size=(2, 1, 256, 256)).float()

    encoder = EncoderWrapper(name="resnet18", pretrained=False)
    decoder = DecoderUNetSmall(encoder_channels=encoder.feature_channels, out_channels=1, base_dec=32)
    model = SegmentationModel(encoder=encoder, decoder=decoder)

    logits = model(x)
    assert logits.shape == y.shape

    loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()  # should not raise
