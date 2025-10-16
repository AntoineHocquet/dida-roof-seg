# src/dida_roofseg/predict.py

"""
Predict roof segmentation masks for test images.
Argparse CLI with sane defaults (no config files)
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dida_roofseg.dataset import RoofDataset
from dida_roofseg.engine import Predictor
from dida_roofseg.io import discover_pairs, save_mask
from dida_roofseg.seed import set_seed
from dida_roofseg import model as model_mod


def parse_args():
    p = argparse.ArgumentParser(description="Predict masks for 5 test images")
    p.add_argument("--data-dir", type=str, default="data/raw")
    p.add_argument("--ckpt-path", type=str, default="models/checkpoints/best.pth")
    p.add_argument("--pred-dir", type=str, default="outputs/predictions")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--encoder", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    #p.add_argument("--in-channels", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=True)

    labeled_images, mask_map, test_images = discover_pairs(args.data_dir)
    if len(test_images) == 0:
        raise RuntimeError("No test images found (images without masks).")
    test_ds = RoofDataset(mode="test", image_paths=test_images, image_size=args.image_size)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # Build the same model shape used in training
    encoder = model_mod.EncoderWrapper(name=args.encoder, pretrained=False)
    decoder = model_mod.DecoderUNetSmall(encoder_channels=encoder.feature_channels)
    model = model_mod.SegmentationModel(encoder=encoder, decoder=decoder)

    predictor = Predictor(model=model, ckpt_path=args.ckpt_path, device=args.device, threshold=args.threshold)

    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    for imgs, meta in test_loader:
        preds = predictor.predict_batch(imgs)  # (B,1,H,W)
        for i in range(preds.size(0)):
            filename = meta["filename"][i]
            # If you resized during inference, you can also resize back to orig here if needed.
            mask = preds[i, 0]  # (H,W)
            out_path = pred_dir / filename  # name match
            save_mask(mask, out_path)

    print(f"[done] Wrote predictions to: {pred_dir.resolve()}")


if __name__ == "__main__":
    main()
