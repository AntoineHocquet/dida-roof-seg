# src/dida_roofseg/cli.py
"""
Unified CLI for training and predicting roof segmentation.
Usage:
  python -m dida_roofseg.cli train   [args...]
  python -m dida_roofseg.cli predict [args...]

Entry point for 'dida-roofseg' is in pyproject.toml under [project.scripts]:
dida-roofseg = "dida_roofseg.cli:main"
"""

from __future__ import annotations
import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from dida_roofseg.dataset import RoofDataset
from dida_roofseg.engine import Trainer, TrainConfig, Predictor
from dida_roofseg.io import discover_pairs, train_val_split, save_mask
from dida_roofseg.seed import set_seed
from dida_roofseg import model as model_mod  # model.py


# -------------------------
# Shared helpers
# -------------------------

def add_shared_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--encoder", type=str, default="resnet18",
                   choices=["resnet18", "resnet34", "resnet50"],
                   help="Backbone encoder.")
    p.add_argument("--image-size", type=int, default=256, # should be a multiple of 32 (original images are 256x256)
                   help="Resize (square) side. Use same at train/predict for consistency.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Binarization threshold for metrics/predictions.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device string, e.g. 'cpu' or 'cuda'.")


def build_model(encoder_name: str, pretrained: bool) -> model_mod.SegmentationModel:
    encoder = model_mod.EncoderWrapper(name=encoder_name, pretrained=pretrained)
    decoder = model_mod.DecoderUNetSmall(encoder_channels=encoder.feature_channels)
    return model_mod.SegmentationModel(encoder=encoder, decoder=decoder)


# -------------------------
# Subcommand: train
# -------------------------

def configure_train_parser(subparsers) -> None:
    tp = subparsers.add_parser("train", help="Train roof segmentation.")
    tp.add_argument("--data-dir", type=str, default="data/raw",
                    help="Directory containing images and masks.")
    tp.add_argument("--ckpt-dir", type=str, default="models/checkpoints",
                    help="Directory to save checkpoints.")
    tp.add_argument("--epochs", type=int, default=20)
    tp.add_argument("--batch-size", type=int, default=4)
    tp.add_argument("--lr-encoder", type=float, default=1e-4)
    tp.add_argument("--lr-decoder", type=float, default=1e-3)
    tp.add_argument("--weight-decay", type=float, default=1e-4)
    tp.add_argument("--freeze-epochs", type=int, default=3,
                    help="Freeze encoder for first N epochs.")
    tp.add_argument("--val-ratio", type=float, default=0.2)

    add_shared_model_args(tp)
    tp.set_defaults(func=run_train)


def run_train(args: argparse.Namespace) -> None:
    set_seed(args.seed, deterministic=True)

    labeled_images, mask_map, test_images = discover_pairs(args.data_dir)
    if len(test_images) != 5:
        print(f"[warn] Expected 5 test images without masks; found {len(test_images)}. Proceeding anyway.")

    train_imgs, val_imgs = train_val_split(labeled_images, val_ratio=args.val_ratio, seed=args.seed)

    train_ds = RoofDataset(mode="train", image_paths=train_imgs, mask_dir_map=mask_map, image_size=args.image_size)
    val_ds   = RoofDataset(mode="val",   image_paths=val_imgs,   mask_dir_map=mask_map, image_size=args.image_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(args.encoder, pretrained=True)

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_encoder=args.lr_encoder,
        lr_decoder=args.lr_decoder,
        weight_decay=args.weight_decay,
        freeze_encoder_epochs=args.freeze_epochs,
        threshold=args.threshold,
        device=args.device,
    )

    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, cfg=cfg, ckpt_dir=args.ckpt_dir)
    best_path = trainer.fit()
    print(f"[done] Best checkpoint saved at: {best_path}")


# -------------------------
# Subcommand: predict
# -------------------------

def configure_predict_parser(subparsers) -> None:
    pp = subparsers.add_parser("predict", help="Predict masks for test images.")
    pp.add_argument("--data-dir", type=str, default="data/raw",
                    help="Directory containing images (test images have no masks).")
    pp.add_argument("--ckpt-path", type=str, default="models/checkpoints/best.pth",
                    help="Path to checkpoint to load.")
    pp.add_argument("--pred-dir", type=str, default="outputs/predictions",
                    help="Directory to save predicted masks.")
    pp.add_argument("--batch-size", type=int, default=1,
                    help="Batch size for inference (1 is fine for CPU).")

    add_shared_model_args(pp)
    pp.set_defaults(func=run_predict)


def run_predict(args: argparse.Namespace) -> None:
    set_seed(args.seed, deterministic=True)

    labeled_images, mask_map, test_images = discover_pairs(args.data_dir)
    if len(test_images) == 0:
        raise RuntimeError("No test images found (images without masks).")

    test_ds = RoofDataset(mode="test", image_paths=test_images, image_size=args.image_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Build same shape as training
    model = build_model(args.encoder, pretrained=False)
    predictor = Predictor(model=model, ckpt_path=args.ckpt_path, device=args.device, threshold=args.threshold)

    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    for imgs, meta in test_loader:
        preds = predictor.predict_batch(imgs)  # (B,1,H,W)
        B = preds.size(0)
        for i in range(B):
            filename = meta["filename"][i]
            mask = preds[i, 0]  # (H,W)
            save_mask(mask, pred_dir / filename)

    print(f"[done] Wrote predictions to: {pred_dir.resolve()}")


# -------------------------
# Top-level parser
# -------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified CLI for DIDA roof segmentation.")
    subparsers = p.add_subparsers(required=True, dest="command")
    configure_train_parser(subparsers)
    configure_predict_parser(subparsers)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
