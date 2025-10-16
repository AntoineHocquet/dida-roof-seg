# src/dida_roofseg/train.py

"""
Train roof segmentation
Argparse CLI with sane defaults (no config files)
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dida_roofseg.dataset import RoofDataset
from dida_roofseg.engine import Trainer, TrainConfig
from dida_roofseg.io import discover_pairs, train_val_split
from dida_roofseg.seed import set_seed
from dida_roofseg import model as model_mod  # model.py


def parse_args():
    """
    Parse command-line arguments.
    Output is argparse.Namespace with args as attributes.
    Example usage:
      args = parse_args()
      print(args.epochs, args.batch_size, args.lr_encoder)
    20 4 0.0001
    """
    p = argparse.ArgumentParser(description="Train roof segmentation")
    p.add_argument("--data-dir", type=str, default="data/raw", help="Directory with images (+ masks)")
    p.add_argument("--ckpt-dir", type=str, default="models/checkpoints")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr-encoder", type=float, default=1e-4)
    p.add_argument("--lr-decoder", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--freeze-epochs", type=int, default=3)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--image-size", type=int, default=512, help="Resize square side (None to skip)")
    p.add_argument("--encoder", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    #p.add_argument("--in-channels", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main():
    """
    Flow:
    - Set seed; discover 25 masked images and 5 test images by filename matching.
    - Split masked images into train/val by --val-ratio (e.g., 0.2).
    - Build RoofDatasets and DataLoaders.
    - Create SegmentationModel(EncoderWrapper, DecoderUNetSmall).
    - Instantiate Trainer and call fit().
    - Print final val IoU/Dice and path to best.pth.
    """
    args = parse_args()
    set_seed(args.seed, deterministic=True)

    labeled_images, mask_map, test_images = discover_pairs(args.data_dir)
    if len(test_images) != 5:
        print(f"[warn] Expected 5 test images without masks; found {len(test_images)}. Proceeding anyway.")

    train_imgs, val_imgs = train_val_split(labeled_images, val_ratio=args.val_ratio, seed=args.seed)

    train_ds = RoofDataset(mode="train", image_paths=train_imgs, mask_dir_map=mask_map, image_size=args.image_size)
    val_ds = RoofDataset(mode="val", image_paths=val_imgs, mask_dir_map=mask_map, image_size=args.image_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ----- Build model -----
    encoder = model_mod.EncoderWrapper(name=args.encoder, pretrained=True) # in_channels=args.in_channels
    decoder = model_mod.DecoderUNetSmall(encoder_channels=encoder.feature_channels)  # adapted to encoder
    model = model_mod.SegmentationModel(encoder=encoder, decoder=decoder)

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

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        ckpt_dir=args.ckpt_dir
    )
    best_path = trainer.fit()
    print(f"[done] Best checkpoint saved at: {best_path}")


if __name__ == "__main__":
    main()
