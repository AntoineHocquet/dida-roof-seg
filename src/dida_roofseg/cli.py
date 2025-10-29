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
import json

import torch
from torch.utils.data import DataLoader

from dida_roofseg.dataset import RoofDataset
from dida_roofseg.engine import Trainer, TrainConfig, Predictor
from dida_roofseg.io import discover_pairs, train_val_split, save_mask, resize_mask_to
from dida_roofseg.seed import set_seed
from dida_roofseg import model as model_mod  # model.py
from dida_roofseg.viz import plot_batch, plot_learning_curves

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


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
    p.add_argument("--plot-dir", type=str, default="outputs",
                   help="Directory to save plots.")
    p.add_argument("--pretrained", action="store_true", # `action` is a flag that takes no argument.
               help="Use ImageNet-pretrained encoder.")
    p.add_argument("--base-dec", type=int, default=64,
               help="Base width of the decoder (try 32 on CPU).")


def build_model(encoder_name: str, pretrained: bool, base_dec: int) -> model_mod.SegmentationModel:
    encoder = model_mod.EncoderWrapper(name=encoder_name, pretrained=pretrained)
    decoder = model_mod.DecoderUNetSmall(encoder_channels=encoder.feature_channels, base_dec=base_dec)
    return model_mod.SegmentationModel(encoder=encoder, decoder=decoder)

def plot_from_predictor(
        predictor: Predictor,
        save_path: str | Path | None = None
    ) -> None:
    """
    Utilises plot_batch to plot all images from a predictor.
    """
    pass # TODO

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
    tp.add_argument("--plots", action="store_true",
                help="If set, save learning curves to --plot-dir.")


    add_shared_model_args(tp)
    tp.set_defaults(func=run_train)


def run_train(args: argparse.Namespace) -> None:
    # size check-in
    if args.image_size % 32 != 0:
        raise ValueError(f"--image-size must be a multiple of 32; got {args.image_size}")

    # Set seed for reproducibility
    # (Comment out the next line if you prefer faster, slightly nondeterministic training)
    set_seed(args.seed, deterministic=True)

    labeled_images, mask_map, test_images = discover_pairs(args.data_dir)
    if len(test_images) != 5:
        print(f"[warn] Expected 5 test images without masks; found {len(test_images)}. Proceeding anyway.")

    train_imgs, val_imgs = train_val_split(labeled_images, val_ratio=args.val_ratio, seed=args.seed)

    train_ds = RoofDataset(mode="train", image_paths=train_imgs, mask_dir_map=mask_map, image_size=args.image_size)
    val_ds   = RoofDataset(mode="val",   image_paths=val_imgs,   mask_dir_map=mask_map, image_size=args.image_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(args.encoder, pretrained=True, base_dec=args.base_dec)

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
    best_path_iou, best_path_dice, last_path, history_path = trainer.fit()
    print(
        f"\n[Training done]: Best IoU checkpoint saved at: {best_path_iou.resolve()}",
        f"\n         Best Dice checkpoint saved at: {best_path_dice.resolve()}",
        f"\n         Last checkpoint saved at: {last_path.resolve()}",
        f"\n         Training history saved at: {history_path.resolve()}"
    )

    if args.plots:
        print("[Plotting] learning curves...")
        history = json.load(open(history_path, "r"))
        plot_learning_curves(history, save_path=Path(args.plot_dir) / "learning_curves.png", show=False)

# -------------------------
# Subcommand: predict
# -------------------------

def configure_predict_parser(subparsers) -> None:
    pp = subparsers.add_parser("predict", help="Predict masks for a chosen split.")
    pp.add_argument("--data-dir", type=str, default="data/raw",
                    help="Directory containing images and masks (and test images without masks).")
    pp.add_argument("--ckpt-path", type=str, default="models/checkpoints/best_iou.pth",
                    help="Path to checkpoint to load, defaults to best IoU.")
    pp.add_argument("--pred-dir", type=str, default="outputs/predictions",
                    help="Directory to save predicted masks.")
    pp.add_argument("--batch-size", type=int, default=1,
                    help="Batch size for inference (1 is fine for CPU).")
    pp.add_argument("--mode", type=str, default="test", choices=["train", "val", "test"],
                    help="Which split to run predictions for.")
    pp.add_argument("--val-ratio", type=float, default=0.2,
                    help="Validation split ratio (used to reconstruct the split at predict-time).")

    add_shared_model_args(pp)
    pp.set_defaults(func=run_predict)


def run_predict(args: argparse.Namespace) -> None:
    # shape sanity
    if args.image_size % 32 != 0:
        raise ValueError(f"--image-size must be a multiple of 32; got {args.image_size}")

    set_seed(args.seed, deterministic=True)

    # ---- Discover dataset & reconstruct splits ----
    labeled_images, mask_map, test_images = discover_pairs(args.data_dir)
    train_imgs, val_imgs = train_val_split(labeled_images, val_ratio=args.val_ratio, seed=args.seed)

    # pick the split requested
    if args.mode == "train":
        sel_paths = train_imgs
    elif args.mode == "val":
        sel_paths = val_imgs
    else:  # "test"
        sel_paths = test_images
        if len(test_images) == 0:
            raise RuntimeError("No test images found (images without masks).")

    # Build the dataset for the chosen split
    roofdataset = RoofDataset(
        mode=args.mode,
        image_paths=sel_paths,
        mask_dir_map=mask_map if args.mode in {"train", "val"} else {},
        image_size=args.image_size
    )

    # ---- Build model & predictor for the chosen split ----
    model = build_model(args.encoder, pretrained=False, base_dec=args.base_dec)
    predictor = Predictor(
        model=model,
        ckpt_path=args.ckpt_path,
        dataset=roofdataset,
        mode=args.mode,
        device=args.device,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )

    # Always also prepare a training predictor for scores (fresh, same ckpt)
    train_ds_for_scores = RoofDataset(
        mode="train",
        image_paths=train_imgs,
        mask_dir_map=mask_map,
        image_size=args.image_size
    )
    predictor_train = Predictor(
        model=build_model(args.encoder, pretrained=False, base_dec=args.base_dec),
        ckpt_path=args.ckpt_path,
        dataset=train_ds_for_scores,
        mode="train",
        device=args.device,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )

    # ---- Output dirs ----
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ---- Save predictions for the selected split ----
    print(f"[predicting] Split={args.mode} | N={len(sel_paths)} | saving to {pred_dir.resolve()}")
    # Stable, order-consistent stems (DataLoader keeps order with shuffle=False)
    ordered_stems = [p.stem for p in sel_paths]
    k = 0  # running index over dataset order

    for preds, metas in predictor:  # uses Predictor.__iter__()
        B = preds.size(0)
        for i in range(B):
            stem = ordered_stems[k + i]

            if args.mode == "test":
                # Use original sizes from meta (present only in test mode)
                h0, w0 = metas[i]["orig_size"]
                pred_resized = resize_mask_to(preds[i], (h0, w0))  # (1,h0,w0)
                out_path = pred_dir / f"{stem}.png"
                save_mask(pred_resized, out_path)
            else:
                # Train/val don't carry meta; save at current (image_size,image_size)
                out_path = pred_dir / f"{stem}.png"
                save_mask(preds[i], out_path)

        k += B

    print(f"[done] Wrote {len(sel_paths)} predictions to: {pred_dir.resolve()}")

    # ---- Quick plot grid for the chosen split (first batch) ----
    grid_path = plot_dir / f"{args.mode}_pred_grid.png"
    made_grid = False
    for batch in predictor.predict_with_inputs():
        if args.mode in {"train", "val"}:
            imgs, preds, masks = batch
            plot_batch(
                imgs, preds, masks,
                max_n=min(6, imgs.size(0)),
                title=f"{args.mode.upper()} predictions",
                save_path=grid_path,
                show=False,
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            )
        else:
            imgs, preds, metas = batch
            # no GT overlay on test; still show preds over images
            plot_batch(
                imgs, preds, None,
                max_n=min(6, imgs.size(0))),
                title=f"{args.mode.upper()} predictions",
                save_path=grid_path,
                show=False,
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            )
        made_grid = True
        break  # only first batch
    if made_grid:
        print(f"[plot] Saved grid to: {grid_path.resolve()}")

    # ---- Training scores (always) ----
    train_scores = predictor_train.yield_training_scores()
    print(
        "[Train scores] "
        f"loss={train_scores['train_loss']:.4f} | "
        f"IoU={train_scores['train_iou']:.4f} | "
        f"Dice={train_scores['train_dice']:.4f}"
    )

    # ---- Persist a predict-run config ----
    run_cfg = {
        "command": "predict",
        "mode": args.mode,
        "data_dir": str(Path(args.data_dir).resolve()),
        "val_ratio": args.val_ratio,
        "num_train": len(train_imgs),
        "num_val": len(val_imgs),
        "num_test": len(test_images),
        "num_predicted": len(sel_paths),
        "encoder": args.encoder,
        "pretrained": args.pretrained,
        "base_dec": args.base_dec,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "threshold": args.threshold,
        "device": args.device,
        "seed": args.seed,
        "ckpt_path": str(Path(args.ckpt_path).resolve()),
        "pred_dir": str(pred_dir.resolve()),
        "plot_grid": str(grid_path.resolve()) if made_grid else None,
        "train_scores": train_scores,
    }
    with open(pred_dir / "run_predict.json", "w") as f:
        json.dump(run_cfg, f, indent=2)
    print(f"[meta] Saved predict config to: {(pred_dir / 'run_predict.json').resolve()}")

    # ---- Done ----
    print("[done]")

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
