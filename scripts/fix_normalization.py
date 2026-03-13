"""
FIX PREPROCESSING: Min-Max → Per-Lead Z-Score Normalization
=============================================================
Ghouse's pipeline applied per-lead min-max scaling [0,1].
This kills anomaly detection because MSE differences become tiny.

This script:
  1. Loads existing train/val/test .npy files
  2. Computes per-lead mean & std from TRAIN set only (no leakage)
  3. Applies z-score: (x - mean) / std per lead
  4. Saves to a new directory (does NOT overwrite originals)

USAGE (Colab):
    !python fix_normalization.py --data_dir data/ptb-xl --output_dir data/ptb-xl-zscore

Then run experiments with:
    !python convae_ablation.py --data_dir data/ptb-xl-zscore
"""

import argparse
import os
import shutil
import numpy as np
from pathlib import Path


def fix_normalization(data_dir, output_dir):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {data_dir}")
    print(f"Output: {output_dir}")

    # ── Load ──
    train_X = np.load(data_dir / "train_signals.npy")
    train_y = np.load(data_dir / "train_labels.npy")
    val_X = np.load(data_dir / "val_signals.npy")
    val_y = np.load(data_dir / "val_labels.npy")
    test_X = np.load(data_dir / "test_signals.npy")
    test_y = np.load(data_dir / "test_labels.npy")

    print(f"\nLoaded shapes:")
    print(f"  train: {train_X.shape}  labels: {train_y.shape}")
    print(f"  val:   {val_X.shape}    labels: {val_y.shape}")
    print(f"  test:  {test_X.shape}   labels: {test_y.shape}")

    # ── Before stats ──
    print(f"\nBEFORE (train_X):")
    print(f"  Global: min={train_X.min():.4f} max={train_X.max():.4f} "
          f"mean={train_X.mean():.4f} std={train_X.std():.4f}")

    # ── Compute per-lead stats from TRAIN only ──
    # train_X shape: (N, 12, 1000)
    # Compute mean/std per lead across all samples and timesteps
    lead_means = train_X.mean(axis=(0, 2), keepdims=True)  # (1, 12, 1)
    lead_stds = train_X.std(axis=(0, 2), keepdims=True)    # (1, 12, 1)

    # Avoid division by zero (dead leads)
    lead_stds = np.where(lead_stds < 1e-8, 1.0, lead_stds)

    print(f"\nPer-lead normalization stats (from train):")
    for i in range(12):
        print(f"  Lead {i:2d}: mean={lead_means[0, i, 0]:+.6f}  std={lead_stds[0, i, 0]:.6f}")

    # ── Apply z-score to ALL splits using TRAIN stats ──
    train_X_z = (train_X - lead_means) / lead_stds
    val_X_z = (val_X - lead_means) / lead_stds
    test_X_z = (test_X - lead_means) / lead_stds

    # ── After stats ──
    print(f"\nAFTER (train_X_z):")
    print(f"  Global: min={train_X_z.min():.4f} max={train_X_z.max():.4f} "
          f"mean={train_X_z.mean():.4f} std={train_X_z.std():.4f}")

    print(f"\n  Per-lead verification:")
    for i in range(12):
        lead_data = train_X_z[:, i, :]
        print(f"    Lead {i:2d}: mean={lead_data.mean():+.6f}  std={lead_data.std():.6f}")

    # ── Save ──
    np.save(output_dir / "train_signals.npy", train_X_z.astype(np.float32))
    np.save(output_dir / "train_labels.npy", train_y)
    np.save(output_dir / "val_signals.npy", val_X_z.astype(np.float32))
    np.save(output_dir / "val_labels.npy", val_y)
    np.save(output_dir / "test_signals.npy", test_X_z.astype(np.float32))
    np.save(output_dir / "test_labels.npy", test_y)

    # Also save normalization stats for reproducibility / inference
    np.save(output_dir / "norm_means.npy", lead_means.squeeze())
    np.save(output_dir / "norm_stds.npy", lead_stds.squeeze())

    # Copy over subclass labels and client splits if they exist
    for fname in ["train_subclass_labels.npy", "val_subclass_labels.npy",
                   "test_subclass_labels.npy"]:
        src = data_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            print(f"  Copied {fname}")

    client_dir = data_dir / "client_splits"
    if client_dir.exists():
        out_client = output_dir / "client_splits"
        out_client.mkdir(exist_ok=True)
        for f in client_dir.glob("*.npy"):
            shutil.copy2(f, out_client / f.name)
        print(f"  Copied client_splits/")

    print(f"\n✓ Z-score normalized data saved to {output_dir}")
    print(f"\nNow run experiments with:")
    print(f"  !python convae_ablation.py --data_dir {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to Ghouse's preprocessed data (min-max scaled)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output path (default: {data_dir}-zscore)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(args.data_dir).rstrip("/") + "-zscore"

    fix_normalization(args.data_dir, args.output_dir)
