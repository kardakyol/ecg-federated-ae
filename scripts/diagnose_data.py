"""
DATA DIAGNOSTIC — Run this FIRST before any model experiments.
================================================================
Checks Ghouse's preprocessed .npy files for common issues that
cause all models to hover around ~0.60 AUROC.

USAGE (Colab):
    # After mounting drive / downloading data:
    !python diagnose_data.py --data_dir data/ptb-xl

    # If using raw PTB-XL path on Colab:
    !python diagnose_data.py --data_dir data/ptb-xl-raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3

OUTPUT: Prints a full diagnostic report to console + saves diagnose_report.txt
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path


def load_npy_safe(path):
    """Load .npy with error reporting."""
    p = Path(path)
    if not p.exists():
        return None, f"NOT FOUND: {p}"
    arr = np.load(p, allow_pickle=True)
    return arr, None


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def diagnose(data_dir):
    report = []
    def log(msg):
        print(msg)
        report.append(msg)

    data_dir = Path(data_dir)
    log(f"Data directory: {data_dir}")
    log(f"Exists: {data_dir.exists()}")

    if not data_dir.exists():
        log("ERROR: data_dir does not exist!")
        return report

    # ── 1. List all files ──
    print_section("1. DIRECTORY CONTENTS")
    all_files = sorted(data_dir.rglob("*"))
    npy_files = [f for f in all_files if f.suffix == ".npy"]
    log(f"Total files: {len(all_files)}")
    log(f".npy files: {len(npy_files)}")
    for f in npy_files:
        size_mb = f.stat().st_size / (1024**2)
        log(f"  {f.relative_to(data_dir)}  ({size_mb:.1f} MB)")

    # ── 2. Try standard split names ──
    print_section("2. LOADING SPLITS")

    # Common naming patterns
    split_patterns = [
        # Pattern A: {split}_signals.npy, {split}_labels.npy
        {"train_X": "train_signals.npy", "train_y": "train_labels.npy",
         "val_X": "val_signals.npy", "val_y": "val_labels.npy",
         "test_X": "test_signals.npy", "test_y": "test_labels.npy"},
        # Pattern B: X_{split}.npy, y_{split}.npy
        {"train_X": "X_train.npy", "train_y": "y_train.npy",
         "val_X": "X_val.npy", "val_y": "y_val.npy",
         "test_X": "X_test.npy", "test_y": "y_test.npy"},
        # Pattern C: signals_{split}.npy
        {"train_X": "signals_train.npy", "train_y": "labels_train.npy",
         "val_X": "signals_val.npy", "val_y": "labels_val.npy",
         "test_X": "signals_test.npy", "test_y": "labels_test.npy"},
    ]

    splits = {}
    found_pattern = None
    for i, pattern in enumerate(split_patterns):
        all_exist = all((data_dir / v).exists() for v in pattern.values())
        if all_exist:
            found_pattern = i
            for key, fname in pattern.items():
                splits[key] = np.load(data_dir / fname)
            log(f"Found pattern {i}: {list(pattern.values())}")
            break

    if not splits:
        # Try to find ANY .npy and report
        log("WARNING: No standard split pattern found!")
        log("Attempting to load all .npy files and guess structure...")
        for f in npy_files:
            arr = np.load(f, allow_pickle=True)
            log(f"  {f.name}: shape={arr.shape}, dtype={arr.dtype}, "
                f"min={arr.min():.4f}, max={arr.max():.4f}")
        log("\nCANNOT PROCEED — fix file naming or update load_splits()")
        return report

    # ── 3. Shape & dtype check ──
    print_section("3. SHAPE & DTYPE")
    for name, arr in splits.items():
        log(f"  {name:10s}: shape={str(arr.shape):20s}  dtype={arr.dtype}")

    # Check expected shapes
    train_X = splits["train_X"]
    expected_channels_first = train_X.ndim == 3 and train_X.shape[1] == 12
    expected_channels_last = train_X.ndim == 3 and train_X.shape[2] == 12
    log(f"\n  Channels-first (B, 12, T): {expected_channels_first}")
    log(f"  Channels-last  (B, T, 12): {expected_channels_last}")

    if expected_channels_last and not expected_channels_first:
        log("  ⚠️  DATA IS CHANNELS-LAST — models expect channels-first!")
        log("     Fix: signals = signals.transpose(0, 2, 1)  in preprocessing")

    if train_X.ndim != 3:
        log(f"  ⚠️  UNEXPECTED NDIM={train_X.ndim} — expected 3D (B, C, T)")

    # ── 4. Signal statistics ──
    print_section("4. SIGNAL STATISTICS (train_X)")
    log(f"  Global min:  {train_X.min():.6f}")
    log(f"  Global max:  {train_X.max():.6f}")
    log(f"  Global mean: {train_X.mean():.6f}")
    log(f"  Global std:  {train_X.std():.6f}")

    # Per-lead stats (assuming channels-first)
    if expected_channels_first:
        log(f"\n  Per-lead statistics:")
        for lead in range(min(train_X.shape[1], 12)):
            lead_data = train_X[:, lead, :]
            log(f"    Lead {lead:2d}: mean={lead_data.mean():+.4f}  "
                f"std={lead_data.std():.4f}  "
                f"min={lead_data.min():.4f}  max={lead_data.max():.4f}")

    # Check for common issues
    log(f"\n  ISSUE CHECKS:")

    # Issue: not normalized
    if abs(train_X.mean()) > 0.1 or abs(train_X.std() - 1.0) > 0.5:
        log(f"  ⚠️  DATA MAY NOT BE Z-SCORE NORMALIZED")
        log(f"     Expected: mean≈0, std≈1")
        log(f"     Got: mean={train_X.mean():.4f}, std={train_X.std():.4f}")
    else:
        log(f"  ✓ Z-score normalization looks OK")

    # Issue: clipped to [0,1]
    if train_X.min() >= 0.0 and train_X.max() <= 1.0:
        log(f"  ⚠️  VALUES IN [0,1] — possibly min-max scaled or clipped")
        log(f"     This kills anomaly detection — MSE differences become tiny")

    # Issue: wrong dtype
    if train_X.dtype not in [np.float32, np.float64]:
        log(f"  ⚠️  DTYPE IS {train_X.dtype} — expected float32")
        log(f"     Integer data loses precision, hurts reconstruction")
    else:
        log(f"  ✓ dtype is {train_X.dtype}")

    # Issue: NaN / Inf
    nan_count = np.isnan(train_X).sum()
    inf_count = np.isinf(train_X).sum()
    if nan_count > 0 or inf_count > 0:
        log(f"  ⚠️  FOUND NaN={nan_count}, Inf={inf_count}")
    else:
        log(f"  ✓ No NaN/Inf values")

    # Issue: constant signals (dead leads)
    if expected_channels_first:
        per_sample_std = train_X.std(axis=2)  # (B, 12)
        dead_leads = (per_sample_std < 1e-6).sum(axis=0)  # per lead
        for lead in range(min(train_X.shape[1], 12)):
            if dead_leads[lead] > 0:
                pct = 100 * dead_leads[lead] / train_X.shape[0]
                log(f"  ⚠️  Lead {lead}: {dead_leads[lead]} samples "
                    f"({pct:.1f}%) have near-zero variance (dead lead)")

    # ── 5. Label distribution ──
    print_section("5. LABEL DISTRIBUTION")
    for split_name, key in [("train", "train_y"), ("val", "val_y"), ("test", "test_y")]:
        labels = splits[key]
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        log(f"  {split_name:5s}: total={total}")
        for u, c in zip(unique, counts):
            log(f"    label={u}: {c} ({100*c/total:.1f}%)")

    # Check class balance
    train_y = splits["train_y"]
    if len(np.unique(train_y)) == 2:
        normal_pct = 100 * (train_y == 0).sum() / len(train_y)
        log(f"\n  Normal ratio in train: {normal_pct:.1f}%")
        if normal_pct > 95:
            log(f"  ⚠️  VERY IMBALANCED — >95% normal")
            log(f"     This is expected for anomaly detection (train on normal only)")
            log(f"     But check: is model actually trained on normal-only subset?")
        elif normal_pct < 60:
            log(f"  ⚠️  LOW NORMAL RATIO — anomaly detection needs mostly normal training data")
    else:
        log(f"  ⚠️  Labels have {len(np.unique(train_y))} unique values — expected binary (0/1)")
        log(f"     Unique values: {np.unique(train_y)}")

    # ── 6. Train/test signal overlap check ──
    print_section("6. TRAIN-TEST SIGNAL OVERLAP (SANITY)")
    test_X = splits["test_X"]
    # Check if any test sample is identical to a train sample (data leakage)
    n_check = min(100, len(test_X))
    overlap_count = 0
    for i in range(n_check):
        diffs = np.abs(train_X - test_X[i:i+1]).sum(axis=(1, 2))
        if (diffs < 1e-6).any():
            overlap_count += 1
    if overlap_count > 0:
        log(f"  ⚠️  {overlap_count}/{n_check} test samples found in train set — DATA LEAKAGE!")
    else:
        log(f"  ✓ No overlap detected (checked {n_check} test samples)")

    # ── 7. Reconstruction difficulty estimate ──
    print_section("7. RECONSTRUCTION DIFFICULTY ESTIMATE")
    if expected_channels_first:
        normal_mask = splits["train_y"] == 0
        abnormal_mask = splits["train_y"] == 1

        if normal_mask.sum() > 0 and abnormal_mask.sum() > 0:
            normal_signals = train_X[normal_mask]
            abnormal_signals = train_X[abnormal_mask]

            # Mean signal per class
            normal_mean = normal_signals.mean(axis=0)
            abnormal_mean = abnormal_signals.mean(axis=0)

            # MSE between class means
            class_mse = ((normal_mean - abnormal_mean) ** 2).mean()
            log(f"  MSE between normal/abnormal class means: {class_mse:.6f}")

            # Within-class variance
            normal_var = normal_signals.var(axis=0).mean()
            abnormal_var = abnormal_signals.var(axis=0).mean()
            log(f"  Within-class variance (normal):   {normal_var:.6f}")
            log(f"  Within-class variance (abnormal): {abnormal_var:.6f}")

            # Signal-to-noise ratio for anomaly detection
            if normal_var > 0:
                snr = class_mse / normal_var
                log(f"  Anomaly SNR (class_mse / normal_var): {snr:.6f}")
                if snr < 0.01:
                    log(f"  ⚠️  VERY LOW SNR — normal and abnormal signals are nearly identical")
                    log(f"     This means the raw reconstruction approach may not work")
                    log(f"     Consider: per-lead scoring, frequency-domain features, or deeper models")
                elif snr < 0.1:
                    log(f"  ⚠️  LOW SNR — anomaly detection will be challenging")
                else:
                    log(f"  ✓ SNR looks reasonable for anomaly detection")
        else:
            log(f"  Cannot compute — need both normal and abnormal samples in train")

    # ── Summary ──
    print_section("DIAGNOSTIC SUMMARY")
    log("Review all ⚠️ warnings above. Common fixes:")
    log("  1. If not z-score normalized → fix preprocessing")
    log("  2. If channels-last → transpose to channels-first")
    log("  3. If dtype is int → cast to float32")
    log("  4. If very low SNR → bottleneck must be tight (8-16)")
    log("  5. If label distribution is wrong → check label extraction")

    # Save report
    report_path = "diagnose_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    log(f"\nReport saved to {report_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Diagnostic for ECG Pipeline")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to preprocessed PTB-XL .npy files")
    args = parser.parse_args()
    diagnose(args.data_dir)
