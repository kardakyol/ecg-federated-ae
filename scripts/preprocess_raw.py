"""
RAW PTB-XL PREPROCESSING — wfdb → bandpass → z-score

Pipeline:
  1. Load raw signals via wfdb (500 Hz or 100 Hz)
  2. Bandpass filter: 0.05–47 Hz (Butterworth 4th order)
  3. Baseline wander removal (highpass component of bandpass)
  4. Resample to 100 Hz if needed (→ 1000 timesteps for 10s)
  5. Patient-level split using strat_fold (folds 1-8 train, 9 val, 10 test)
  6. Binary labels: NORM → 0, all other diagnostic → 1
  7. Per-lead z-score normalization (stats from train only)
  8. Train set: normal only; Val/Test: both classes
  9. Save as channels-first .npy (B, 12, 1000)

REQUIREMENTS:
    pip install wfdb scipy numpy pandas

USAGE (Colab):
    # Standard (100 Hz records):
    !python preprocess_raw.py --raw_dir data/ptb-xl-raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 --output_dir data/ptb-xl-clean --sampling_rate 100

    # High-res (500 Hz → downsample to 100):
    !python preprocess_raw.py --raw_dir data/ptb-xl-raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 --output_dir data/ptb-xl-clean --sampling_rate 500

OUTPUT: data/ptb-xl-clean/ with train/val/test signals + labels + subclass labels
"""

import argparse
import ast
import os
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, resample_poly
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════
# SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════════

def bandpass_filter(signal, lowcut=0.05, highcut=47.0, fs=500, order=4):
    """Apply Butterworth bandpass filter.

    Args:
        signal: (T, 12) raw ECG signal
        lowcut: high-pass cutoff (removes baseline wander)
        highcut: low-pass cutoff (removes high-freq noise)
        fs: sampling frequency
        order: filter order

    Returns:
        filtered signal (T, 12)
    """
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq

    # Clamp to valid range
    low = max(low, 1e-5)
    high = min(high, 0.9999)

    sos = butter(order, [low, high], btype="band", output="sos")
    filtered = np.zeros_like(signal)
    for lead in range(signal.shape[1]):
        filtered[:, lead] = sosfiltfilt(sos, signal[:, lead])
    return filtered


def load_raw_signal(raw_dir, filename, sampling_rate):
    """Load a single ECG record via wfdb.

    Args:
        raw_dir: PTB-XL root directory
        filename: relative path from metadata (e.g., 'records100/00000/00001_lr')
        sampling_rate: 100 or 500

    Returns:
        signal: (T, 12) numpy array, or None if failed
    """
    # Build full path (without extension — wfdb adds it)
    if sampling_rate == 100:
        # 100 Hz files are in filename_lr
        record_path = os.path.join(raw_dir, filename)
    else:
        # 500 Hz files: replace 'records100' with 'records500' and '_lr' with '_hr'
        record_path = os.path.join(raw_dir, filename.replace("records100", "records500").replace("_lr", "_hr"))

    try:
        record = wfdb.rdrecord(record_path)
        return record.p_signal  # (T, 12)
    except Exception as e:
        return None


# ══════════════════════════════════════════════════════════════
# LABEL PROCESSING
# ══════════════════════════════════════════════════════════════

def build_label_mapping(raw_dir, likelihood_threshold=50.0):
    """Build binary labels from PTB-XL metadata.

    Returns:
        db: DataFrame with added 'binary_label' and 'diagnostic_classes' columns
        stats: dict with label statistics
    """
    raw_dir = Path(raw_dir)
    db = pd.read_csv(raw_dir / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(raw_dir / "scp_statements.csv", index_col=0)

    # Find diagnostic class column
    class_col = None
    for candidate in ["diagnostic_class", "superclass", "diagnostic_superclass"]:
        if candidate in scp.columns:
            class_col = candidate
            break

    if class_col is None:
        raise ValueError(f"No diagnostic class column in scp_statements. Columns: {list(scp.columns)}")

    print(f"Using '{class_col}' column from scp_statements.csv")

    # Build code → diagnostic class lookup (only diagnostic codes)
    code_to_class = {}
    for code, row in scp.iterrows():
        val = row.get(class_col)
        if pd.notna(val) and val != "":
            code_to_class[code] = val

    print(f"Diagnostic codes mapped: {len(code_to_class)}")

    # Parse scp_codes and assign labels
    binary_labels = []
    diag_classes_list = []
    subclass_labels = []

    for ecg_id, row in db.iterrows():
        codes = ast.literal_eval(row["scp_codes"])

        # Filter by likelihood threshold and get diagnostic classes
        diag_classes = set()
        specific_codes = []
        for code, likelihood in codes.items():
            if likelihood >= likelihood_threshold and code in code_to_class:
                diag_classes.add(code_to_class[code])
                specific_codes.append(code)

        if len(diag_classes) == 0:
            binary_labels.append(-1)  # no diagnostic info → exclude
            diag_classes_list.append(set())
            subclass_labels.append("UNKNOWN")
        elif diag_classes == {"NORM"}:
            binary_labels.append(0)
            diag_classes_list.append(diag_classes)
            subclass_labels.append("NORM")
        else:
            binary_labels.append(1)
            diag_classes_list.append(diag_classes)
            # Subclass: pick the non-NORM class (or first if multiple)
            non_norm = diag_classes - {"NORM"}
            subclass_labels.append(sorted(non_norm)[0] if non_norm else "MIXED")

    db["binary_label"] = binary_labels
    db["diagnostic_classes"] = diag_classes_list
    db["subclass_label"] = subclass_labels

    # Stats
    total = len(db)
    n_norm = (db["binary_label"] == 0).sum()
    n_abn = (db["binary_label"] == 1).sum()
    n_excl = (db["binary_label"] == -1).sum()

    stats = {
        "total": total, "normal": n_norm, "abnormal": n_abn, "excluded": n_excl
    }

    print(f"\nLabel distribution:")
    print(f"  Normal (NORM):     {n_norm} ({100*n_norm/total:.1f}%)")
    print(f"  Abnormal:          {n_abn} ({100*n_abn/total:.1f}%)")
    print(f"  Excluded (no diag):{n_excl} ({100*n_excl/total:.1f}%)")

    print(f"\nSubclass distribution (abnormal):")
    for cls, cnt in db[db["binary_label"] == 1]["subclass_label"].value_counts().items():
        print(f"  {cls}: {cnt}")

    return db, stats


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def preprocess(args):
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Raw PTB-XL: {raw_dir}")
    print(f"Output: {output_dir}")
    print(f"Sampling rate: {args.sampling_rate} Hz")
    print(f"Target: 100 Hz, 1000 timesteps")

    # ── 1. Build labels ──
    print(f"\n{'='*60}")
    print(f"  STEP 1: LABEL MAPPING")
    print(f"{'='*60}")
    db, stats = build_label_mapping(raw_dir, args.likelihood_threshold)

    # Exclude records with no diagnostic info
    db_valid = db[db["binary_label"] >= 0].copy()
    print(f"\nUsing {len(db_valid)}/{len(db)} records (excluded {stats['excluded']} with no diagnostic)")

    # ── 2. Patient-level split using strat_fold ──
    print(f"\n{'='*60}")
    print(f"  STEP 2: PATIENT-LEVEL SPLIT (strat_fold)")
    print(f"{'='*60}")

    train_df = db_valid[db_valid["strat_fold"].isin(range(1, 9))]
    val_df = db_valid[db_valid["strat_fold"] == 9]
    test_df = db_valid[db_valid["strat_fold"] == 10]

    print(f"  Train (folds 1-8): {len(train_df)}")
    print(f"  Val   (fold 9):    {len(val_df)}")
    print(f"  Test  (fold 10):   {len(test_df)}")

    # ── 3. Load and filter signals ──
    print(f"\n{'='*60}")
    print(f"  STEP 3: LOAD + BANDPASS FILTER + RESAMPLE")
    print(f"{'='*60}")

    target_fs = 100
    target_len = 1000  # 10s × 100 Hz

    def process_split(df, split_name, normal_only=False):
        """Load, filter, and normalize signals for a split."""
        if normal_only:
            df = df[df["binary_label"] == 0]
            print(f"\n  {split_name}: loading {len(df)} NORMAL-ONLY records")
        else:
            print(f"\n  {split_name}: loading {len(df)} records "
                  f"({(df['binary_label']==0).sum()} normal, "
                  f"{(df['binary_label']==1).sum()} abnormal)")

        signals = []
        labels = []
        subclass = []
        skipped = 0

        for ecg_id, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split_name}"):
            # Determine filename
            if args.sampling_rate == 100:
                filename = row["filename_lr"]
            else:
                filename = row["filename_hr"]

            sig = load_raw_signal(raw_dir, filename, args.sampling_rate)
            if sig is None:
                skipped += 1
                continue

            # Bandpass filter at original sampling rate
            sig = bandpass_filter(sig, lowcut=0.05, highcut=47.0,
                                  fs=args.sampling_rate, order=4)

            # Resample to 100 Hz if needed
            if args.sampling_rate != target_fs:
                # resample_poly: up by target_fs, down by original_fs
                from math import gcd
                g = gcd(target_fs, args.sampling_rate)
                sig = resample_poly(sig, target_fs // g, args.sampling_rate // g, axis=0)

            # Ensure exact length
            if sig.shape[0] > target_len:
                sig = sig[:target_len, :]
            elif sig.shape[0] < target_len:
                # Pad with zeros (rare)
                pad = np.zeros((target_len - sig.shape[0], sig.shape[1]))
                sig = np.concatenate([sig, pad], axis=0)

            # Convert to channels-first: (12, 1000)
            sig = sig.T  # (12, T)

            signals.append(sig)
            labels.append(row["binary_label"])
            subclass.append(row["subclass_label"])

        if skipped > 0:
            print(f"  Skipped {skipped} records (load failed)")

        signals = np.array(signals, dtype=np.float32)  # (N, 12, 1000)
        labels = np.array(labels, dtype=np.int64)

        # Subclass encoding
        subclass_map = {"NORM": 0, "MI": 1, "STTC": 2, "HYP": 3, "CD": 4, "MIXED": 5, "UNKNOWN": 6}
        subclass_encoded = np.array([subclass_map.get(s, 6) for s in subclass], dtype=np.int64)

        print(f"  {split_name}: {signals.shape}, labels: {labels.shape}")
        return signals, labels, subclass_encoded

    # Process splits
    train_signals, train_labels, train_subclass = process_split(train_df, "train", normal_only=True)
    val_signals, val_labels, val_subclass = process_split(val_df, "val", normal_only=False)
    test_signals, test_labels, test_subclass = process_split(test_df, "test", normal_only=False)

    # ── 4. Per-lead z-score normalization ──
    print(f"\n{'='*60}")
    print(f"  STEP 4: PER-LEAD Z-SCORE NORMALIZATION")
    print(f"{'='*60}")

    # Compute stats from train set only
    lead_means = train_signals.mean(axis=(0, 2), keepdims=True)  # (1, 12, 1)
    lead_stds = train_signals.std(axis=(0, 2), keepdims=True)    # (1, 12, 1)
    lead_stds = np.where(lead_stds < 1e-8, 1.0, lead_stds)

    print(f"  Per-lead stats (train):")
    for i in range(12):
        print(f"    Lead {i:2d}: mean={lead_means[0,i,0]:+.6f}  std={lead_stds[0,i,0]:.6f}")

    # Apply to all splits
    train_signals = (train_signals - lead_means) / lead_stds
    val_signals = (val_signals - lead_means) / lead_stds
    test_signals = (test_signals - lead_means) / lead_stds

    # Verify
    print(f"\n  After normalization (train):")
    print(f"    Global mean: {train_signals.mean():.6f}")
    print(f"    Global std:  {train_signals.std():.6f}")
    print(f"    Min: {train_signals.min():.4f}, Max: {train_signals.max():.4f}")

    # ── 5. Save ──
    print(f"\n{'='*60}")
    print(f"  STEP 5: SAVE")
    print(f"{'='*60}")

    np.save(output_dir / "train_signals.npy", train_signals)
    np.save(output_dir / "train_labels.npy", train_labels)
    np.save(output_dir / "train_subclass_labels.npy", train_subclass)
    np.save(output_dir / "val_signals.npy", val_signals)
    np.save(output_dir / "val_labels.npy", val_labels)
    np.save(output_dir / "val_subclass_labels.npy", val_subclass)
    np.save(output_dir / "test_signals.npy", test_signals)
    np.save(output_dir / "test_labels.npy", test_labels)
    np.save(output_dir / "test_subclass_labels.npy", test_subclass)

    # Save normalization stats
    np.save(output_dir / "norm_means.npy", lead_means.squeeze())
    np.save(output_dir / "norm_stds.npy", lead_stds.squeeze())

    # Summary
    print(f"\n  Saved to {output_dir}:")
    print(f"    train: {train_signals.shape} ({train_labels.sum()} abnormal)")
    print(f"    val:   {val_signals.shape} ({val_labels.sum()} abnormal)")
    print(f"    test:  {test_signals.shape} ({test_labels.sum()} abnormal)")

    # Final diagnostic
    print(f"\n{'='*60}")
    print(f"  FINAL DIAGNOSTIC")
    print(f"{'='*60}")
    print(f"  Train: 100% normal = {(train_labels == 0).all()}")
    val_abn_pct = 100 * val_labels.sum() / len(val_labels)
    test_abn_pct = 100 * test_labels.sum() / len(test_labels)
    print(f"  Val abnormal:  {val_abn_pct:.1f}%")
    print(f"  Test abnormal: {test_abn_pct:.1f}%")
    print(f"  Normalization: z-score (per-lead, from train)")
    print(f"  Shape: channels-first (B, 12, 1000)")
    print(f"  dtype: float32")
    print(f"\n  ✓ Ready for model training!")
    print(f"  Run: !python convae_ablation.py --data_dir {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PTB-XL Raw Preprocessing Pipeline")
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/ptb-xl-clean")
    parser.add_argument("--sampling_rate", type=int, default=100, choices=[100, 500])
    parser.add_argument("--likelihood_threshold", type=float, default=50.0,
                        help="Minimum likelihood for scp_code to count (default: 50)")
    args = parser.parse_args()
    preprocess(args)
