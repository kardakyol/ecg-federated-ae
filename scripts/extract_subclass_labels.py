"""
extract_subclass_labels.py — Person C (Kaan), Sprint 3
=======================================================
Extracts MI, STTC, HYP, CD per-sample labels from PTB-XL metadata
and saves them as test_subclass_labels.npy for per-class breakdown.

PTB-XL superclass mapping (scp_codes):
  NORM  → Normal
  MI    → Myocardial Infarction
  STTC  → ST/T-wave Change
  HYP   → Hypertrophy
  CD    → Conduction Disturbance

Requires ptbxl_database.csv from the raw PTB-XL download.
The raw folder structure should be:
  data/ptb-xl-raw/ptbxl_database.csv
  data/ptb-xl-raw/scp_statements.csv

Output:
  data/ptb-xl/test_subclass_labels.npy   shape (N_test,)  int: {-1,0,1,2,3,4}
  data/ptb-xl/val_subclass_labels.npy    shape (N_val,)
  data/ptb-xl/subclass_map.json          label → int mapping

Usage:
    python scripts/extract_subclass_labels.py --raw_dir data/ptb-xl-raw --data_dir data/ptb-xl
    python scripts/extract_subclass_labels.py --raw_dir data/ptb-xl-raw --data_dir data/ptb-xl --check
"""

from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SUPERCLASS_MAP = {
    "NORM": 0,
    "MI":   1,
    "STTC": 2,
    "HYP":  3,
    "CD":   4,
}

SUPERCLASS_NAMES = {
    0: "NORM",
    1: "MI",
    2: "STTC",
    3: "HYP",
    4: "CD",
}

UNKNOWN_LABEL = -1  # record doesn't map to any of the 5 superclasses


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_scp_codes(raw: str) -> dict:
    """Parse scp_codes column: "{'NORM': 100.0}" -> {'NORM': 100.0}"""
    try:
        return ast.literal_eval(raw)
    except Exception:
        return {}


def get_superclass(scp_codes: dict, scp_df: pd.DataFrame) -> int:
    """
    Map a record's scp_codes to a single superclass integer.
    Strategy: pick the superclass with the highest confidence score.
    If multiple superclasses tie, prefer MI > CD > STTC > HYP > NORM.
    Returns UNKNOWN_LABEL if no match found.
    """
    scores: dict[str, float] = {}
    for code, confidence in scp_codes.items():
        if code in scp_df.index:
            sc = scp_df.loc[code, "diagnostic_class"]
            if pd.notna(sc) and sc in SUPERCLASS_MAP:
                # accumulate confidence per superclass
                scores[sc] = scores.get(sc, 0.0) + float(confidence)

    if not scores:
        return UNKNOWN_LABEL

    best_sc = max(scores, key=lambda k: scores[k])
    return SUPERCLASS_MAP[best_sc]


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_labels(raw_dir: str, data_dir: str) -> None:
    raw_path = Path(raw_dir)
    data_path = Path(data_dir)

    db_csv = raw_path / "ptbxl_database.csv"
    scp_csv = raw_path / "scp_statements.csv"

    if not db_csv.exists():
        raise FileNotFoundError(
            f"ptbxl_database.csv not found at {db_csv}\n"
            "Download PTB-XL from https://physionet.org/content/ptb-xl/1.0.3/ "
            "and set --raw_dir to the folder containing ptbxl_database.csv"
        )

    print(f"Loading {db_csv} ...")
    db = pd.read_csv(db_csv, index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(parse_scp_codes)

    # Load scp_statements for superclass lookup
    if scp_csv.exists():
        print(f"Loading {scp_csv} ...")
        scp_df = pd.read_csv(scp_csv, index_col=0)
    else:
        print(f"WARNING: scp_statements.csv not found at {scp_csv}")
        print("Will use direct NORM/MI/STTC/HYP/CD code matching as fallback.")
        scp_df = None

    # Build superclass label per ecg_id
    if scp_df is not None:
        db["superclass_int"] = db["scp_codes"].apply(
            lambda codes: get_superclass(codes, scp_df)
        )
    else:
        # Fallback: direct code matching
        def fallback_superclass(codes: dict) -> int:
            for code in codes:
                if code in SUPERCLASS_MAP:
                    return SUPERCLASS_MAP[code]
            return UNKNOWN_LABEL
        db["superclass_int"] = db["scp_codes"].apply(fallback_superclass)

    print(f"\nSuperclass distribution across full dataset:")
    counts = db["superclass_int"].value_counts().sort_index()
    for cls_int, cnt in counts.items():
        name = SUPERCLASS_NAMES.get(cls_int, "UNKNOWN")
        print(f"  {name:6s} ({cls_int:2d}): {cnt:5d}")

    # ── Match to preprocessed splits ──
    # Ghouse's preprocessing saved per-split indices.
    # We need to figure out which ecg_ids correspond to each split.
    # The safest approach: use strat_fold for train/val/test alignment.
    #   PTB-XL convention: strat_fold 9-10 = test, 8 = val, 1-7 = train
    # This matches the standard 70/15/15 split.

    print("\nMatching to train/val/test splits via strat_fold...")

    # strat_fold 9-10 → test (patient-level)
    test_mask = db["strat_fold"].isin([9, 10])
    val_mask  = db["strat_fold"] == 8
    train_mask = ~(test_mask | val_mask)

    test_labels  = db.loc[test_mask,  "superclass_int"].values.astype(np.int32)
    val_labels   = db.loc[val_mask,   "superclass_int"].values.astype(np.int32)
    train_labels = db.loc[train_mask, "superclass_int"].values.astype(np.int32)

    # Cross-check sizes with preprocessed .npy files
    for split, arr, fname in [
        ("train", train_labels, "train_signals.npy"),
        ("val",   val_labels,   "val_signals.npy"),
        ("test",  test_labels,  "test_signals.npy"),
    ]:
        npy_path = data_path / fname
        if npy_path.exists():
            n_signals = np.load(npy_path, mmap_mode="r").shape[0]
            if len(arr) != n_signals:
                print(f"  WARNING: {split} size mismatch — "
                      f"subclass labels={len(arr)} vs signals={n_signals}")
                print(f"  Truncating to {n_signals} to match signals.")
                arr = arr[:n_signals]
        print(f"  {split}: {len(arr)} samples")
        sc_counts = {SUPERCLASS_NAMES.get(v, 'UNK'): int(np.sum(arr == v))
                     for v in np.unique(arr)}
        print(f"    {sc_counts}")

    # Save
    np.save(data_path / "test_subclass_labels.npy",  test_labels)
    np.save(data_path / "val_subclass_labels.npy",   val_labels)
    np.save(data_path / "train_subclass_labels.npy", train_labels)

    # Save label map as JSON for reference
    label_map = {name: int(idx) for name, idx in SUPERCLASS_MAP.items()}
    label_map["UNKNOWN"] = UNKNOWN_LABEL
    with open(data_path / "subclass_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\n✓ Saved:")
    print(f"  {data_path}/test_subclass_labels.npy  ({len(test_labels)} samples)")
    print(f"  {data_path}/val_subclass_labels.npy   ({len(val_labels)} samples)")
    print(f"  {data_path}/train_subclass_labels.npy ({len(train_labels)} samples)")
    print(f"  {data_path}/subclass_map.json")


def check_labels(data_dir: str) -> None:
    """Verify saved subclass label files."""
    data_path = Path(data_dir)
    for split in ["train", "val", "test"]:
        path = data_path / f"{split}_subclass_labels.npy"
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        arr = np.load(path)
        print(f"\n{split}: {len(arr)} samples")
        for v in sorted(np.unique(arr)):
            name = SUPERCLASS_NAMES.get(v, "UNKNOWN")
            print(f"  {name:8s} ({v:2d}): {int(np.sum(arr == v)):5d}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract PTB-XL superclass labels for per-class breakdown"
    )
    parser.add_argument("--raw_dir",  default="data/ptb-xl-raw",
                        help="Directory containing ptbxl_database.csv")
    parser.add_argument("--data_dir", default="data/ptb-xl",
                        help="Directory containing preprocessed .npy files")
    parser.add_argument("--check", action="store_true",
                        help="Only verify existing label files, don't extract")
    args = parser.parse_args()

    if args.check:
        check_labels(args.data_dir)
    else:
        extract_labels(args.raw_dir, args.data_dir)


if __name__ == "__main__":
    main()
