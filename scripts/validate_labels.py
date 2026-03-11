"""
LABEL MAPPING VALIDATION — PTB-XL scp_statements → binary labels
==================================================================
Checks whether Ghouse's normal/abnormal labelling is correct by
inspecting the raw PTB-XL metadata.

PTB-XL label hierarchy:
  - Each record has scp_codes: {"NORM": 100.0, "SR": 0.0, ...}
  - scp_statements.csv maps each code → diagnostic_class:
      NORM, MI, STTC, HYP, CD (5 superclasses)
  - Standard anomaly detection: NORM → label=0, everything else → label=1

Common mistakes:
  1. Using "superclass" column instead of "diagnostic_class"
  2. Including non-diagnostic codes (rhythm, form) in abnormal
  3. Dropping records with mixed labels
  4. Wrong threshold for scp_code likelihood

USAGE (Colab):
    # Point to RAW PTB-XL directory (not preprocessed):
    !python validate_labels.py --raw_dir data/ptb-xl-raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3

    # Also check Ghouse's preprocessed labels:
    !python validate_labels.py --raw_dir data/ptb-xl-raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 --check_preprocessed data/ptb-xl
"""

import argparse
import ast
import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter


def load_ptbxl_metadata(raw_dir):
    """Load ptbxl_database.csv and scp_statements.csv."""
    raw_dir = Path(raw_dir)

    # Try common locations
    db_candidates = [
        raw_dir / "ptbxl_database.csv",
        raw_dir / "ptbxl_database.csv",
    ]
    scp_candidates = [
        raw_dir / "scp_statements.csv",
    ]

    db_path = None
    for p in db_candidates:
        if p.exists():
            db_path = p
            break

    scp_path = None
    for p in scp_candidates:
        if p.exists():
            scp_path = p
            break

    if db_path is None:
        # List what's actually there
        print(f"Contents of {raw_dir}:")
        for f in sorted(raw_dir.iterdir()):
            print(f"  {f.name}")
        raise FileNotFoundError(f"ptbxl_database.csv not found in {raw_dir}")

    if scp_path is None:
        raise FileNotFoundError(f"scp_statements.csv not found in {raw_dir}")

    print(f"Loading: {db_path}")
    print(f"Loading: {scp_path}")

    db = pd.read_csv(db_path, index_col="ecg_id")
    scp = pd.read_csv(scp_path, index_col=0)

    return db, scp


def parse_scp_codes(scp_str):
    """Parse scp_codes string → dict. Handles both eval-able and literal formats."""
    try:
        return ast.literal_eval(scp_str)
    except:
        return {}


def validate_labels(raw_dir, check_preprocessed=None):
    db, scp = load_ptbxl_metadata(raw_dir)

    print(f"\n{'='*60}")
    print(f"  PTB-XL DATABASE OVERVIEW")
    print(f"{'='*60}")
    print(f"Total records: {len(db)}")
    print(f"Columns: {list(db.columns)}")

    # ── 1. Inspect scp_statements.csv ──
    print(f"\n{'='*60}")
    print(f"  SCP STATEMENTS TABLE")
    print(f"{'='*60}")
    print(f"Total SCP codes: {len(scp)}")
    print(f"Columns: {list(scp.columns)}")

    # Check which column exists for diagnostic class
    class_col = None
    for candidate in ["diagnostic_class", "superclass", "diagnostic_superclass"]:
        if candidate in scp.columns:
            class_col = candidate
            break

    if class_col is None:
        print("WARNING: No diagnostic class column found!")
        print(f"Available columns: {list(scp.columns)}")
        return

    print(f"\nUsing column: '{class_col}'")
    print(f"\nDiagnostic class distribution in scp_statements:")
    class_counts = scp[class_col].value_counts(dropna=False)
    for cls, cnt in class_counts.items():
        print(f"  {str(cls):10s}: {cnt} codes")

    # Show diagnostic codes
    print(f"\nDiagnostic codes by class:")
    diagnostic_mask = scp[class_col].notna() & (scp[class_col] != "")
    for cls in scp[class_col].dropna().unique():
        if pd.isna(cls) or cls == "":
            continue
        codes = scp[scp[class_col] == cls].index.tolist()
        print(f"  {cls}: {codes[:15]}{'...' if len(codes) > 15 else ''} ({len(codes)} total)")

    # ── 2. Map each record to diagnostic superclass ──
    print(f"\n{'='*60}")
    print(f"  RECORD-LEVEL LABEL MAPPING")
    print(f"{'='*60}")

    # Parse scp_codes for each record
    db["scp_codes_parsed"] = db["scp_codes"].apply(parse_scp_codes)

    # Build code → diagnostic_class lookup
    code_to_class = {}
    for code, row in scp.iterrows():
        if pd.notna(row.get(class_col)):
            code_to_class[code] = row[class_col]

    print(f"\nCode → class mapping examples:")
    for code, cls in list(code_to_class.items())[:20]:
        print(f"  {code:10s} → {cls}")

    # Assign diagnostic classes to each record
    record_classes = []
    norm_count = 0
    multi_class = 0
    no_diag = 0

    for ecg_id, row in db.iterrows():
        codes = row["scp_codes_parsed"]
        # Get diagnostic classes for this record (with likelihood > 0)
        diag_classes = set()
        diag_codes_found = []
        for code, likelihood in codes.items():
            if likelihood > 0 and code in code_to_class:
                diag_classes.add(code_to_class[code])
                diag_codes_found.append((code, code_to_class[code], likelihood))

        if len(diag_classes) == 0:
            no_diag += 1
            record_classes.append({"ecg_id": ecg_id, "classes": set(), "label": -1})
        elif diag_classes == {"NORM"}:
            norm_count += 1
            record_classes.append({"ecg_id": ecg_id, "classes": diag_classes, "label": 0})
        else:
            if len(diag_classes) > 1:
                multi_class += 1
            record_classes.append({"ecg_id": ecg_id, "classes": diag_classes, "label": 1})

    total = len(record_classes)
    label_0 = sum(1 for r in record_classes if r["label"] == 0)
    label_1 = sum(1 for r in record_classes if r["label"] == 1)
    label_neg = sum(1 for r in record_classes if r["label"] == -1)

    print(f"\nRecord-level statistics:")
    print(f"  Total records:        {total}")
    print(f"  Normal (NORM only):   {label_0} ({100*label_0/total:.1f}%)")
    print(f"  Abnormal (non-NORM):  {label_1} ({100*label_1/total:.1f}%)")
    print(f"  No diagnostic code:   {label_neg} ({100*label_neg/total:.1f}%)")
    print(f"  Multi-class abnormal: {multi_class}")

    # Class combination breakdown
    print(f"\nDiagnostic class combinations (top 20):")
    combo_counter = Counter()
    for r in record_classes:
        if r["classes"]:
            combo_counter[frozenset(r["classes"])] += 1
        else:
            combo_counter[frozenset(["NO_DIAG"])] += 1

    for combo, cnt in combo_counter.most_common(20):
        pct = 100 * cnt / total
        print(f"  {str(set(combo)):40s}: {cnt:5d} ({pct:.1f}%)")

    # ── 3. Expected split sizes ──
    print(f"\n{'='*60}")
    print(f"  EXPECTED SPLIT SIZES (patient-level 70/15/15)")
    print(f"{'='*60}")

    # PTB-XL has strat_fold column for predefined splits
    if "strat_fold" in db.columns:
        print(f"\nUsing strat_fold (PTB-XL predefined folds 1-10):")
        print(f"  Common split: folds 1-8 train, 9 val, 10 test")
        for fold in range(1, 11):
            n = (db["strat_fold"] == fold).sum()
            print(f"    Fold {fold:2d}: {n} records")

        # Standard split
        train_mask = db["strat_fold"].isin(range(1, 9))
        val_mask = db["strat_fold"] == 9
        test_mask = db["strat_fold"] == 10

        for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
            subset = [r for r, m in zip(record_classes, mask) if m]
            n_norm = sum(1 for r in subset if r["label"] == 0)
            n_abn = sum(1 for r in subset if r["label"] == 1)
            n_nodiag = sum(1 for r in subset if r["label"] == -1)
            total_sub = len(subset)
            print(f"\n  {name} (strat_fold based):")
            print(f"    Total: {total_sub}")
            print(f"    Normal: {n_norm} ({100*n_norm/max(total_sub,1):.1f}%)")
            print(f"    Abnormal: {n_abn} ({100*n_abn/max(total_sub,1):.1f}%)")
            print(f"    No diag: {n_nodiag}")

    # ── 4. Compare with Ghouse's preprocessed labels ──
    if check_preprocessed:
        print(f"\n{'='*60}")
        print(f"  COMPARISON WITH PREPROCESSED LABELS")
        print(f"{'='*60}")
        prep_dir = Path(check_preprocessed)

        for split_name in ["train", "val", "test"]:
            label_path = prep_dir / f"{split_name}_labels.npy"
            if label_path.exists():
                labels = np.load(label_path)
                unique, counts = np.unique(labels, return_counts=True)
                print(f"\n  {split_name}_labels.npy:")
                print(f"    Shape: {labels.shape}, dtype: {labels.dtype}")
                for u, c in zip(unique, counts):
                    print(f"    label={u}: {c} ({100*c/len(labels):.1f}%)")
            else:
                print(f"\n  {split_name}_labels.npy: NOT FOUND")

        # Check if train has ONLY normal (expected for anomaly detection)
        train_labels = np.load(prep_dir / "train_labels.npy")
        if (train_labels == 0).all():
            print(f"\n  ✓ Train set is 100% normal (correct for anomaly detection)")
            print(f"    Train size: {len(train_labels)}")

            # Compare expected normal count
            if "strat_fold" in db.columns:
                expected_normal_train = sum(
                    1 for r, m in zip(record_classes, train_mask)
                    if m and r["label"] == 0
                )
                print(f"    Expected normal in train (folds 1-8): {expected_normal_train}")
                if abs(len(train_labels) - expected_normal_train) > 100:
                    print(f"    ⚠️  MISMATCH: preprocessed has {len(train_labels)}, "
                          f"expected ~{expected_normal_train}")
                else:
                    print(f"    ✓ Count matches expected")

        # Check val/test abnormal ratio
        for split_name in ["val", "test"]:
            labels = np.load(prep_dir / f"{split_name}_labels.npy")
            abn_pct = 100 * (labels == 1).sum() / len(labels)
            print(f"\n  {split_name} abnormal ratio: {abn_pct:.1f}%")
            if abn_pct > 60:
                print(f"    ⚠️  HIGH ABNORMAL RATIO — check label mapping")
            elif abn_pct < 30:
                print(f"    ⚠️  LOW ABNORMAL RATIO — might be under-labelling")
            else:
                print(f"    ✓ Ratio looks reasonable")

    print(f"\n{'='*60}")
    print(f"  SUMMARY & RECOMMENDATIONS")
    print(f"{'='*60}")
    print(f"  1. NORM-only records: {label_0}/{total} ({100*label_0/total:.1f}%)")
    print(f"  2. If val/test abnormal > 55%, label mapping may be too aggressive")
    print(f"  3. Records with no diagnostic code ({label_neg}) should be EXCLUDED")
    print(f"  4. Recommended: use likelihood threshold >= 50 for scp_codes")
    print(f"  5. Use strat_fold for splits (no patient leakage guaranteed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Path to raw PTB-XL directory")
    parser.add_argument("--check_preprocessed", type=str, default=None,
                        help="Path to Ghouse's preprocessed data for comparison")
    args = parser.parse_args()
    validate_labels(args.raw_dir, args.check_preprocessed)
