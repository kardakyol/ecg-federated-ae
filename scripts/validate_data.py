import argparse, sys
from pathlib import Path
import numpy as np


def validate(data_dir: str) -> bool:
    d = Path(data_dir)
    ok = True
    for split in ["train", "val", "test"]:
        sig_path = d / f"{split}_signals.npy"
        lbl_path = d / f"{split}_labels.npy"
        if not sig_path.exists() or not lbl_path.exists():
            print(f"  FAIL: {split} files not found"); ok = False; continue
        sig = np.load(sig_path); lbl = np.load(lbl_path)
        checks = [
            (sig.dtype == np.float32, f"dtype {sig.dtype} not float32"),
            (lbl.dtype == np.int64, f"labels dtype {lbl.dtype} not int64"),
            (sig.ndim == 3, f"ndim {sig.ndim} not 3"),
            (sig.shape[1] == 12, f"shape[1]={sig.shape[1]} not 12 leads"),
            (sig.shape[2] == 1000, f"shape[2]={sig.shape[2]} not 1000 timesteps"),
            (len(sig) == len(lbl), f"length mismatch {len(sig)} vs {len(lbl)}"),
            (set(np.unique(lbl)) <= {0, 1}, f"unexpected labels {np.unique(lbl)}"),
            (not np.any(np.isnan(sig)), "contains NaN"),
            (not np.any(np.isinf(sig)), "contains Inf"),
        ]
        all_pass = True
        for passed, msg in checks:
            if not passed: print(f"  FAIL {split}: {msg}"); all_pass = False; ok = False
        if all_pass:
            print(f"  OK {split}: {len(sig)} samples ({(lbl==0).sum()} normal, {(lbl==1).sum()} abnormal)")
    print("\nAll checks passed!" if ok else "\nSome checks FAILED.")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/ptb-xl")
    args = parser.parse_args()
    sys.exit(0 if validate(args.data_dir) else 1)
