"""
PTB-XL preprocessing pipeline.

Downloads (or loads) PTB-XL from PhysioNet, applies signal preprocessing,
encodes labels, performs patient-level train/val/test split, and saves
the result as .npy files consumable by utils/dataset.py.

Supports two modes:
  --sampling_rate 100  (default) — uses 100 Hz records directly, bandpass
                        0.05-45 Hz, no decimation needed. Faster download.
  --sampling_rate 500  — uses 500 Hz records, bandpass 0.05-100 Hz, then
                        decimates to 100 Hz. Scientifically stricter.

Both modes produce the same output: (N, 12, 1000) float32 at 100 Hz.

Usage:
    python scripts/preprocess_ptbxl.py --output_dir data/ptb-xl
    python scripts/preprocess_ptbxl.py --output_dir data/ptb-xl --raw_dir data/ptb-xl-raw --seed 42
    python scripts/preprocess_ptbxl.py --sampling_rate 500  # slower but stricter
"""
from __future__ import annotations

import argparse
import ast
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, decimate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PHYSIONET_DB = "ptb-xl"
PHYSIONET_VERSION = "1.0.3"
PHYSIONET_FILES_BASE = (
    "https://storage.cloud.google.com/physionet-data"
    f"/ptb-xl/1.0.3"
)
NUM_LEADS = 12
NUM_TIMESTEPS = 1000  # 10 s * 100 Hz output
FILTER_ORDER = 4


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def bandpass_coefficients(lowcut: float, highcut: float, fs: float, order: int = 4):
    """Butterworth bandpass filter coefficients."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)  # guard against Nyquist
    return butter(order, [low, high], btype="band")


def preprocess_signal(
    signal: np.ndarray, fs: int, target_fs: int = 100
) -> np.ndarray:
    """Bandpass filter (and decimate if needed) a single multi-lead signal.

    Parameters
    ----------
    signal    : (T_in, 12) raw signal at *fs* Hz
    fs        : sampling frequency of the input
    target_fs : desired output sampling rate

    Returns
    -------
    (12, T_out) float32 — channels-first at target_fs
    """
    nyq = fs * 0.5
    highcut = min(100.0, nyq - 1.0)  # 100 Hz or Nyquist-1, whichever is lower
    b, a = bandpass_coefficients(0.05, highcut, fs, FILTER_ORDER)
    filtered = filtfilt(b, a, signal, axis=0)

    if fs != target_fs:
        factor = fs // target_fs
        filtered = decimate(filtered, factor, axis=0, zero_phase=True)

    return filtered.T.astype(np.float32)  # (12, T_out)


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def load_label_mapping(raw_dir: Path) -> dict[str, str]:
    """Map SCP code -> diagnostic_class from scp_statements.csv."""
    scp_path = raw_dir / "scp_statements.csv"
    scp_df = pd.read_csv(scp_path, index_col=0)
    mapping = {}
    for code, row in scp_df.iterrows():
        dc = row.get("diagnostic_class", np.nan)
        if pd.notna(dc):
            mapping[code] = dc
    return mapping


def encode_labels(
    metadata: pd.DataFrame, scp_mapping: dict[str, str], threshold: float = 50.0
) -> np.ndarray:
    """Binary labels: 0 = NORM, 1 = abnormal.

    A record is NORM only if its *only* diagnostic superclass (with
    likelihood >= threshold) is NORM. Any other class makes it abnormal.
    Records with no diagnostic codes above threshold are excluded (label = -1).
    """
    labels = np.full(len(metadata), -1, dtype=np.int64)

    for i, scp_str in enumerate(metadata["scp_codes"]):
        scp_dict = ast.literal_eval(scp_str)
        superclasses = set()
        for code, likelihood in scp_dict.items():
            if likelihood >= threshold and code in scp_mapping:
                superclasses.add(scp_mapping[code])

        if not superclasses:
            continue

        if superclasses == {"NORM"}:
            labels[i] = 0
        else:
            labels[i] = 1

    return labels


# ---------------------------------------------------------------------------
# Patient-level splitting
# ---------------------------------------------------------------------------

def patient_split(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split record indices by patient_id so no patient spans two sets."""
    valid_mask = labels >= 0
    valid_indices = np.where(valid_mask)[0]
    patient_ids = metadata.iloc[valid_indices]["patient_id"].values

    unique_patients = np.unique(patient_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_patients)

    n = len(unique_patients)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_patients = set(unique_patients[:n_train])
    val_patients = set(unique_patients[n_train : n_train + n_val])
    # remaining patients go to test (no explicit set needed)

    train_idx, val_idx, test_idx = [], [], []
    for idx, pid in zip(valid_indices, patient_ids):
        if pid in train_patients:
            train_idx.append(idx)
        elif pid in val_patients:
            val_idx.append(idx)
        else:
            test_idx.append(idx)

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, session=None) -> bool:
    """Download a single file. Returns True on success."""
    import requests as _req

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        s = session or _req
        r = s.get(url, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception:
        return False


def download_ptbxl(raw_dir: Path, sampling_rate: int = 100) -> None:
    """Download PTB-XL from PhysioNet if not already present.

    Only downloads the metadata CSVs and the records at the chosen sampling
    rate (100 Hz ≈ 340 MB vs 500 Hz ≈ 1.4 GB), which is much faster than
    downloading the full 1.7 GB zip.
    """
    import concurrent.futures
    import requests

    marker = raw_dir / "ptbxl_database.csv"
    if marker.exists():
        log.info("PTB-XL already downloaded at %s", raw_dir)
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    base = PHYSIONET_FILES_BASE

    # 1. Download metadata CSVs
    log.info("Downloading PTB-XL metadata...")
    session = requests.Session()
    for fname in ["ptbxl_database.csv", "scp_statements.csv"]:
        url = f"{base}/{fname}"
        dest = raw_dir / fname
        if not _download_file(url, dest, session):
            raise RuntimeError(f"Failed to download {url}")
        log.info("  %s OK", fname)

    # 2. Read metadata to get record paths
    meta = pd.read_csv(raw_dir / "ptbxl_database.csv", index_col="ecg_id")
    fname_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    record_paths = meta[fname_col].values.tolist()
    log.info("Need to download %d records at %d Hz...", len(record_paths), sampling_rate)

    # 3. Build download list: .hea + .dat for each record
    jobs: list[tuple[str, Path]] = []
    for rp in record_paths:
        for ext in [".hea", ".dat"]:
            stem = rp.rsplit(".", 1)[0] if "." in rp else rp
            url = f"{base}/{stem}{ext}"
            dest = raw_dir / f"{stem}{ext}"
            if not (dest.exists() and dest.stat().st_size > 0):
                jobs.append((url, dest))

    log.info("Downloading %d files (%d already cached)...",
             len(jobs), len(record_paths) * 2 - len(jobs))

    # 4. Parallel download with progress
    done = 0
    failed = 0

    def _dl(item):
        url, dest = item
        return _download_file(url, dest, session)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_dl, j): j for j in jobs}
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            if not fut.result():
                failed += 1
                url, dest = futures[fut]
                log.warning("Failed: %s", url)
            if done % 2000 == 0:
                log.info("  downloaded %d / %d files", done, len(jobs))

    log.info("Download complete: %d OK, %d failed out of %d", done - failed, failed, len(jobs))
    if failed > len(jobs) * 0.05:
        raise RuntimeError(
            f"Too many download failures ({failed}/{len(jobs)}). "
            f"Check network connection and retry."
        )
    log.info("PTB-XL ready at %s", raw_dir)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    raw_dir: Path, output_dir: Path, seed: int = 42, sampling_rate: int = 100
) -> None:
    download_ptbxl(raw_dir, sampling_rate=sampling_rate)

    # ---- Load metadata ----
    meta_path = raw_dir / "ptbxl_database.csv"
    metadata = pd.read_csv(meta_path, index_col="ecg_id")
    log.info("Loaded metadata: %d records", len(metadata))

    scp_mapping = load_label_mapping(raw_dir)
    log.info("SCP -> diagnostic_class mapping: %d codes", len(scp_mapping))

    # ---- Encode labels ----
    labels = encode_labels(metadata, scp_mapping)
    n_valid = int((labels >= 0).sum())
    n_norm = int((labels == 0).sum())
    n_abn = int((labels == 1).sum())
    log.info(
        "Labels: %d valid (%d NORM, %d abnormal), %d excluded",
        n_valid, n_norm, n_abn, int((labels < 0).sum()),
    )

    # ---- Choose filename column based on sampling rate ----
    if sampling_rate == 500:
        fname_col = "filename_hr"
        fs = 500
        log.info("Using 500 Hz records -> bandpass 0.05-100 Hz -> decimate to 100 Hz")
    else:
        fname_col = "filename_lr"
        fs = 100
        log.info("Using 100 Hz records -> bandpass 0.05-45 Hz (no decimation)")

    # ---- Load and preprocess signals ----
    log.info("Loading and preprocessing %d records at %d Hz...", len(metadata), fs)
    all_signals = []

    for i, (ecg_id, row) in enumerate(metadata.iterrows()):
        rec_path = str(raw_dir / row[fname_col])
        try:
            signal, _ = wfdb.rdsamp(rec_path)
            processed = preprocess_signal(signal, fs, target_fs=100)

            # Ensure exactly (12, 1000)
            if processed.shape[1] > NUM_TIMESTEPS:
                processed = processed[:, :NUM_TIMESTEPS]
            elif processed.shape[1] < NUM_TIMESTEPS:
                pad_width = NUM_TIMESTEPS - processed.shape[1]
                processed = np.pad(processed, ((0, 0), (0, pad_width)))

            all_signals.append(processed)
        except Exception as e:
            log.warning("Skipping record %s (index %d): %s", ecg_id, i, e)
            all_signals.append(
                np.zeros((NUM_LEADS, NUM_TIMESTEPS), dtype=np.float32)
            )
            labels[i] = -1

        if (i + 1) % 2000 == 0:
            log.info("  processed %d / %d", i + 1, len(metadata))

    signals = np.stack(all_signals, axis=0).astype(np.float32)  # (N, 12, 1000)
    log.info("Signal array shape: %s", signals.shape)

    # ---- Patient-level split ----
    train_idx, val_idx, test_idx = patient_split(metadata, labels, seed=seed)
    log.info("Split: train=%d, val=%d, test=%d", len(train_idx), len(val_idx), len(test_idx))

    # ---- Normalize per lead (fit on train, apply to all) ----
    train_signals = signals[train_idx].copy()
    val_signals = signals[val_idx].copy()
    test_signals = signals[test_idx].copy()

    lead_mins = np.zeros(NUM_LEADS, dtype=np.float32)
    lead_maxs = np.zeros(NUM_LEADS, dtype=np.float32)
    for lead in range(NUM_LEADS):
        lead_mins[lead] = train_signals[:, lead, :].min()
        lead_maxs[lead] = train_signals[:, lead, :].max()

    for arr in [train_signals, val_signals, test_signals]:
        for lead in range(NUM_LEADS):
            lo, hi = lead_mins[lead], lead_maxs[lead]
            rng = hi - lo
            if rng > 1e-8:
                arr[:, lead, :] = (arr[:, lead, :] - lo) / rng
            else:
                arr[:, lead, :] = 0.0

    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    # ---- Save ----
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, sig, lbl in [
        ("train", train_signals, train_labels),
        ("val", val_signals, val_labels),
        ("test", test_signals, test_labels),
    ]:
        np.save(output_dir / f"{name}_signals.npy", sig.astype(np.float32))
        np.save(output_dir / f"{name}_labels.npy", lbl.astype(np.int64))
        n0 = int((lbl == 0).sum())
        n1 = int((lbl == 1).sum())
        log.info("Saved %s: %d samples (%d normal, %d abnormal)", name, len(lbl), n0, n1)

    log.info("All files saved to %s", output_dir)

    # ---- Run validation ----
    log.info("Running validation...")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from validate_data import validate

    if validate(str(output_dir)):
        log.info("Validation PASSED")
    else:
        log.error("Validation FAILED — check output above")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Preprocess PTB-XL for the project")
    parser.add_argument(
        "--raw_dir", type=str, default="data/ptb-xl-raw",
        help="Directory with raw PTB-XL download (downloaded automatically if missing)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/ptb-xl",
        help="Directory to save preprocessed .npy files",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for patient-level splitting",
    )
    parser.add_argument(
        "--sampling_rate", type=int, default=100, choices=[100, 500],
        help="Source sampling rate: 100 Hz (faster) or 500 Hz (bandpass to 100 Hz then decimate)",
    )
    args = parser.parse_args()

    run_pipeline(
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        sampling_rate=args.sampling_rate,
    )


if __name__ == "__main__":
    main()
