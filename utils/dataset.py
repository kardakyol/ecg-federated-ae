"""
SHARED DATASET UTILITIES - everyone uses the same data loading code.

WHY THIS EXISTS:
    Ghouse (Person A) outputs preprocessed data. Everyone else loads it
    through this module. If Shardul writes his own loader and Kaan writes
    another, format mismatches cause silent bugs (e.g. channels-first
    vs channels-last).

SUPPORTED FORMATS:
    A) {split}_signals.npy + {split}_labels.npy  (preferred)
    B) {split}.pt with {"signals": ..., "labels": ...}

AUTO-FIX: handles (N, 1000, 12) -> (N, 12, 1000) transpose automatically.

WHO USES THIS:
    Everyone who touches data: Shardul, Kaan, Raheeb, Ghadah, Hilal, Ghouse.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)


class ECGDataset(Dataset):
    """12-lead ECG dataset. signals:(N,12,1000) float32, labels:(N,) int64."""

    def __init__(self, signals: np.ndarray, labels: np.ndarray) -> None:
        assert signals.ndim == 3 and signals.shape[1] == 12
        assert len(signals) == len(labels)
        self.signals = torch.from_numpy(signals).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self): return len(self.signals)
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.signals[idx], self.labels[idx]

    @property
    def n_normal(self): return int((self.labels == 0).sum())
    @property
    def n_abnormal(self): return int((self.labels == 1).sum())


def load_splits(data_dir: str | Path) -> Dict[str, ECGDataset]:
    """Load Ghouse preprocessed PTB-XL splits. Supports .npy and .pt formats."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data dir not found: {data_dir}\n"
            f"Ensure Ghouse preprocessing pipeline has run.\n"
            f"Expected: {data_dir}/train_signals.npy + train_labels.npy"
        )
    splits = {}
    for split in ["train", "val", "test"]:
        sig_npy = data_dir / f"{split}_signals.npy"
        lbl_npy = data_dir / f"{split}_labels.npy"
        pt_path = data_dir / f"{split}.pt"

        if sig_npy.exists() and lbl_npy.exists():
            signals = np.load(sig_npy)
            labels = np.load(lbl_npy)
        elif pt_path.exists():
            data = torch.load(pt_path, map_location="cpu")
            signals = data["signals"].numpy() if torch.is_tensor(data["signals"]) else data["signals"]
            labels = data["labels"].numpy() if torch.is_tensor(data["labels"]) else data["labels"]
        else:
            raise FileNotFoundError(f"Cannot find {split} data in {data_dir}")

        signals = np.asarray(signals, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)

        # Auto-fix shape issues
        if signals.ndim == 3 and signals.shape[1] != 12 and signals.shape[2] == 12:
            logger.warning(f"{split}: Transposing (N,T,C) -> (N,C,T)")
            signals = np.transpose(signals, (0, 2, 1))
        elif signals.ndim == 2:
            logger.warning(f"{split}: Reshaping flat -> (N,12,1000)")
            signals = signals.reshape(-1, 12, 1000)

        splits[split] = ECGDataset(signals, labels)
        logger.info(f"  {split}: {len(signals)} ({(labels==0).sum()} normal, {(labels==1).sum()} abnormal)")
    return splits


def create_dataloaders(splits: Dict[str, ECGDataset], batch_size=128, num_workers=0) -> Dict[str, DataLoader]:
    """Create train (normal only), val, val_normal, test loaders.

    WHY TRAIN IS NORMAL-ONLY:
        Unsupervised anomaly detection trains on normal data only.
        At test time, abnormal signals produce higher reconstruction error.
    """
    train_ds = splits["train"]
    normal_idx = (train_ds.labels == 0).nonzero(as_tuple=True)[0].tolist()
    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_normal_idx = (splits["val"].labels == 0).nonzero(as_tuple=True)[0].tolist()
    return {
        "train": DataLoader(Subset(train_ds, normal_idx), batch_size=batch_size, shuffle=True, drop_last=True, **kw),
        "val": DataLoader(splits["val"], batch_size=batch_size, shuffle=False, **kw),
        "val_normal": DataLoader(Subset(splits["val"], val_normal_idx), batch_size=batch_size, shuffle=False, **kw),
        "test": DataLoader(splits["test"], batch_size=batch_size, shuffle=False, **kw),
    }


def create_synthetic_data(n_train=2000, n_val=500, n_test=500, abnormal_ratio=0.2, seed=42) -> Dict[str, ECGDataset]:
    """Synthetic ECG-like data for pipeline testing before Ghouse data is ready."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 10, 1000)
    splits = {}
    for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        n_abn = int(n * abnormal_ratio)
        n_nor = n - n_abn
        normal = np.stack([np.stack([
            0.5 * np.sin(2*np.pi*(1+0.1*l)*t + rng.uniform(0,2*np.pi)) + 0.05*rng.randn(1000)
            for l in range(12)]) for _ in range(n_nor)]).astype(np.float32)
        abnormal = np.stack([np.stack([
            0.5 * np.sin(2*np.pi*(1+0.1*l)*t + rng.uniform(0,2*np.pi)) + 0.05*rng.randn(1000)
            + 2.0 * np.exp(-0.5*((t - rng.uniform(2,8))/0.1)**2)
            for l in range(12)]) for _ in range(n_abn)]).astype(np.float32)
        sigs = np.concatenate([normal, abnormal])
        lbls = np.concatenate([np.zeros(n_nor), np.ones(n_abn)]).astype(np.int64)
        perm = rng.permutation(n)
        splits[split] = ECGDataset(sigs[perm], lbls[perm])
    return splits
