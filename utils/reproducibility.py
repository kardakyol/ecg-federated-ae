"""
SHARED REPRODUCIBILITY - seeds, device selection, logging.
SEEDS = [42, 123, 456] is the project-wide constant. Everyone uses these.
"""
from __future__ import annotations
import logging, os, random, sys
from pathlib import Path
import numpy as np
import torch

SEEDS = [42, 123, 456]


def set_seed(seed: int) -> None:
    """Fix ALL random sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif preferred == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(log_dir=None, level=logging.INFO, name="ecg") -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); logger.addHandler(ch)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
        fh.setFormatter(fmt); logger.addHandler(fh)
    return logger
