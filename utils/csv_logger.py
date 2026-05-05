import csv
from pathlib import Path
from typing import Any

STANDARD_COLUMNS = [
    "model", "setting", "beta", "epsilon", "precision_type", "seed",
    "auroc", "auprc", "sensitivity", "specificity", "precision_score", "f1",
    "model_size_mb", "flops_m", "inference_latency_ms", "peak_memory_mb",
    "training_time_s",
]


class ResultLogger:
    """Append-mode CSV logger with standard columns."""

    def __init__(self, path: str | Path, extra_columns: list = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.columns = STANDARD_COLUMNS + (extra_columns or [])
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.columns).writeheader()

    def log(self, **kwargs: Any) -> None:
        row = {col: kwargs.get(col, "") for col in self.columns}
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.columns).writerow(row)
