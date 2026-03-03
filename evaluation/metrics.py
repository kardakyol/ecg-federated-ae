"""
SHARED EVALUATION METRICS - SINGLE SOURCE OF TRUTH
DO NOT MODIFY without team consensus.

WHY THIS EXISTS:
    If everyone writes their own AUROC function, we get inconsistent numbers
    in the paper. This module is the ONE place metrics are computed.

USAGE:
    from evaluation.metrics import compute_metrics, aggregate_seeds
    result = compute_metrics(y_true, scores, threshold)
    print(result.auroc, result.f1)

WHO USES THIS:
    Shardul  - centralised AE baselines
    Kaan     - centralised VAE baselines
    Raheeb   - federated evaluation per round
    Ghadah   - quantised model evaluation
    Hilal    - DP model evaluation
    Ghouse   - ablation study evaluation
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve,
)


@dataclass
class MetricsResult:
    """Container for all evaluation metrics from one experiment run.
    to_dict() gives flat dict for CSV logging. Curves are for plotting only."""
    auroc: float = 0.0
    auprc: float = 0.0
    sensitivity: float = 0.0  # TPR / recall
    specificity: float = 0.0  # TNR
    precision: float = 0.0
    f1: float = 0.0
    threshold: float = 0.0
    n_normal: int = 0
    n_abnormal: int = 0
    # Curves for plotting (not in CSV)
    fpr: Optional[np.ndarray] = field(default=None, repr=False)
    tpr: Optional[np.ndarray] = field(default=None, repr=False)
    precision_curve: Optional[np.ndarray] = field(default=None, repr=False)
    recall_curve: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, float]:
        return {
            "auroc": self.auroc, "auprc": self.auprc,
            "sensitivity": self.sensitivity, "specificity": self.specificity,
            "precision": self.precision, "f1": self.f1,
            "threshold": self.threshold,
            "n_normal": self.n_normal, "n_abnormal": self.n_abnormal,
        }

    def __str__(self) -> str:
        return (f"AUROC={self.auroc:.4f} AUPRC={self.auprc:.4f} "
                f"Sens={self.sensitivity:.4f} Spec={self.specificity:.4f} "
                f"F1={self.f1:.4f}")


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> MetricsResult:
    """Compute ALL metrics. y_true: 0=normal 1=abnormal. scores: higher=anomalous."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    auroc = float(roc_auc_score(y_true, scores))
    auprc = float(average_precision_score(y_true, scores))
    fpr, tpr, _ = roc_curve(y_true, scores)
    prec_c, rec_c, _ = precision_recall_curve(y_true, scores)
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    return MetricsResult(
        auroc=auroc, auprc=auprc, sensitivity=float(sens),
        specificity=float(spec), precision=float(prec), f1=float(f1),
        threshold=float(threshold),
        n_normal=int((y_true == 0).sum()), n_abnormal=int((y_true == 1).sum()),
        fpr=fpr, tpr=tpr, precision_curve=prec_c, recall_curve=rec_c,
    )


def aggregate_seeds(results: List[MetricsResult]) -> Dict[str, Dict[str, float]]:
    """Aggregate over seeds -> mean +/- std for paper tables."""
    names = ["auroc", "auprc", "sensitivity", "specificity", "precision", "f1"]
    return {
        n: {"mean": float(np.mean([getattr(r, n) for r in results])),
            "std": float(np.std([getattr(r, n) for r in results], ddof=1))
                   if len(results) > 1 else 0.0}
        for n in names
    }


def format_aggregated(agg: Dict[str, Dict[str, float]]) -> str:
    lines = []
    for n, s in agg.items():
        lines.append(f"  {n.upper():12s}: {s['mean']:.4f} +/- {s['std']:.4f}")
    return chr(10).join(lines)
