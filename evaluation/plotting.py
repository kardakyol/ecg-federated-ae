from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from evaluation.metrics import MetricsResult

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 12, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0", "#00BCD4"]


def plot_roc(results: Dict[str, MetricsResult], title="ROC Curves",
             save_path: Optional[str] = None) -> plt.Figure:
    """Overlay ROC curves. results: {"Model Name": MetricsResult}."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, (name, m) in enumerate(results.items()):
        if m.fpr is not None:
            ax.plot(m.fpr, m.tpr, color=COLORS[i % len(COLORS)],
                    lw=1.5, label=f"{name} (AUROC={m.auroc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Random")
    ax.set(xlabel="FPR", ylabel="TPR", title=title, xlim=[0,1], ylim=[0,1.02])
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    if save_path: fig.savefig(save_path)
    return fig


def plot_pr(results: Dict[str, MetricsResult], title="PR Curves",
            save_path: Optional[str] = None) -> plt.Figure:
    """Overlay Precision-Recall curves."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, (name, m) in enumerate(results.items()):
        if m.precision_curve is not None:
            ax.plot(m.recall_curve, m.precision_curve, color=COLORS[i % len(COLORS)],
                    lw=1.5, label=f"{name} (AUPRC={m.auprc:.4f})")
    ax.set(xlabel="Recall", ylabel="Precision", title=title, xlim=[0,1], ylim=[0,1.02])
    ax.legend(loc="lower left"); ax.grid(True, alpha=0.3)
    if save_path: fig.savefig(save_path)
    return fig


def plot_bar_comparison(data: Dict[str, Dict[str, Dict[str, float]]],
                        metrics=None, title="Model Comparison",
                        save_path: Optional[str] = None) -> plt.Figure:
    if metrics is None:
        metrics = ["auroc", "auprc", "sensitivity", "specificity", "f1"]
    models = list(data.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, model in enumerate(models):
        means = [data[model][m]["mean"] for m in metrics]
        stds = [data[model][m]["std"] for m in metrics]
        ax.bar(x + i * width, means, width, yerr=stds, label=model,
               color=COLORS[i % len(COLORS)], capsize=3)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set(ylabel="Score", title=title, ylim=[0, 1.05])
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    if save_path: fig.savefig(save_path)
    return fig
