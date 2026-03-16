"""
SHARED PLOTTING - same style, colors, DPI for all figures in the paper.
300 dpi, IEEE Access compatible. Everyone imports from here.
"""
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
    """Grouped bar chart with error bars for paper figures."""
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


def plot_perclass_bar(
    perclass_data: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    metric: str = "auroc",
    conditions: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Grouped bar chart: per-class metric for each model.

    Args:
        perclass_data: {model_name: {condition: {metric: {"mean", "std"}}}}
        metric: which metric to plot (default "auroc")
        conditions: which conditions to include (default: all except "overall")
        title: plot title
        save_path: where to save
    """
    models = list(perclass_data.keys())
    if conditions is None:
        all_conds = set()
        for model_agg in perclass_data.values():
            all_conds.update(model_agg.keys())
        all_conds.discard("overall")
        conditions = sorted(all_conds)

    if title is None:
        title = f"Per-Class {metric.upper()} by Model"

    x = np.arange(len(conditions))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(max(8, len(conditions) * 1.8), 5))

    for i, model in enumerate(models):
        means, stds = [], []
        for cond in conditions:
            entry = perclass_data.get(model, {}).get(cond, {}).get(metric, {})
            means.append(entry.get("mean", 0.0))
            stds.append(entry.get("std", 0.0))
        ax.bar(x + i * width, means, width, yerr=stds, label=model,
               color=COLORS[i % len(COLORS)], capsize=3, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(conditions)
    ax.set(ylabel=metric.upper(), title=title, ylim=[0, 1.05])
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    if save_path:
        fig.savefig(save_path)
    return fig
