#!/usr/bin/env python3
"""
SHARED EVALUATION AUTOMATION SCRIPT
Reads experiment results CSVs and generates figures, significance tests,
and computation cost summaries.

USAGE:
    python scripts/run_evaluation.py --results outputs/fl_results.csv
    python scripts/run_evaluation.py --results outputs/fl_results.csv --run_significance
    python scripts/run_evaluation.py --results outputs/fl_results.csv --compute_costs
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import aggregate_seeds, MetricsResult
from evaluation.plotting import plot_bar_comparison, plot_roc, plot_pr, COLORS
from evaluation.compute_cost import compute_all_costs
from evaluation.statistical_tests import (
    pairwise_wilcoxon,
    save_significance_csv,
)
from utils.reproducibility import setup_logging

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "vanilla_ae": ("models.vanilla_ae", "VanillaAE"),
    "conv_ae": ("models.conv_ae", "ConvAE"),
    "vae": ("models.vae", "VAE"),
}

EVAL_METRICS = ["auroc", "auprc", "sensitivity", "specificity", "f1"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate experiment results: generate figures, run significance tests, compute costs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_evaluation.py --results outputs/fl_results.csv\n"
            "  python scripts/run_evaluation.py --results outputs/fl_results.csv --run_significance\n"
            "  python scripts/run_evaluation.py --results outputs/fl_results.csv --compute_costs\n"
        ),
    )
    p.add_argument(
        "--results", required=True, type=str,
        help="Path to CSV file with experiment results (STANDARD_COLUMNS format).",
    )
    p.add_argument(
        "--figures_dir", default="outputs/figures", type=str,
        help="Directory for generated figures (default: outputs/figures).",
    )
    p.add_argument(
        "--run_significance", action="store_true",
        help="Run pairwise Wilcoxon signed-rank tests between models.",
    )
    p.add_argument(
        "--compute_costs", action="store_true",
        help="Compute FLOPs, inference latency, and peak memory for each model.",
    )
    p.add_argument(
        "--checkpoints_dir", default="checkpoints", type=str,
        help="Directory containing model checkpoints (for --compute_costs).",
    )
    p.add_argument(
        "--device", default="cpu", type=str,
        help="Device for computation cost benchmarking (default: cpu).",
    )
    return p.parse_args()


def load_results(csv_path: str) -> pd.DataFrame:
    """Load and validate a results CSV."""
    path = Path(csv_path)
    if not path.exists():
        logger.error(f"Results file not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from {path}")

    required = {"model", "auroc"}
    missing = required - set(df.columns)
    if missing:
        logger.error(f"CSV missing required columns: {missing}")
        sys.exit(1)

    return df


def generate_bar_comparison(df: pd.DataFrame, figures_dir: Path) -> None:
    """Generate a grouped bar chart comparing models across metrics."""
    agg_data = {}
    groups = df.groupby("model") if "setting" not in df.columns else df.groupby(["model", "setting"])

    if "setting" in df.columns:
        for (model, setting), group in df.groupby(["model", "setting"]):
            label = f"{model} ({setting})"
            metrics_dict = {}
            for m in EVAL_METRICS:
                if m in group.columns:
                    vals = group[m].dropna().values
                    if len(vals) > 0:
                        metrics_dict[m] = {
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        }
            if metrics_dict:
                agg_data[label] = metrics_dict
    else:
        for model, group in df.groupby("model"):
            metrics_dict = {}
            for m in EVAL_METRICS:
                if m in group.columns:
                    vals = group[m].dropna().values
                    if len(vals) > 0:
                        metrics_dict[m] = {
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        }
            if metrics_dict:
                agg_data[str(model)] = metrics_dict

    if not agg_data:
        logger.warning("No data available for bar comparison plot.")
        return

    available_metrics = set()
    for v in agg_data.values():
        available_metrics.update(v.keys())
    metrics_to_plot = [m for m in EVAL_METRICS if m in available_metrics]

    save_path = figures_dir / "bar_comparison.pdf"
    plot_bar_comparison(agg_data, metrics=metrics_to_plot, title="Model Comparison", save_path=str(save_path))
    logger.info(f"Bar comparison plot saved to {save_path}")


def generate_summary_table(df: pd.DataFrame) -> str:
    """Print a formatted summary table of mean +/- std per model."""
    lines = []
    header = f"{'Model':<30s}"
    for m in EVAL_METRICS:
        if m in df.columns:
            header += f" {m.upper():>18s}"
    lines.append(header)
    lines.append("-" * len(header))

    groupby_col = ["model", "setting"] if "setting" in df.columns else ["model"]
    for keys, group in df.groupby(groupby_col):
        if isinstance(keys, tuple):
            label = f"{keys[0]} ({keys[1]})"
        else:
            label = str(keys)
        row = f"{label:<30s}"
        for m in EVAL_METRICS:
            if m in group.columns:
                vals = group[m].dropna().values
                if len(vals) > 1:
                    row += f" {np.mean(vals):>7.4f}+/-{np.std(vals, ddof=1):.4f}"
                elif len(vals) == 1:
                    row += f" {vals[0]:>18.4f}"
                else:
                    row += f" {'N/A':>18s}"
        lines.append(row)

    return "\n".join(lines)


def run_significance_tests(df: pd.DataFrame, figures_dir: Path) -> None:
    """Run pairwise Wilcoxon tests across models for each metric."""
    if "seed" not in df.columns:
        logger.warning("No 'seed' column found — cannot run significance tests.")
        return

    groupby_col = "model"
    if "setting" in df.columns:
        df = df.copy()
        df["_group"] = df["model"].astype(str) + "_" + df["setting"].astype(str)
        groupby_col = "_group"

    all_pairs = []
    for metric in EVAL_METRICS:
        if metric not in df.columns:
            continue

        results_dict = {}
        for name, group in df.groupby(groupby_col):
            vals = group.sort_values("seed")[metric].dropna().values.tolist()
            if len(vals) >= 2:
                results_dict[str(name)] = vals

        if len(results_dict) < 2:
            logger.info(f"Skipping Wilcoxon for {metric}: need at least 2 models with >=2 seeds each.")
            continue

        min_len = min(len(v) for v in results_dict.values())
        results_dict = {k: v[:min_len] for k, v in results_dict.items()}

        pairs, summary = pairwise_wilcoxon(results_dict, metric=metric)
        print(f"\n{summary}\n")
        all_pairs.extend(pairs)

    if all_pairs:
        save_path = figures_dir / "significance_tests.csv"
        save_significance_csv(all_pairs, save_path)
        logger.info(f"Significance results saved to {save_path}")


def _load_model(model_name: str, checkpoints_dir: Path) -> torch.nn.Module | None:
    """Attempt to load a model from checkpoint or instantiate fresh."""
    import importlib

    clean_name = model_name.strip().lower()
    if clean_name not in MODEL_REGISTRY:
        logger.warning(f"Unknown model '{model_name}', skipping cost computation.")
        return None

    module_path, class_name = MODEL_REGISTRY[clean_name]

    ckpt_patterns = [
        checkpoints_dir / f"{clean_name}.pt",
        checkpoints_dir / f"{clean_name}_best.pt",
        checkpoints_dir / f"{model_name}.pt",
    ]

    try:
        mod = importlib.import_module(module_path)
        model_class = getattr(mod, class_name)
        model = model_class()

        for ckpt_path in ckpt_patterns:
            if ckpt_path.exists():
                state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                if isinstance(state, dict) and "model_state_dict" in state:
                    model.load_state_dict(state["model_state_dict"])
                else:
                    model.load_state_dict(state)
                logger.info(f"Loaded checkpoint: {ckpt_path}")
                break
        else:
            logger.info(f"No checkpoint found for {clean_name}, using randomly initialised model for cost estimation.")

        return model
    except Exception as e:
        logger.warning(f"Failed to load model '{model_name}': {e}")
        return None


def run_compute_costs(df: pd.DataFrame, figures_dir: Path, checkpoints_dir: Path, device: str) -> None:
    """Compute and report FLOPs, latency, memory for each unique model."""
    model_names = df["model"].dropna().unique()
    rows = []

    for name in model_names:
        model = _load_model(str(name), Path(checkpoints_dir))
        if model is None:
            continue

        logger.info(f"Computing costs for {name}...")
        costs = compute_all_costs(model, device=device)
        costs["model"] = str(name)
        rows.append(costs)

        print(f"\n  {name}:")
        for k, v in costs.items():
            if k != "model":
                print(f"    {k}: {v}")

    if rows:
        cost_df = pd.DataFrame(rows)
        save_path = figures_dir / "computation_costs.csv"
        cost_df.to_csv(save_path, index=False)
        logger.info(f"Computation costs saved to {save_path}")


def main() -> None:
    args = parse_args()
    setup_logging()

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Evaluation Automation")
    logger.info("=" * 60)

    df = load_results(args.results)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(generate_summary_table(df))
    print("=" * 60 + "\n")

    generate_bar_comparison(df, figures_dir)

    if args.run_significance:
        logger.info("Running significance tests...")
        run_significance_tests(df, figures_dir)

    if args.compute_costs:
        logger.info("Computing model costs...")
        run_compute_costs(
            df, figures_dir,
            Path(args.checkpoints_dir),
            args.device,
        )

    logger.info("Evaluation complete. Figures saved to %s", figures_dir)


if __name__ == "__main__":
    main()
