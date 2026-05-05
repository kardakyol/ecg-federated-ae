from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path so imports work when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from configs.vae_config import VAEConfig, VAEArchitectureConfig, VAETrainingConfig
from models.vae import VAE
from training.train_vae import VAETrainer
from evaluation.anomaly_scorer import compute_anomaly_scores, calibrate_threshold
from evaluation.metrics import compute_metrics, aggregate_seeds, format_aggregated
from evaluation.plotting import plot_roc, plot_pr, COLORS
from utils.reproducibility import SEEDS, set_seed, get_device, setup_logging
from utils.csv_logger import ResultLogger
from utils.dataset import load_splits, create_dataloaders, create_synthetic_data

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_training_curves(history: dict, beta: float, save_path: str) -> None:
    """Plot training and validation loss curves for one beta configuration."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    epochs = range(1, len(history["train_total_loss"]) + 1)

    # Left panel: total loss + KL weight
    ax1 = axes[0]
    ax1.plot(epochs, history["train_total_loss"], 'b-', lw=1.5, label='Train')
    ax1.plot(epochs, history["val_total_loss"], 'r-', lw=1.5, label='Val')
    ax1.set(xlabel='Epoch', ylabel='Total Loss',
            title=f'Training Curves (beta={beta})')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # KL weight overlay on secondary axis
    ax1_kl = ax1.twinx()
    ax1_kl.plot(epochs, history["kl_weight"], 'g--', lw=1, alpha=0.6,
                label='KL weight')
    ax1_kl.set_ylabel('KL Annealing Weight', color='green')
    ax1_kl.set_ylim(-0.05, 1.15)

    # Right panel: MSE and KL separately
    ax2 = axes[1]
    ax2.plot(epochs, history["train_mse_loss"], 'b-', lw=1.5, label='Train MSE')
    ax2.plot(epochs, history["val_mse_loss"], 'r-', lw=1.5, label='Val MSE')
    ax2.plot(epochs, history["train_kl_loss"], 'b--', lw=1, alpha=0.7,
             label='Train KL')
    ax2.plot(epochs, history["val_kl_loss"], 'r--', lw=1, alpha=0.7,
             label='Val KL')
    ax2.set(xlabel='Epoch', ylabel='Loss Component',
            title=f'Loss Components (beta={beta})')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Training curves saved: {save_path}")


def run_single_experiment(
    config: VAEConfig,
    loaders: dict,
    beta: float,
    seed: int,
    device: torch.device,
    checkpoint_dir: str,
    fig_dir: str,
) -> dict:
    """Run one (beta, seed) experiment. Returns metrics dict.

    EXPERIMENT PIPELINE:
        1. Set seed for reproducibility
        2. Create fresh VAE model (random init depends on seed)
        3. Train with KL annealing and early stopping
        4. Calibrate threshold on normal validation data
        5. Evaluate on test set with MC anomaly scoring
        6. Return metrics for CSV logging
    """
    set_seed(seed)
    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment: beta={beta}, seed={seed}")
    logger.info(f"{'='*60}")

    # Fresh model for each seed (different random initialisation)
    model = VAE(config.architecture)
    trainer = VAETrainer(model, config.training, device, checkpoint_dir)

    # Train
    t0 = time.time()
    history = trainer.train(loaders["train"], loaders["val"], beta=beta)
    training_time = time.time() - t0

    # Save training curves (only for first seed to avoid clutter)
    if seed == SEEDS[0]:
        plot_training_curves(
            history, beta,
            save_path=f"{fig_dir}/vae_training_curves_beta{beta}.pdf"
        )

    # Calibrate threshold on normal validation data
    threshold = calibrate_threshold(
        model, loaders["val_normal"],
        percentile=config.scoring.threshold_percentile,
        alpha=config.scoring.alpha,
        n_mc_samples=config.scoring.n_mc_samples,
        device=device,
    )

    # Evaluate on test set
    scores, labels = compute_anomaly_scores(
        model, loaders["test"],
        alpha=config.scoring.alpha,
        n_mc_samples=config.scoring.n_mc_samples,
        device=device,
    )
    result = compute_metrics(labels, scores, threshold)
    logger.info(f"Test results: {result}")

    return {
        "model": "vae",
        "setting": "centralised",
        "beta": beta,
        "epsilon": "",
        "precision_type": "fp32",
        "seed": seed,
        "auroc": result.auroc,
        "auprc": result.auprc,
        "sensitivity": result.sensitivity,
        "specificity": result.specificity,
        "precision_score": result.precision,
        "f1": result.f1,
        "model_size_mb": model.model_size_mb(),
        "training_time_s": training_time,
        "metrics_result": result,  # kept for aggregation, not logged to CSV
    }


def main():
    parser = argparse.ArgumentParser(description="VAE Centralised Baselines")
    parser.add_argument("--data_dir", type=str, default="data/ptb-xl",
                        help="Path to preprocessed PTB-XL data")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (before data ready)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1 beta, 1 seed, 5 epochs")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda, mps, or cpu")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory for CSV results")
    parser.add_argument("--fig_dir", type=str, default="outputs/figures",
                        help="Directory for figures")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory for model checkpoints")
    args = parser.parse_args()

    # Setup
    setup_logging(log_dir="logs", name="vae_baseline")
    device = get_device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {device}")

    # Load data
    if args.synthetic:
        logger.info("Using SYNTHETIC data")
        splits = create_synthetic_data(n_train=2000, n_val=500, n_test=500)
    else:
        logger.info(f"Loading PTB-XL from {args.data_dir}")
        splits = load_splits(args.data_dir)

    config = VAEConfig()

    if args.quick:
        config.training.epochs = 5
        config.training.patience = 3
        config.training.kl_annealing_epochs = 2
        config.scoring.n_mc_samples = 3
        betas = [0.5]
        seeds = [42]
        logger.info("QUICK MODE: 1 beta, 1 seed, 5 epochs")
    else:
        betas = config.training.betas
        seeds = SEEDS

    loaders = create_dataloaders(splits, batch_size=config.training.batch_size)
    logger.info(
        f"Data loaded: train={len(loaders['train'].dataset)}, "
        f"val={len(loaders['val'].dataset)}, "
        f"test={len(loaders['test'].dataset)}"
    )

    # CSV logger
    csv_logger = ResultLogger(f"{args.output_dir}/vae_baselines.csv")

    # Run all experiments
    all_results = {}  # {beta: [MetricsResult, ...]}

    for beta in betas:
        all_results[beta] = []

        for seed in seeds:
            result = run_single_experiment(
                config, loaders, beta, seed, device,
                args.checkpoint_dir, args.fig_dir,
            )
            csv_logger.log(**{k: v for k, v in result.items()
                             if k != "metrics_result"})
            all_results[beta].append(result["metrics_result"])

    # Aggregate and report
    logger.info(f"\n{'='*60}")
    logger.info("AGGREGATED RESULTS (mean +/- std over seeds)")
    logger.info(f"{'='*60}")

    roc_results = {}
    pr_results = {}

    for beta in betas:
        agg = aggregate_seeds(all_results[beta])
        logger.info(f"\nbeta = {beta}:")
        logger.info(format_aggregated(agg))

        # Use first seed's result for ROC/PR curve plotting
        label = f"VAE (beta={beta})"
        roc_results[label] = all_results[beta][0]
        pr_results[label] = all_results[beta][0]

    # Generate ROC and PR curve figures
    if len(betas) > 1:
        plot_roc(roc_results,
                 title="VAE ROC Curves - Beta Comparison",
                 save_path=f"{args.fig_dir}/vae_roc_curves.pdf")
        plot_pr(pr_results,
                title="VAE PR Curves - Beta Comparison",
                save_path=f"{args.fig_dir}/vae_pr_curves.pdf")

    logger.info(f"\nResults saved to: {args.output_dir}/vae_baselines.csv")
    logger.info("Deliverables complete.")


if __name__ == "__main__":
    main()
