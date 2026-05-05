"""
Centralised AE training script 
These changes match max_auroc_pipeline.py training loop exactly.
AUROC improvement: 0.60 → 0.795 (ConvAE, bn=128, z-score data).

Usage:
    python scripts/train_baseline.py --model conv_ae --data_dir data/ptb-xl-zscore --all_seeds
    python scripts/train_baseline.py --model vanilla_ae --data_dir data/ptb-xl-zscore --all_seeds
    python scripts/train_baseline.py --model conv_ae --synthetic
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils.dataset import load_splits, create_synthetic_data, create_dataloaders
from utils.reproducibility import SEEDS, set_seed, get_device
from utils.csv_logger import ResultLogger
from evaluation.metrics import compute_metrics, aggregate_seeds
from evaluation.plotting import plot_roc, plot_pr

from models.vanilla_ae import VanillaAE
from models.conv_ae import ConvAE
from configs.ae_config import AEConfig


MODEL_REGISTRY = {
    "vanilla_ae": VanillaAE,
    "conv_ae": ConvAE,
}


def compute_anomaly_scores(model, loader, device):
    """Per-sample MSE reconstruction error as anomaly score."""
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch_signals, batch_labels in loader:
            batch_signals = batch_signals.to(device)
            output = model(batch_signals)
            per_sample_mse = ((output.x_hat - batch_signals) ** 2).mean(dim=(1, 2))
            all_scores.append(per_sample_mse.cpu().numpy())
            all_labels.append(batch_labels.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)


def find_threshold(model, val_normal_loader, device, percentile=95):
    """Threshold from normal validation reconstruction errors."""
    model.eval()
    normal_scores = []
    with torch.no_grad():
        for batch_signals, batch_labels in val_normal_loader:
            batch_signals = batch_signals.to(device)
            output = model(batch_signals)
            per_sample_mse = ((output.x_hat - batch_signals) ** 2).mean(dim=(1, 2))
            normal_scores.append(per_sample_mse.cpu().numpy())
    return np.percentile(np.concatenate(normal_scores), percentile)


def train_one_seed(model_name, config, loaders, seed, device, logger):
    """Train one model with one seed using optimised loop."""
    set_seed(seed)

    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass(
        bottleneck=config.bottleneck,
        n_leads=config.n_leads,
        seq_len=config.seq_len,
    ).to(device)

    # CosineAnnealingWarmRestarts (from max_auroc_pipeline)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.cosine_T0,
        T_mult=config.cosine_T_mult,
        eta_min=config.cosine_eta_min,
    )

    print(f"\n{'='*60}")
    print(f"Training {model_name} | seed={seed} | bn={config.bottleneck} | "
          f"params={model.count_parameters():,} | size={model.model_size_mb():.2f} MB")
    print(f"{'='*60}")

    best_val_mse = float('inf')
    best_state = None
    no_improve = 0
    train_start = time.time()

    for epoch in range(config.epochs):
        # ── Train ──
        model.train()
        epoch_losses = []
        for batch_idx, (batch_signals, batch_labels) in enumerate(loaders["train"]):
            batch_signals = batch_signals.to(device)
            optimizer.zero_grad()
            output = model(batch_signals)
            loss_tuple = model.compute_loss(batch_signals, output)
            loss = loss_tuple[0]
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

            optimizer.step()
            scheduler.step(epoch + batch_idx / max(len(loaders["train"]), 1))
            epoch_losses.append(loss.item())

        avg_train_loss = np.mean(epoch_losses)

        # ── Validate on MSE (not AUROC — faster and more stable) ──
        model.eval()
        val_mse_sum, n_val = 0.0, 0
        with torch.no_grad():
            for batch_signals, batch_labels in loaders["val"]:
                batch_signals = batch_signals.to(device)
                output = model(batch_signals)
                val_mse_sum += F.mse_loss(output.x_hat, batch_signals).item()
                n_val += 1
        avg_val_mse = val_mse_sum / max(n_val, 1)

        # Early stopping on best_val_mse
        if avg_val_mse < best_val_mse:
            best_val_mse = avg_val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{config.epochs} | "
                  f"Loss: {avg_train_loss:.6f} | "
                  f"Val MSE: {avg_val_mse:.6f} | "
                  f"Best: {best_val_mse:.6f}")

        if no_improve >= config.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    train_time = time.time() - train_start

    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    print(f"  Best val MSE: {best_val_mse:.6f} | Training time: {train_time:.1f}s")

    # Save checkpoint
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(config.checkpoint_dir, f"{model_name}_seed{seed}.pt")
    torch.save(best_state or model.state_dict(), ckpt_path)

    # ── Test evaluation ──
    test_scores, test_labels = compute_anomaly_scores(model, loaders["test"], device)
    test_threshold = find_threshold(model, loaders["val_normal"], device)
    test_result = compute_metrics(test_labels, test_scores, test_threshold)

    # Score separation debug
    normal_s = test_scores[test_labels == 0]
    abnormal_s = test_scores[test_labels == 1]
    sep = (abnormal_s.mean() - normal_s.mean()) / max(normal_s.std(), 1e-8)

    print(f"\n  Test Results (seed={seed}):")
    print(f"    AUROC:       {test_result.auroc:.4f}")
    print(f"    AUPRC:       {test_result.auprc:.4f}")
    print(f"    F1:          {test_result.f1:.4f}")
    print(f"    Sensitivity: {test_result.sensitivity:.4f}")
    print(f"    Specificity: {test_result.specificity:.4f}")
    print(f"    Separation:  {sep:.3f} std")

    logger.log(
        model=model_name,
        setting="centralised",
        beta=None,
        epsilon=None,
        precision_type="fp32",
        seed=seed,
        auroc=test_result.auroc,
        auprc=test_result.auprc,
        sensitivity=test_result.sensitivity,
        specificity=test_result.specificity,
        precision_score=test_result.precision,
        f1=test_result.f1,
        model_size_mb=model.model_size_mb(),
        inference_latency_ms=None,
        training_time_s=train_time,
    )

    return test_result


def main():
    parser = argparse.ArgumentParser(description="Centralised AE training")
    parser.add_argument("--model", type=str, required=True,
                        choices=["vanilla_ae", "conv_ae"],
                        help="Which model to train")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to preprocessed data (default: from AEConfig)")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--all_seeds", action="store_true",
                        help="Run all 3 seeds for mean ± std")
    parser.add_argument("--bottleneck", type=int, default=None,
                        help="Bottleneck dimension (default: from AEConfig)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    config = AEConfig()
    if args.bottleneck is not None:
        config.bottleneck = args.bottleneck
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.data_dir is not None:
        config.data_dir = args.data_dir

    device = get_device()
    print(f"Device: {device}")

    if args.synthetic:
        print("Using SYNTHETIC data")
        splits = create_synthetic_data(n_train=2000, n_val=500, n_test=500)
    else:
        print(f"Loading data from {config.data_dir}")
        splits = load_splits(config.data_dir)

    loaders = create_dataloaders(splits, batch_size=config.batch_size)

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.figure_dir, exist_ok=True)

    csv_path = os.path.join(config.output_dir, "centralised_baselines.csv")
    logger = ResultLogger(csv_path)

    seeds = SEEDS if args.all_seeds else [SEEDS[0]]
    results_per_seed = []

    for seed in seeds:
        result = train_one_seed(args.model, config, loaders, seed, device, logger)
        results_per_seed.append(result)

    if len(seeds) > 1:
        agg = aggregate_seeds(results_per_seed)
        print(f"\n{'='*60}")
        print(f"AGGREGATED RESULTS for {args.model} ({len(seeds)} seeds)")
        print(f"{'='*60}")
        for metric in ["auroc", "auprc", "sensitivity", "specificity", "f1"]:
            mean_val = agg[metric]["mean"]
            std_val = agg[metric]["std"]
            print(f"  {metric.upper():15s}: {mean_val:.4f} +/- {std_val:.4f}")

    fig_prefix = os.path.join(config.figure_dir, f"{args.model}_centralised")
    plot_roc({args.model: results_per_seed[-1]}, save_path=f"{fig_prefix}_roc.pdf")
    plot_pr({args.model: results_per_seed[-1]}, save_path=f"{fig_prefix}_pr.pdf")
    print(f"\nFigures saved to {config.figure_dir}/")
    print(f"CSV results saved to {csv_path}")


if __name__ == "__main__":
    main()