"""
BOTTLENECK ABLATION — Sprint 3 (Persons B + C)
================================================
Tests {8, 16, 32, 64, 128} for vanilla_ae, conv_ae, and vae.
Uses Sprint 3 optimised training (cosine annealing, grad clip, val MSE early stop).

Produces: outputs/ablation_bottleneck.csv

Usage:
    python training/ablation_bottleneck.py --data_dir data/ptb-xl-zscore
    python training/ablation_bottleneck.py --data_dir data/ptb-xl-zscore --model conv_ae
    python training/ablation_bottleneck.py --synthetic --epochs 20
"""

import argparse
import os
import time
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils.dataset import load_splits, create_synthetic_data, create_dataloaders
from utils.reproducibility import SEEDS, set_seed, get_device, setup_logging
from utils.csv_logger import ResultLogger
from evaluation.metrics import compute_metrics, aggregate_seeds, format_aggregated

from models.vanilla_ae import VanillaAE
from models.conv_ae import ConvAE
from models.vae import VAE


logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "vanilla_ae": VanillaAE,
    "conv_ae": ConvAE,
    "vae": VAE,
}

# Sprint 3: added bn=8 to ablation range
BOTTLENECK_SIZES = [8, 16, 32, 64, 128]


def compute_anomaly_scores(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            output = model(signals)
            per_sample_mse = ((output.x_hat - signals) ** 2).mean(dim=(1, 2))
            all_scores.append(per_sample_mse.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)


def find_threshold(model, val_normal_loader, device, percentile=95):
    model.eval()
    scores = []
    with torch.no_grad():
        for signals, labels in val_normal_loader:
            signals = signals.to(device)
            output = model(signals)
            mse = ((output.x_hat - signals) ** 2).mean(dim=(1, 2))
            scores.append(mse.cpu().numpy())
    return float(np.percentile(np.concatenate(scores), percentile))


def train_single(model_name, bottleneck, loaders, seed, device,
                 epochs=200, lr=1e-3, weight_decay=1e-5, patience=25):
    """Train one model config with Sprint 3 optimised loop."""
    set_seed(seed)

    model = MODEL_REGISTRY[model_name](bottleneck=bottleneck).to(device)

    # Sprint 3: CosineAnnealingWarmRestarts (matches max_auroc_pipeline)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    n_params = model.count_parameters()
    size_mb = model.model_size_mb()
    logger.info(f"  {model_name} | bn={bottleneck} | seed={seed} | "
                f"params={n_params:,} | size={size_mb:.2f} MB")

    best_val_mse = float('inf')
    best_state = None
    no_improve = 0
    train_start = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for batch_idx, (signals, labels) in enumerate(loaders["train"]):
            signals = signals.to(device)
            optimizer.zero_grad()
            output = model(signals)
            loss = model.compute_loss(signals, output)[0]
            loss.backward()
            # Sprint 3: gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(epoch + batch_idx / max(len(loaders["train"]), 1))
            epoch_losses.append(loss.item())

        # Sprint 3: validate on val MSE (not AUROC)
        model.eval()
        val_mse_sum, n_val = 0.0, 0
        with torch.no_grad():
            for signals, labels in loaders["val"]:
                signals = signals.to(device)
                output = model(signals)
                val_mse_sum += F.mse_loss(output.x_hat, signals).item()
                n_val += 1
        avg_val_mse = val_mse_sum / max(n_val, 1)

        if avg_val_mse < best_val_mse:
            best_val_mse = avg_val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 25 == 0:
            logger.info(f"    Epoch {epoch+1}/{epochs} | Loss: {np.mean(epoch_losses):.6f} | "
                        f"Val MSE: {avg_val_mse:.6f} | Best: {best_val_mse:.6f}")

        if no_improve >= patience:
            logger.info(f"    Early stop at epoch {epoch+1}")
            break

    train_time = time.time() - train_start
    if best_state:
        model.load_state_dict(best_state)

    test_scores, test_labels = compute_anomaly_scores(model, loaders["test"], device)
    test_threshold = find_threshold(model, loaders["val_normal"], device)
    test_result = compute_metrics(test_labels, test_scores, test_threshold)

    logger.info(f"    -> Test {test_result} | time={train_time:.1f}s")

    return test_result, size_mb, train_time


def main():
    parser = argparse.ArgumentParser(description="Bottleneck Ablation — Sprint 3")
    parser.add_argument("--data_dir", type=str, default="data/ptb-xl-zscore")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--model", type=str, default=None,
                        choices=["vanilla_ae", "conv_ae", "vae"])
    parser.add_argument("--bottlenecks", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    setup_logging()

    device = get_device()
    logger.info(f"Device: {device}")

    if args.synthetic:
        logger.info("Using SYNTHETIC data")
        splits = create_synthetic_data(n_train=2000, n_val=500, n_test=500)
    else:
        logger.info(f"Loading data from {args.data_dir}")
        splits = load_splits(args.data_dir)

    loaders = create_dataloaders(splits, batch_size=args.batch_size)

    # Sprint 3: all 3 models by default, bn 8-128
    models_to_run = [args.model] if args.model else ["vanilla_ae", "conv_ae", "vae"]
    bottlenecks = args.bottlenecks or BOTTLENECK_SIZES
    seeds = args.seeds or SEEDS

    os.makedirs("outputs", exist_ok=True)
    csv_logger = ResultLogger("outputs/ablation_bottleneck.csv",
                              extra_columns=["bottleneck"])

    logger.info(f"Models: {models_to_run}")
    logger.info(f"Bottlenecks: {bottlenecks}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Total runs: {len(models_to_run) * len(bottlenecks) * len(seeds)}")

    for model_name in models_to_run:
        for bn in bottlenecks:
            logger.info(f"\n{'='*60}")
            logger.info(f"{model_name} | bottleneck={bn}")
            logger.info(f"{'='*60}")

            results_per_seed = []

            for seed in seeds:
                result, size_mb, train_time = train_single(
                    model_name, bn, loaders, seed, device,
                    epochs=args.epochs, lr=args.lr,
                )
                results_per_seed.append(result)

                csv_logger.log(
                    model=model_name,
                    bottleneck=bn,
                    setting=f"ablation_bn{bn}",
                    seed=seed,
                    auroc=result.auroc,
                    auprc=result.auprc,
                    sensitivity=result.sensitivity,
                    specificity=result.specificity,
                    precision_score=result.precision,
                    f1=result.f1,
                    model_size_mb=size_mb,
                    training_time_s=train_time,
                )

            agg = aggregate_seeds(results_per_seed)
            logger.info(f"\n  Aggregated ({len(seeds)} seeds):")
            logger.info(format_aggregated(agg))

    logger.info(f"\nResults saved to outputs/ablation_bottleneck.csv")


if __name__ == "__main__":
    main()