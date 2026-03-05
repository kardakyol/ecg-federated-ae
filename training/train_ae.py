import argparse
import os
import time

import numpy as np
import torch
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
    """Compute per-sample reconstruction error (MSE) as anomaly score.

    Higher score = more anomalous. Used as input to compute_metrics().

    Args:
        model: trained autoencoder (BaseAutoencoder subclass)
        loader: DataLoader yielding (signals, labels) batches
        device: torch device

    Returns:
        scores: np.ndarray (N,) — per-sample MSE anomaly scores
        labels: np.ndarray (N,) — ground truth labels (0=normal, 1=abnormal)
    """
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch_signals, batch_labels in loader:
            batch_signals = batch_signals.to(device)
            output = model(batch_signals)

            per_sample_mse = ((output.x_hat - batch_signals) ** 2).mean(dim=(1, 2))
            all_scores.append(per_sample_mse.cpu().numpy())
            all_labels.append(batch_labels.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)



def find_threshold(model, val_normal_loader, device, percentile=95):
    """Find anomaly threshold from normal validation samples.

    Uses the percentile of reconstruction errors on normal samples.
    Samples above this threshold are classified as anomalous.

    Args:
        model: trained autoencoder
        val_normal_loader: DataLoader with ONLY normal validation samples
                           (loaders["val_normal"] from create_dataloaders)
        device: torch device
        percentile: threshold percentile (default 95)

    Returns:
        threshold: float — reconstruction error threshold
    """
    model.eval()
    normal_scores = []

    with torch.no_grad():
        for batch_signals, batch_labels in val_normal_loader:
            batch_signals = batch_signals.to(device)
            output = model(batch_signals)
            per_sample_mse = ((output.x_hat - batch_signals) ** 2).mean(dim=(1, 2))
            normal_scores.append(per_sample_mse.cpu().numpy())

    normal_scores = np.concatenate(normal_scores)
    threshold = np.percentile(normal_scores, percentile)
    return threshold


def train_one_seed(model_name, config, loaders, seed, device, logger):
    """Train one model with one seed. Returns MetricsResult for aggregation.

    Args:
        model_name: "vanilla_ae" or "conv_ae"
        config: AEConfig instance
        loaders: dict from create_dataloaders()
        seed: random seed
        device: torch device
        logger: ResultLogger instance

    Returns:
        test_result: MetricsResult from compute_metrics()
    """
    set_seed(seed)

    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass(
        bottleneck=config.bottleneck,
        n_leads=config.n_leads,
        seq_len=config.seq_len,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config.scheduler_patience,
        factor=config.scheduler_factor,
    )

    print(f"\n{'='*60}")
    print(f"Training {model_name} | seed={seed} | params={model.count_parameters():,} | "
          f"size={model.model_size_mb():.2f} MB")
    print(f"{'='*60}")


    best_val_auroc = 0.0
    best_state = None
    train_start = time.time()

    for epoch in range(config.epochs):
        model.train()
        epoch_losses = []

        for batch_signals, batch_labels in loaders["train"]:
            batch_signals = batch_signals.to(device)

            optimizer.zero_grad()
            output = model(batch_signals)
            loss_tuple = model.compute_loss(batch_signals, output)
            loss = loss_tuple[0]  
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_train_loss = np.mean(epoch_losses)
        scheduler.step(avg_train_loss)


        val_scores, val_labels = compute_anomaly_scores(model, loaders["val"], device)
        val_threshold = find_threshold(model, loaders["val_normal"], device)
        val_result = compute_metrics(val_labels, val_scores, val_threshold)

        if val_result.auroc > best_val_auroc:
            best_val_auroc = val_result.auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{config.epochs} | "
                  f"Loss: {avg_train_loss:.6f} | "
                  f"Val AUROC: {val_result.auroc:.4f} | "
                  f"Val AUPRC: {val_result.auprc:.4f}")

    train_time = time.time() - train_start

   
    model.load_state_dict(best_state)
    print(f"  Best val AUROC: {best_val_auroc:.4f} | Training time: {train_time:.1f}s")

  
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(config.checkpoint_dir, f"{model_name}_seed{seed}.pt")
    torch.save(best_state, ckpt_path)


    test_scores, test_labels = compute_anomaly_scores(model, loaders["test"], device)
    test_threshold = find_threshold(model, loaders["val_normal"], device)
    test_result = compute_metrics(test_labels, test_scores, test_threshold)

    print(f"\n  Test Results (seed={seed}):")
    print(f"    AUROC:       {test_result.auroc:.4f}")
    print(f"    AUPRC:       {test_result.auprc:.4f}")
    print(f"    Sensitivity: {test_result.sensitivity:.4f}")
    print(f"    Specificity: {test_result.specificity:.4f}")
    print(f"    F1:          {test_result.f1:.4f}")
    print(f"    Precision:   {test_result.precision:.4f}")

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
    parser = argparse.ArgumentParser(description="Centralised AE training (Shardul)")
    parser.add_argument("--model", type=str, required=True,
                        choices=["vanilla_ae", "conv_ae"],
                        help="Which model to train")
    parser.add_argument("--data_dir", type=str, default="data/ptb-xl",
                        help="Path to preprocessed data")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data")
    parser.add_argument("--all_seeds", action="store_true",
                        help="Run all 3 seeds for mean +/- std")
    parser.add_argument("--bottleneck", type=int, default=32,
                        help="Bottleneck dimension (default: 32)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override default epoch count")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override default batch size")
    args = parser.parse_args()

    config = AEConfig(bottleneck=args.bottleneck)
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    
    device = get_device()
    print(f"Device: {device}")

   
    if args.synthetic:
        print("Using SYNTHETIC data")
        splits = create_synthetic_data(n_train=2000, n_val=500, n_test=500)
    else:
        print(f"Loading preprocessed data from {args.data_dir}")
        splits = load_splits(args.data_dir)

    loaders = create_dataloaders(splits, batch_size=config.batch_size)

    
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.figure_dir, exist_ok=True)

    csv_path = os.path.join(config.output_dir, "centralised_baselines.csv")
    logger = ResultLogger(csv_path)

    #  Training 
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
    plot_roc(
        {args.model: results_per_seed[-1]},
        save_path=f"{fig_prefix}_roc.pdf",
    )
    plot_pr(
        {args.model: results_per_seed[-1]},
        save_path=f"{fig_prefix}_pr.pdf",
    )
    print(f"\nFigures saved to {config.figure_dir}/")
    print(f"CSV results saved to {csv_path}")


if __name__ == "__main__":
    main()