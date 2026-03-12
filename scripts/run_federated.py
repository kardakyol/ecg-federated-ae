"""
KAAN + SHARDUL (Persons B + C) — Sprint 2/3: Federated AE Training
====================================================================
Sprint 3 updates:
  - bottleneck=128 (from ablation)
  - Gradient clipping in local_train (max_norm=1.0)
  - weight_decay=1e-5 in local optimizer
  - Default data_dir=data/ptb-xl-zscore

Flower simulation mode preserved exactly as-is for Raheeb.
Only manual FedAvg local_train and MODEL_REGISTRY changed.

USAGE:
    python scripts/run_federated.py --data_dir data/ptb-xl-zscore
    python scripts/run_federated.py --data_dir data/ptb-xl-zscore --model conv --quick
    python scripts/run_federated.py --data_dir data/ptb-xl-zscore --use-flower
"""

import argparse
import os
import time
import math
import numpy as np
import torch

from models.vanilla_ae import VanillaAE
from models.conv_ae import ConvAE
from models.vae import VAE
from models.base import BaseAutoencoder
from utils.dataset import (
    create_synthetic_data, create_dataloaders, load_splits, ECGDataset,
)
from utils.reproducibility import SEEDS, set_seed, get_device
from utils.csv_logger import ResultLogger
from evaluation.metrics import compute_metrics, aggregate_seeds, format_aggregated


# Sprint 3: bottleneck=128 default
MODEL_REGISTRY = {
    "vanilla": ("VanillaAE", lambda: VanillaAE(bottleneck=128)),
    "conv":    ("ConvAE",    lambda: ConvAE(bottleneck=128)),
    "vae":     ("VAE",       lambda: VAE()),
}


# ====================================================================
# Data Partitioning (unchanged)
# ====================================================================
def partition_iid(global_splits, num_clients, seed=42):
    rng = np.random.RandomState(seed)
    train_ds = global_splits["train"]
    n = len(train_ds)
    indices = rng.permutation(n)
    shard_size = n // num_clients

    client_splits = []
    for i in range(num_clients):
        start = i * shard_size
        end = start + shard_size if i < num_clients - 1 else n
        idx = indices[start:end]

        client_train = ECGDataset(
            train_ds.signals[idx].numpy(),
            train_ds.labels[idx].numpy(),
        )
        client_splits.append({
            "train": client_train,
            "val": global_splits["val"],
            "test": global_splits["test"],
        })

    return client_splits


# ====================================================================
# Manual FedAvg (Sprint 3: added grad clip + weight decay)
# ====================================================================
def fedavg_aggregate(client_params_list, client_sizes):
    total = sum(client_sizes)
    weights = [s / total for s in client_sizes]
    num_layers = len(client_params_list[0])
    return [
        sum(w * client_params_list[c][layer_idx]
            for c, w in enumerate(weights))
        for layer_idx in range(num_layers)
    ]


def local_train(model, loaders, epochs, model_type, beta=0.5, lr=0.001):
    """One client's local training. Sprint 3: added grad clip + weight decay."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    is_vae = model_type == "vae"
    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        model.train()
        kl_weight = min(1.0, (epoch + 1) / max(epochs, 1)) if is_vae else 1.0

        for batch in loaders["train"]:
            x = batch[0]
            optimizer.zero_grad()
            output = model(x)

            if is_vae:
                loss, *_ = model.compute_loss(x, output, beta=beta, kl_weight=kl_weight)
            else:
                loss, *_ = model.compute_loss(x, output)

            loss.backward()
            # Sprint 3: gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    return model.get_parameters(), len(loaders["train"].dataset), total_loss / max(num_batches, 1)


def evaluate_global_model(model, global_splits, device):
    model.eval()
    loaders = create_dataloaders(global_splits, batch_size=64)

    all_scores, all_labels = [], []
    with torch.no_grad():
        for signals, labels in loaders["test"]:
            signals = signals.to(device)
            output = model(signals)
            mse = torch.mean((output.x_hat - signals) ** 2, dim=(1, 2))
            all_scores.append(mse.cpu().numpy())
            all_labels.append(labels.numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    normal_scores = []
    with torch.no_grad():
        for signals, lbl in loaders["val_normal"]:
            signals = signals.to(device)
            output = model(signals)
            mse = torch.mean((output.x_hat - signals) ** 2, dim=(1, 2))
            normal_scores.append(mse.cpu().numpy())

    threshold = float(np.percentile(np.concatenate(normal_scores), 95))

    normal_s = scores[labels == 0]
    abnormal_s = scores[labels == 1]
    sep = (abnormal_s.mean() - normal_s.mean()) / max(normal_s.std(), 1e-8)
    print(f"\n=== SCORE DEBUG ===")
    print(f"Normal:   mean={normal_s.mean():.6f} std={normal_s.std():.6f}")
    print(f"Abnormal: mean={abnormal_s.mean():.6f} std={abnormal_s.std():.6f}")
    print(f"Separation: {sep:.3f} std")
    print(f"==================\n")
    return compute_metrics(labels, scores, threshold)


def run_manual_fedavg(model_type, seed, global_splits, num_clients,
                      num_rounds, local_epochs, beta, device):
    set_seed(seed)
    display_name, model_fn = MODEL_REGISTRY[model_type]

    global_model = model_fn().to(device)
    global_params = global_model.get_parameters()

    client_splits = partition_iid(global_splits, num_clients, seed=seed)
    client_loaders = [create_dataloaders(cs, batch_size=32) for cs in client_splits]

    print(f"\n{'='*60}")
    print(f"FL Training: {display_name} | seed={seed} | "
          f"K={num_clients} | R={num_rounds} | E={local_epochs}")
    print(f"  params={global_model.count_parameters():,} | "
          f"size={global_model.model_size_mb():.2f} MB")
    print(f"{'='*60}")

    round_losses = []
    t0 = time.time()

    for rnd in range(1, num_rounds + 1):
        client_results = []
        for c_idx in range(num_clients):
            local_model = model_fn().to(device)
            local_model.set_parameters(global_params)
            params, n, loss = local_train(
                local_model, client_loaders[c_idx],
                local_epochs, model_type, beta,
            )
            client_results.append((params, n, loss))

        all_params = [r[0] for r in client_results]
        all_sizes = [r[1] for r in client_results]
        all_losses = [r[2] for r in client_results]
        global_params = fedavg_aggregate(all_params, all_sizes)
        global_model.set_parameters(global_params)

        avg_loss = np.average(all_losses, weights=all_sizes)
        round_losses.append(avg_loss)

        if rnd % 10 == 0 or rnd == 1 or rnd == num_rounds:
            print(f"  Round {rnd:3d}/{num_rounds} | Avg loss: {avg_loss:.6f}")

    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    result = evaluate_global_model(global_model, global_splits, device)
    print(f"  Test: {result}")

    return result, round_losses, train_time


# ====================================================================
# Flower Simulation (UNCHANGED — Raheeb's code)
# ====================================================================
def run_flower_simulation(model_type, seed, global_splits, num_clients,
                          num_rounds, local_epochs, beta, device):
    """Run FedAvg using actual Flower simulation + ray backend."""
    import flwr as fl
    from flwr.common import Context
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
    from flwr.client import ClientApp
    from fl.flower_client import ECGClient

    set_seed(seed)
    display_name, model_fn = MODEL_REGISTRY[model_type]

    client_splits = partition_iid(global_splits, num_clients, seed=seed)

    print(f"\n{'='*60}")
    print(f"FL Training (Flower): {display_name} | seed={seed} | "
          f"K={num_clients} | R={num_rounds} | E={local_epochs}")
    print(f"{'='*60}")

    def client_fn(context: Context):
        pid = int(context.node_config["partition-id"])
        client = ECGClient(
            client_id=str(pid),
            model_type=model_type,
            data_splits=client_splits[pid],
            batch_size=32,
        )
        return client.to_client()

    client_app = ClientApp(client_fn=client_fn)

    _model_type = model_type
    _local_epochs = local_epochs
    _beta = beta
    _num_clients = num_clients

    def server_fn(context: Context):
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            min_fit_clients=_num_clients,
            min_available_clients=_num_clients,
            on_fit_config_fn=lambda _: {
                "local_epochs": _local_epochs,
                "beta": _beta,
            },
        )
        config = ServerConfig(num_rounds=num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    server_app = ServerApp(server_fn=server_fn)

    t0 = time.time()
    fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=num_clients,
    )
    train_time = time.time() - t0
    print(f"  Flower simulation time: {train_time:.1f}s")

    global_model = model_fn().to(device)
    print("  (Flower simulation complete — use manual mode for full eval)")
    result = evaluate_global_model(global_model, global_splits, device)
    return result, [], train_time


# ====================================================================
# Main (Sprint 3: default data_dir changed)
# ====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Sprint 2/3: Federated AE Training (Persons B + C)"
    )
    parser.add_argument("--model", type=str, default=None,
                        choices=["vanilla", "conv", "vae"])
    parser.add_argument("--data_dir", type=str, default="data/ptb-xl-zscore")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--use-flower", action="store_true")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    if args.synthetic:
        print("Using SYNTHETIC data")
        global_splits = create_synthetic_data(n_train=2000, n_val=500, n_test=500)
    else:
        print(f"Loading PTB-XL from {args.data_dir}")
        global_splits = load_splits(args.data_dir)

    for split_name in ["train", "val", "test"]:
        ds = global_splits[split_name]
        print(f"  {split_name}: {len(ds)} samples "
              f"({ds.n_normal} normal, {ds.n_abnormal} abnormal)")

    models_to_run = [args.model] if args.model else ["vanilla", "conv", "vae"]
    seeds = args.seeds or SEEDS
    num_rounds = 5 if args.quick else args.rounds
    if args.quick:
        seeds = [seeds[0]]
        print("QUICK MODE: 1 seed, 5 rounds")

    run_fn = run_flower_simulation if args.use_flower else run_manual_fedavg

    os.makedirs("outputs", exist_ok=True)
    csv_path = "outputs/federated_results.csv"
    logger = ResultLogger(csv_path)

    all_results = {}

    for model_type in models_to_run:
        display_name = MODEL_REGISTRY[model_type][0]
        model_results = []

        for seed in seeds:
            result, round_losses, train_time = run_fn(
                model_type=model_type,
                seed=seed,
                global_splits=global_splits,
                num_clients=args.clients,
                num_rounds=num_rounds,
                local_epochs=args.epochs,
                beta=args.beta,
                device=device,
            )
            model_results.append(result)

            logger.log(
                model=display_name,
                setting=f"federated_K{args.clients}_R{num_rounds}_E{args.epochs}",
                beta=args.beta if model_type == "vae" else None,
                epsilon=None,
                precision_type="fp32",
                seed=seed,
                auroc=result.auroc,
                auprc=result.auprc,
                sensitivity=result.sensitivity,
                specificity=result.specificity,
                precision_score=result.precision,
                f1=result.f1,
                model_size_mb=MODEL_REGISTRY[model_type][1]().model_size_mb(),
                inference_latency_ms=None,
                training_time_s=train_time,
            )

        all_results[display_name] = model_results

    if len(seeds) > 1:
        print(f"\n{'='*70}")
        print(f"FEDERATED RESULTS SUMMARY (K={args.clients}, R={num_rounds}, E={args.epochs})")
        print(f"{'='*70}")
        for model_name, results in all_results.items():
            agg = aggregate_seeds(results)
            print(f"\n{model_name}:")
            print(format_aggregated(agg))

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()