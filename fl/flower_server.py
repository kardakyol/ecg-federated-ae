"""
RAHEEB: Flower FL server — FedAvg simulation for ECG anomaly detection.
Updated for Sprint 2 completion and Sprint 3 metric alignment.
"""

import argparse
import logging
from typing import Dict, List, Tuple, Optional
import flwr as fl
from flwr.common import Context, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.simulation import start_simulation
from flwr.client import ClientApp
from fl.flower_client import ECGClient
from fl.strategies import Strategy
from pathlib import Path
from utils.csv_logger import ResultLogger
import matplotlib.pyplot as plt
from evaluation.plotting import COLORS
import os
import torch

# Environment stabilization for Ray/Flower simulation
os.environ["RAY_metrics_export_binaries_run_dir"] = ""
os.environ["RAY_DEDUP_LOGS"] = "0"

logger = logging.getLogger(__name__)

# ── Global Configs (Updated by Argparse) ───────────────────────────────────────────────────────
_NUM_ROUNDS = 3
_NUM_CLIENTS = 2
_LOCAL_EPOCHS = 1
_MODEL_TYPE = "vanilla"
_ALPHA = 0.5


# ── Factory to create ECGClient instances for the simulation ─────────────────────────────────────────────────────── 
def client_fn(cid: str):
    return ECGClient(
        client_id=cid,
        model_type=_MODEL_TYPE,
        alpha=_ALPHA,
    ).to_client()


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    global _NUM_ROUNDS, _NUM_CLIENTS, _LOCAL_EPOCHS, _MODEL_TYPE, _ALPHA

    parser = argparse.ArgumentParser(
        description="Flower Federated Learning Simulation"
    )
    parser.add_argument(
        "--rounds", type=int, default=3,
        help="Number of FL rounds (default: 3)"
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Local epochs per round (default: 1)"
    )
    parser.add_argument(
        "--model", type=str, default="vanilla",
        help="Model: vanilla, conv, or vae (default: vanilla)"
    )
    parser.add_argument(
        "--clients", type=int, default=10,
        help="Number of virtual clients (default: 10)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Verify setup (client creation, model, data) without running simulation"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Dirichlet alpha for non-IID split (default: 0.5)"
    )
    args = parser.parse_args()

    # Update module-level config
    _NUM_ROUNDS = args.rounds
    _NUM_CLIENTS = args.clients
    _LOCAL_EPOCHS = args.epochs
    _MODEL_TYPE = args.model
    _ALPHA = args.alpha

    if args.dry_run:
        # Verify everything initialises without error
        print(f"[dry-run] Model type : {_MODEL_TYPE}")
        print(f"[dry-run] Clients    : {_NUM_CLIENTS}")
        print(f"[dry-run] Rounds     : {_NUM_ROUNDS}")
        print(f"[dry-run] Epochs/rnd : {_LOCAL_EPOCHS}")

        test_client = ECGClient(client_id="dry-run", model_type=_MODEL_TYPE)
        params = test_client.get_parameters(config={})
        print(f"[dry-run] Model params: {sum(p.size for p in params):,} values")
        print(f"[dry-run] Model size : {test_client.model.model_size_mb():.2f} MB")
        print(f"[dry-run] Train set  : {len(test_client.loaders['train'].dataset)} samples")
        print(f"[dry-run] Val set    : {len(test_client.loaders['val'].dataset)} samples")

        # Quick single-batch forward pass
        batch = next(iter(test_client.loaders["train"]))
        x = batch[0]
        output = test_client.model(x)
        loss, *_ = test_client.model.compute_loss(x, output)
        print(f"[dry-run] Forward OK : input={tuple(x.shape)} -> output={tuple(output.x_hat.shape)}")
        print(f"[dry-run] Loss       : {loss.item():.6f}")
        print(f"[dry-run] All checks passed.")
        return

    logger.info(
        f"Starting FL: {_NUM_CLIENTS} clients, {_NUM_ROUNDS} rounds, "
        f"model={_MODEL_TYPE}, epochs/round={_LOCAL_EPOCHS}"
    )

    # Strategy configuration for Sprint 2 (30% client participation 
    strategy = Strategy(
        fraction_fit=0.3,
        fraction_evaluate=0.3,
        min_fit_clients=3,
        min_available_clients=_NUM_CLIENTS,
        on_fit_config_fn=lambda _: {
            "local_epochs": _LOCAL_EPOCHS,
            "model_type": _MODEL_TYPE,
        },
    )

    has_gpu = torch.cuda.is_available()
    client_res = {
        "num_cpus": 4,
        "num_gpus": 0.5 if has_gpu else 0.0
    }

    logger.info(f"Starting FL simulation: {_NUM_ROUNDS} rounds...")
    history = start_simulation(
        client_fn=client_fn,
        num_clients=_NUM_CLIENTS,
        config=ServerConfig(num_rounds=_NUM_ROUNDS),
        strategy=strategy,
        client_resources=client_res,
        ray_init_args={
            "num_gpus": 1 if has_gpu else 0,
            "include_dashboard": False,
            "_temp_dir": "C:\\temp"
        }
    )

    if history is None:
        logger.error("Simulation returned none")
        return

    # ── Result Processing & Logging ───────────────────────────────────────────────────────
    output_dir = Path("outputs")
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    dist_metrics = getattr(history, "metrics_distributed", {})
    auroc_key = next((k for k in dist_metrics.keys() if "auroc" in k.lower()), None) 
    
    if auroc_key:
        data = dist_metrics[auroc_key]
        rounds = [item[0] for item in data]
        aurocs = [item[1] for item in data]

        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(rounds, aurocs, color=COLORS[0], marker='o', lw=1.5, 
                label=f"{_MODEL_TYPE}")
        ax.set(xlabel="FL Round", ylabel="Weighted AUROC",
               title=f"Federated Convergence")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plot_path = fig_dir / f"convergence_{_MODEL_TYPE}_alpha{_ALPHA}.png"
        fig.savefig(plot_path) 
        plt.close(fig)
        logger.info(f"Convergence plot saved to {plot_path}")
    
    if dist_metrics:
        final_metrics_summary = {k: v[-1][1] for k, v in dist_metrics.items()}

        logger_csv = ResultLogger(
            output_dir / "convergence_results.csv",
            extra_columns=[
                "epochs", "rounds", "auprc", "f1", "sensitivity", "specificity",
                "precision_score", "f1", "model_size_mb", "flops_m", 
                "inference_latency_ms", "peak_memory_mb", "training_time_s"
            ]
        )
        logger_csv.log(
            model=_MODEL_TYPE,
            setting="federated",
            beta=_ALPHA,
            epochs=_LOCAL_EPOCHS,
            rounds=_NUM_ROUNDS,
            **final_metrics_summary
        )
        logger.info(f"Results logged to: {output_dir}/convergence_results.csv")
        
    logger.info("FL simulation complete.")


if __name__ == "__main__":
    from utils.reproducibility import setup_logging
    setup_logging()
    main()
