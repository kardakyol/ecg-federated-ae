"""
RAHEEB: Flower FL server — FedAvg simulation for ECG anomaly detection.

Usage:
    python fl/flower_server.py                                    # Sprint 1 defaults
    python fl/flower_server.py --rounds 50 --clients 10           # Sprint 2
    python fl/flower_server.py --dry-run                          # Verify setup only
"""

import argparse
import logging

import flwr as fl
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.client import ClientApp

from fl.flower_client import ECGClient

logger = logging.getLogger(__name__)


# ── Module-level config (updated by argparse before simulation) ──────
_NUM_ROUNDS = 50
_NUM_CLIENTS = 10
_LOCAL_EPOCHS = 5
_MODEL_TYPE = "conv"
_BETA = 0.5


# ── Client App ───────────────────────────────────────────────────────
def client_fn(context: Context):
    return ECGClient(
        client_id=str(context.node_config["partition-id"]),
        model_type=_MODEL_TYPE,
    ).to_client()


client_app = ClientApp(client_fn=client_fn)


# ── Server App ───────────────────────────────────────────────────────
def server_fn(context: Context):
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=_NUM_CLIENTS,
        min_available_clients=_NUM_CLIENTS,
        on_fit_config_fn=lambda _: {
            "local_epochs": _LOCAL_EPOCHS,
            "model_type": _MODEL_TYPE,
            "beta": _BETA,
        },
    )
    config = ServerConfig(num_rounds=_NUM_ROUNDS)
    return ServerAppComponents(strategy=strategy, config=config)


server_app = ServerApp(server_fn=server_fn)


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    global _NUM_ROUNDS, _NUM_CLIENTS, _LOCAL_EPOCHS, _MODEL_TYPE, _BETA

    parser = argparse.ArgumentParser(
        description="Flower Federated Learning Simulation"
    )
    parser.add_argument(
        "--rounds", type=int, default=50,
        help="Number of FL rounds (default: 50)"
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Local epochs per round (default: 5)"
    )
    parser.add_argument(
        "--model", type=str, default="conv",
        help="Model: vanilla, conv, or vae (default: conv)"
    )
    parser.add_argument(
        "--clients", type=int, default=10,
        help="Number of virtual clients (default: 10)"
    )
    parser.add_argument(
        "--beta", type=float, default=0.5,
        help="Beta for VAE loss (default: 0.5, ignored for non-VAE)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Verify setup (client creation, model, data) without running simulation"
    )
    args = parser.parse_args()

    # Update module-level config
    _NUM_ROUNDS = args.rounds
    _NUM_CLIENTS = args.clients
    _LOCAL_EPOCHS = args.epochs
    _MODEL_TYPE = args.model
    _BETA = args.beta

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

    fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=_NUM_CLIENTS,
    )

    logger.info("FL simulation complete.")


if __name__ == "__main__":
    from utils.reproducibility import setup_logging
    setup_logging()
    main()
