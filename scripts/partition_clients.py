"""
Non-IID Dirichlet client partitioning for federated learning.

Partitions the training set across K clients using a Dirichlet distribution
to create heterogeneous (non-IID) label distributions. Saves per-client
index arrays and generates a distribution histogram for the paper.

Usage:
    python scripts/partition_clients.py --data_dir data/ptb-xl --alpha 0.5 --num_clients 10
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 12, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
COLORS = ["#2196F3", "#FF9800"]


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> list[np.ndarray]:
    """Partition sample indices across clients using Dirichlet(alpha).

    For each class, draws a Dirichlet distribution over K clients and
    assigns samples proportionally. Lower alpha = more heterogeneous.

    Returns a list of K arrays, each containing indices into *labels*.
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(labels)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)

        proportions = rng.dirichlet([alpha] * num_clients)
        proportions = proportions / proportions.sum()

        splits = (proportions * len(cls_idx)).astype(int)
        remainder = len(cls_idx) - splits.sum()
        for i in range(remainder):
            splits[i % num_clients] += 1

        start = 0
        for k in range(num_clients):
            end = start + splits[k]
            client_indices[k].extend(cls_idx[start:end].tolist())
            start = end

    return [np.array(sorted(idx), dtype=np.int64) for idx in client_indices]


def plot_client_distribution(
    client_indices: list[np.ndarray],
    labels: np.ndarray,
    save_path: str | Path,
) -> None:
    """Bar chart: normal vs abnormal samples per client."""
    num_clients = len(client_indices)
    normal_counts = []
    abnormal_counts = []

    for idx in client_indices:
        client_labels = labels[idx]
        normal_counts.append(int((client_labels == 0).sum()))
        abnormal_counts.append(int((client_labels == 1).sum()))

    x = np.arange(num_clients)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, normal_counts, width, label="Normal", color=COLORS[0])
    ax.bar(x + width / 2, abnormal_counts, width, label="Abnormal", color=COLORS[1])

    ax.set_xlabel("Client ID")
    ax.set_ylabel("Number of Samples")
    ax.set_title(r"Per-Client Data Distribution (Dirichlet $\alpha$)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(num_clients)])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    log.info("Saved distribution plot to %s", save_path)


def main():
    parser = argparse.ArgumentParser(description="Non-IID Dirichlet client partitioning")
    parser.add_argument("--data_dir", type=str, default="data/ptb-xl",
                        help="Directory with preprocessed .npy files")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet concentration parameter (lower = more non-IID)")
    parser.add_argument("--num_clients", type=int, default=10,
                        help="Number of FL clients (K)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_fig", type=str, default="outputs/figures/client_distribution.pdf",
                        help="Path for the distribution histogram figure")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    labels_path = data_dir / "train_labels.npy"
    if not labels_path.exists():
        log.error("train_labels.npy not found in %s — run preprocess_ptbxl.py first", data_dir)
        return

    labels = np.load(labels_path)
    if (labels == 1).any():
        normal_idx = np.where(labels == 0)[0]
        labels = labels[normal_idx]

    # ---- Partition ----
    client_indices = dirichlet_partition(labels, args.num_clients, args.alpha, args.seed)

    # ---- Save per-client index files ----
    splits_dir = data_dir / "client_splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    partition_meta = {
        "alpha": args.alpha,
        "num_clients": args.num_clients,
        "seed": args.seed,
        "total_samples": int(len(labels)),
        "clients": {},
    }

    for k, idx in enumerate(client_indices):
        np.save(splits_dir / f"client_{k}_indices.npy", idx)
        client_labels = labels[idx]
        n_normal = int((client_labels == 0).sum())
        n_abnormal = int((client_labels == 1).sum())
        partition_meta["clients"][f"client_{k}"] = {
            "total": len(idx),
            "normal": n_normal,
            "abnormal": n_abnormal,
            "abnormal_ratio": round(n_abnormal / len(idx), 4) if len(idx) > 0 else 0.0,
        }
        log.info("Client %d: %d samples (%d normal, %d abnormal)", k, len(idx), n_normal, n_abnormal)

    meta_path = splits_dir / "partition_meta.json"
    with open(meta_path, "w") as f:
        json.dump(partition_meta, f, indent=2)
    log.info("Saved partition metadata to %s", meta_path)

    # ---- Plot ----
    plot_client_distribution(client_indices, labels, args.output_fig)
    log.info("Done. Client splits saved to %s", splits_dir)


if __name__ == "__main__":
    main()
