"""
CONV-AE BOTTLENECK ABLATION — Run AFTER diagnose_data.py
==========================================================
Tests bottleneck = {8, 16, 32, 64, 128} × 3 seeds for ConvAE.

Hypothesis: bottleneck=32 is too large — the model memorises both
normal and abnormal patterns, killing anomaly separation.
Tighter bottlenecks (8, 16) force the model to learn only the
dominant normal manifold, increasing MSE for anomalies.

Also includes learning rate sweep and epoch tuning.

USAGE (Colab):
    # Basic ablation (bottleneck only):
    !python convae_ablation.py --data_dir data/ptb-xl

    # Quick test (1 seed, fewer epochs):
    !python convae_ablation.py --data_dir data/ptb-xl --quick

    # Full sweep with LR:
    !python convae_ablation.py --data_dir data/ptb-xl --sweep_lr

    # Synthetic data test:
    !python convae_ablation.py --synthetic --quick

OUTPUT: outputs/convae_ablation_results.csv + console summary
"""

import argparse
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ── Project imports (adjust if needed) ──
try:
    from models.conv_ae import ConvAE
    from models.base import AEOutput
    from utils.dataset import create_synthetic_data, create_dataloaders, load_splits
    from utils.reproducibility import SEEDS, set_seed, get_device
    from evaluation.metrics import compute_metrics
    HAS_PROJECT = True
except ImportError:
    HAS_PROJECT = False
    print("WARNING: Could not import project modules.")
    print("Running in standalone mode with inline ConvAE definition.\n")


# ── Standalone ConvAE (if project imports fail) ──
if not HAS_PROJECT:
    from dataclasses import dataclass

    @dataclass
    class AEOutput:
        x_hat: torch.Tensor

    SEEDS = [42, 123, 456]

    def set_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class ConvAE(nn.Module):
        def __init__(self, bottleneck=32, n_leads=12, seq_len=1000):
            super().__init__()
            self.bottleneck = bottleneck
            self.n_leads = n_leads
            self.seq_len = seq_len

            # Encoder
            self.enc_conv1 = nn.Conv1d(n_leads, 32, kernel_size=7, stride=2, padding=3)
            self.enc_gn1 = nn.GroupNorm(min(32, 32), 32)
            self.enc_conv2 = nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3)
            self.enc_gn2 = nn.GroupNorm(min(32, 64), 64)
            self.enc_conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
            self.enc_gn3 = nn.GroupNorm(min(32, 128), 128)
            self.enc_conv4 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)
            self.enc_gn4 = nn.GroupNorm(min(32, 256), 256)

            curr_len = seq_len
            for k, s, p in [(7,2,3),(7,2,3),(5,2,2),(5,2,2)]:
                curr_len = math.floor((curr_len + 2*p - k) / s) + 1
            self._enc_temporal = curr_len
            self._enc_flat_dim = 256 * self._enc_temporal

            self.enc_fc = nn.Linear(self._enc_flat_dim, bottleneck)

            # Decoder
            self.dec_fc = nn.Linear(bottleneck, self._enc_flat_dim)
            self.dec_conv1 = nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
            self.dec_gn1 = nn.GroupNorm(min(32, 128), 128)
            self.dec_conv2 = nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
            self.dec_gn2 = nn.GroupNorm(min(32, 64), 64)
            self.dec_conv3 = nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1)
            self.dec_gn3 = nn.GroupNorm(min(32, 32), 32)
            self.dec_conv4 = nn.ConvTranspose1d(32, n_leads, kernel_size=7, stride=2, padding=3, output_padding=1)

        def forward(self, x):
            h = F.relu(self.enc_gn1(self.enc_conv1(x)))
            h = F.relu(self.enc_gn2(self.enc_conv2(h)))
            h = F.relu(self.enc_gn3(self.enc_conv3(h)))
            h = F.relu(self.enc_gn4(self.enc_conv4(h)))
            h_flat = h.view(h.shape[0], -1)
            z = F.relu(self.enc_fc(h_flat))

            h = F.relu(self.dec_fc(z))
            h = h.view(h.size(0), 256, -1)
            h = F.relu(self.dec_gn1(self.dec_conv1(h)))
            h = F.relu(self.dec_gn2(self.dec_conv2(h)))
            h = F.relu(self.dec_gn3(self.dec_conv3(h)))
            x_hat = self.dec_conv4(h)

            if x_hat.shape[-1] != x.shape[-1]:
                x_hat = F.interpolate(x_hat, size=x.shape[-1], mode='linear', align_corners=False)

            return AEOutput(x_hat=x_hat)

        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════

def train_centralised(model, train_loader, val_loader, epochs, lr, device,
                      patience=10, verbose=True):
    """Train ConvAE with early stopping on val MSE."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_mse = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = F.mse_loss(output.x_hat, x)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        avg_train = train_loss / max(n_batches, 1)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                output = model(x)
                loss = F.mse_loss(output.x_hat, x)
                val_loss += loss.item()
                n_val += 1

        avg_val = val_loss / max(n_val, 1)
        scheduler.step(avg_val)

        # Early stopping
        if avg_val < best_val_mse:
            best_val_mse = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train MSE: {avg_train:.6f} | Val MSE: {avg_val:.6f} | "
                  f"Best: {best_val_mse:.6f} | LR: {optimizer.param_groups[0]['lr']:.1e}")

        if no_improve >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return best_val_mse


def evaluate_model(model, test_loader, val_normal_loader, device):
    """Evaluate using MSE anomaly scoring with 95th percentile threshold."""
    model.eval()

    # Test scores
    all_scores, all_labels = [], []
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            output = model(signals)
            mse = torch.mean((output.x_hat - signals) ** 2, dim=(1, 2))
            all_scores.append(mse.cpu().numpy())
            all_labels.append(labels.numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    # Threshold from val normal
    normal_scores = []
    with torch.no_grad():
        for signals, lbl in val_normal_loader:
            signals = signals.to(device)
            output = model(signals)
            mse = torch.mean((output.x_hat - signals) ** 2, dim=(1, 2))
            normal_scores.append(mse.cpu().numpy())

    normal_scores = np.concatenate(normal_scores)
    threshold = float(np.percentile(normal_scores, 95))

    # Score debug
    normal_s = scores[labels == 0]
    abnormal_s = scores[labels == 1]
    sep = (abnormal_s.mean() - normal_s.mean()) / max(normal_s.std(), 1e-8)

    print(f"    Score Debug: Normal mean={normal_s.mean():.6f} std={normal_s.std():.6f}")
    print(f"                 Abnormal mean={abnormal_s.mean():.6f} std={abnormal_s.std():.6f}")
    print(f"                 Separation: {sep:.3f} std")
    print(f"                 Threshold (95th pctl): {threshold:.6f}")

    if HAS_PROJECT:
        result = compute_metrics(labels, scores, threshold)
        return {
            "auroc": result.auroc,
            "auprc": result.auprc,
            "sensitivity": result.sensitivity,
            "specificity": result.specificity,
            "f1": result.f1,
            "separation": sep,
            "threshold": threshold,
        }
    else:
        # Standalone metrics
        from sklearn.metrics import roc_auc_score, average_precision_score
        preds = (scores > threshold).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()

        auroc = roc_auc_score(labels, scores)
        auprc = average_precision_score(labels, scores)
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * sens / max(prec + sens, 1e-8)

        return {
            "auroc": auroc, "auprc": auprc,
            "sensitivity": float(sens), "specificity": float(spec),
            "f1": float(f1), "separation": sep, "threshold": threshold,
        }


# ══════════════════════════════════════════════════════════════
# DATA LOADING HELPERS
# ══════════════════════════════════════════════════════════════

def load_data(args):
    """Load data with project utils or standalone numpy."""
    if args.synthetic:
        if HAS_PROJECT:
            return create_synthetic_data(n_train=2000, n_val=500, n_test=500)
        else:
            return _create_synthetic_standalone()

    if HAS_PROJECT:
        return load_splits(args.data_dir)
    else:
        return _load_splits_standalone(args.data_dir)


def _create_synthetic_standalone():
    """Standalone synthetic data for testing."""
    from torch.utils.data import TensorDataset
    def make_ds(n, abnormal_frac=0.2):
        n_abn = int(n * abnormal_frac)
        n_nor = n - n_abn
        normal = torch.randn(n_nor, 12, 1000) * 0.5
        abnormal = torch.randn(n_abn, 12, 1000) * 0.5 + 0.3  # shifted
        signals = torch.cat([normal, abnormal])
        labels = torch.cat([torch.zeros(n_nor), torch.ones(n_abn)])
        return TensorDataset(signals, labels)
    return {"train": make_ds(2000, 0.0), "val": make_ds(500, 0.2), "test": make_ds(500, 0.3)}


def _load_splits_standalone(data_dir):
    """Standalone data loading matching common .npy patterns."""
    from torch.utils.data import TensorDataset
    data_dir = Path(data_dir)

    patterns = [
        ("train_signals.npy", "train_labels.npy",
         "val_signals.npy", "val_labels.npy",
         "test_signals.npy", "test_labels.npy"),
        ("X_train.npy", "y_train.npy",
         "X_val.npy", "y_val.npy",
         "X_test.npy", "y_test.npy"),
    ]

    for p in patterns:
        if all((data_dir / f).exists() for f in p):
            X_tr = torch.FloatTensor(np.load(data_dir / p[0]))
            y_tr = torch.FloatTensor(np.load(data_dir / p[1]))
            X_va = torch.FloatTensor(np.load(data_dir / p[2]))
            y_va = torch.FloatTensor(np.load(data_dir / p[3]))
            X_te = torch.FloatTensor(np.load(data_dir / p[4]))
            y_te = torch.FloatTensor(np.load(data_dir / p[5]))
            return {
                "train": TensorDataset(X_tr, y_tr),
                "val": TensorDataset(X_va, y_va),
                "test": TensorDataset(X_te, y_te),
            }

    raise FileNotFoundError(f"No recognized data files in {data_dir}")


def make_loaders(splits, batch_size=64):
    """Create dataloaders, including val_normal for threshold."""
    if HAS_PROJECT:
        return create_dataloaders(splits, batch_size=batch_size)

    from torch.utils.data import DataLoader, TensorDataset

    loaders = {}
    for name, ds in splits.items():
        loaders[name] = DataLoader(ds, batch_size=batch_size,
                                   shuffle=(name == "train"))

    # val_normal: filter val set to label==0 only
    val_ds = splits["val"]
    if hasattr(val_ds, 'tensors'):
        signals, labels = val_ds.tensors
        mask = labels == 0
        val_normal = TensorDataset(signals[mask], labels[mask])
        loaders["val_normal"] = DataLoader(val_normal, batch_size=batch_size)
    else:
        loaders["val_normal"] = loaders["val"]  # fallback

    return loaders


# ══════════════════════════════════════════════════════════════
# MAIN ABLATION
# ══════════════════════════════════════════════════════════════

def run_ablation(args):
    device = get_device()
    print(f"Device: {device}")

    # Load data
    splits = load_data(args)
    loaders = make_loaders(splits, batch_size=64)

    print(f"Train: {len(splits['train'])} samples")
    print(f"Val:   {len(splits['val'])} samples")
    print(f"Test:  {len(splits['test'])} samples")

    # Config
    bottlenecks = args.bottlenecks
    lrs = args.lrs if args.sweep_lr else [args.lr]
    seeds = [SEEDS[0]] if args.quick else SEEDS
    epochs = args.epochs

    print(f"\nAblation config:")
    print(f"  Bottlenecks: {bottlenecks}")
    print(f"  LRs: {lrs}")
    print(f"  Seeds: {seeds}")
    print(f"  Max epochs: {epochs}")

    # Results storage
    results = []
    csv_lines = ["bottleneck,lr,seed,auroc,auprc,sensitivity,specificity,f1,separation,threshold,params,train_time_s"]

    for bn in bottlenecks:
        for lr in lrs:
            seed_results = []
            for seed in seeds:
                set_seed(seed)
                print(f"\n{'─'*50}")
                print(f"ConvAE | bottleneck={bn} | lr={lr} | seed={seed}")
                print(f"{'─'*50}")

                model = ConvAE(bottleneck=bn).to(device)
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"  Parameters: {n_params:,}")

                t0 = time.time()
                best_val = train_centralised(
                    model, loaders["train"], loaders["val"],
                    epochs=epochs, lr=lr, device=device,
                    patience=15, verbose=True
                )
                train_time = time.time() - t0
                print(f"  Training time: {train_time:.1f}s | Best val MSE: {best_val:.6f}")

                metrics = evaluate_model(model, loaders["test"], loaders["val_normal"], device)
                print(f"  AUROC: {metrics['auroc']:.4f} | AUPRC: {metrics['auprc']:.4f} | "
                      f"F1: {metrics['f1']:.4f}")

                entry = {
                    "bottleneck": bn, "lr": lr, "seed": seed,
                    **metrics, "params": n_params, "train_time_s": train_time,
                }
                results.append(entry)
                seed_results.append(metrics)

                csv_lines.append(
                    f"{bn},{lr},{seed},{metrics['auroc']:.6f},{metrics['auprc']:.6f},"
                    f"{metrics['sensitivity']:.6f},{metrics['specificity']:.6f},"
                    f"{metrics['f1']:.6f},{metrics['separation']:.4f},"
                    f"{metrics['threshold']:.6f},{n_params},{train_time:.1f}"
                )

            # Per-config summary
            if len(seed_results) > 1:
                aurocs = [r["auroc"] for r in seed_results]
                print(f"\n  >>> bn={bn} lr={lr}: "
                      f"AUROC = {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")

    # ── Save CSV ──
    os.makedirs("outputs", exist_ok=True)
    csv_path = "outputs/convae_ablation_results.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"\nResults saved to {csv_path}")

    # ── Final Summary ──
    print(f"\n{'='*70}")
    print(f"CONVAE ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'BN':>4s} | {'LR':>8s} | {'AUROC':>14s} | {'AUPRC':>14s} | {'F1':>14s} | {'Sep':>8s}")
    print(f"{'─'*4}-+-{'─'*8}-+-{'─'*14}-+-{'─'*14}-+-{'─'*14}-+-{'─'*8}")

    # Group by (bn, lr)
    from itertools import groupby
    for (bn, lr), grp in groupby(results, key=lambda r: (r["bottleneck"], r["lr"])):
        grp = list(grp)
        aurocs = [r["auroc"] for r in grp]
        auprcs = [r["auprc"] for r in grp]
        f1s = [r["f1"] for r in grp]
        seps = [r["separation"] for r in grp]

        if len(grp) > 1:
            auroc_str = f"{np.mean(aurocs):.4f}±{np.std(aurocs):.4f}"
            auprc_str = f"{np.mean(auprcs):.4f}±{np.std(auprcs):.4f}"
            f1_str = f"{np.mean(f1s):.4f}±{np.std(f1s):.4f}"
            sep_str = f"{np.mean(seps):.2f}"
        else:
            auroc_str = f"{aurocs[0]:.4f}"
            auprc_str = f"{auprcs[0]:.4f}"
            f1_str = f"{f1s[0]:.4f}"
            sep_str = f"{seps[0]:.2f}"

        print(f"{bn:4d} | {lr:8.1e} | {auroc_str:>14s} | {auprc_str:>14s} | {f1_str:>14s} | {sep_str:>8s}")

    # Best config
    best = max(results, key=lambda r: r["auroc"])
    print(f"\n🏆 Best: bottleneck={best['bottleneck']} lr={best['lr']} "
          f"seed={best['seed']} → AUROC={best['auroc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ConvAE Bottleneck Ablation")
    parser.add_argument("--data_dir", type=str, default="data/ptb-xl")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--bottlenecks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    parser.add_argument("--sweep_lr", action="store_true",
                        help="Also sweep learning rates")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1 seed, 30 epochs")
    args = parser.parse_args()

    if args.seeds:
        SEEDS = args.seeds
    if args.quick:
        args.epochs = 30

    run_ablation(args)
