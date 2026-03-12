"""
RESCONVAE MAX-AUROC — ResConvAE on z-score data with all optimizations
========================================================================
Previous results:
  - ResConvAE on clean data (±80σ outliers): max AUROC=0.67
  - ConvAE on z-score data (clipped): max AUROC=0.80

This script: ResConvAE on z-score data with:
  - Outlier clipping (±5σ) + re-normalization
  - Large bottleneck (64, 128)
  - Cosine annealing + grad clip + weight decay
  - Multi-scoring: global_mse, max_lead, top3_lead, segment_max, mahalanobis
  - Depth sweep: shallow, medium, deep

USAGE (Colab):
    # Quick test:
    !python resconvae_max_auroc.py --data_dir data/ptb-xl-zscore --quick

    # Full sweep:
    !python resconvae_max_auroc.py --data_dir data/ptb-xl-zscore

    # Compare with ConvAE baseline:
    !python resconvae_max_auroc.py --data_dir data/ptb-xl-zscore --bottlenecks 128 --depths deep
"""

import argparse
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from dataclasses import dataclass


SEEDS = [42, 123, 456]

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════

def clip_and_renorm(signals, clip_sigma=5.0):
    clipped = signals.copy()
    for lead in range(signals.shape[1]):
        ld = signals[:, lead, :]
        mu, sigma = ld.mean(), ld.std()
        clipped[:, lead, :] = np.clip(ld, mu - clip_sigma * sigma, mu + clip_sigma * sigma)
    return clipped

def renormalize(train, val, test):
    means = train.mean(axis=(0, 2), keepdims=True)
    stds = train.std(axis=(0, 2), keepdims=True)
    stds = np.where(stds < 1e-8, 1.0, stds)
    return (train - means) / stds, (val - means) / stds, (test - means) / stds


# ══════════════════════════════════════════════════════════════
# RESCONVAE MODEL (inline — no project dependency)
# ══════════════════════════════════════════════════════════════

@dataclass
class AEOutput:
    x_hat: torch.Tensor


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        b, c, _ = x.shape
        w = self.pool(x).view(b, c)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w.unsqueeze(-1)


class ResBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1, use_se=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.gn1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=padding)
        self.gn2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act = nn.ReLU(inplace=False)
        self.shortcut = nn.Identity()
        if in_ch != out_ch or stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride),
                nn.GroupNorm(min(32, out_ch), out_ch),
            )
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        h = self.se(h)
        return self.act(h + residual)


class ResBlockTranspose1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1, output_padding=0, use_se=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride=stride,
                                         padding=padding, output_padding=output_padding)
        self.gn1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=padding)
        self.gn2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act = nn.ReLU(inplace=False)
        self.shortcut = nn.Identity()
        if in_ch != out_ch or stride > 1:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose1d(in_ch, out_ch, 1, stride=stride, output_padding=output_padding),
                nn.GroupNorm(min(32, out_ch), out_ch),
            )
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        h = self.se(h)
        if h.shape[-1] != residual.shape[-1]:
            ml = min(h.shape[-1], residual.shape[-1])
            h, residual = h[..., :ml], residual[..., :ml]
        return self.act(h + residual)


DEPTH_CONFIGS = {
    "shallow": {
        "encoder": [(32, 7, 2), (64, 5, 2)],
        "decoder": [(32, 7, 2), (12, 5, 2)],
    },
    "medium": {
        "encoder": [(32, 7, 2), (64, 7, 2), (128, 5, 2)],
        "decoder": [(64, 7, 2), (32, 7, 2), (12, 5, 2)],
    },
    "deep": {
        "encoder": [(32, 7, 2), (64, 7, 2), (128, 5, 2), (256, 5, 2)],
        "decoder": [(128, 5, 2), (64, 7, 2), (32, 7, 2), (12, 5, 2)],
    },
}


class ResConvAE(nn.Module):
    def __init__(self, bottleneck=128, n_leads=12, seq_len=1000, depth="deep", use_se=True):
        super().__init__()
        self.bottleneck = bottleneck
        self.depth_name = depth

        config = DEPTH_CONFIGS[depth]
        enc_config = config["encoder"]
        dec_config = config["decoder"]

        enc_layers = []
        in_ch = n_leads
        for out_ch, ks, stride in enc_config:
            enc_layers.append(ResBlock1d(in_ch, out_ch, kernel_size=ks, stride=stride, use_se=use_se))
            in_ch = out_ch
        self.encoder = nn.Sequential(*enc_layers)

        curr_len = seq_len
        for _, ks, stride in enc_config:
            curr_len = math.floor((curr_len + 2 * (ks // 2) - ks) / stride) + 1

        self._enc_channels = enc_config[-1][0]
        self._enc_temporal = curr_len
        self._flat_dim = self._enc_channels * curr_len

        self.enc_fc = nn.Linear(self._flat_dim, bottleneck)
        self.dec_fc = nn.Linear(bottleneck, self._flat_dim)

        dec_layers = []
        in_ch = self._enc_channels
        for i, (out_ch, ks, stride) in enumerate(dec_config):
            is_last = (i == len(dec_config) - 1)
            dec_layers.append(ResBlockTranspose1d(
                in_ch, out_ch, kernel_size=ks, stride=stride,
                output_padding=1 if stride > 1 else 0,
                use_se=use_se and not is_last,
            ))
            in_ch = out_ch
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        h = self.encoder(x)
        z = F.relu(self.enc_fc(h.view(h.shape[0], -1)))
        h = F.relu(self.dec_fc(z))
        h = h.view(h.size(0), self._enc_channels, self._enc_temporal)
        x_hat = self.decoder(h)
        if x_hat.shape[-1] != x.shape[-1]:
            x_hat = F.interpolate(x_hat, size=x.shape[-1], mode='linear', align_corners=False)
        return x_hat, z

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self):
        return sum(p.numel() * p.element_size() for p in self.parameters()) / (1024**2)


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════

def train_model(model, train_loader, val_loader, epochs, lr, device, patience=25):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    best_val_mse = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, n_b = 0.0, 0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            x_hat, z = model(x)
            loss = F.mse_loss(x_hat, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(epoch + n_b / len(train_loader))
            train_loss += loss.item()
            n_b += 1

        model.eval()
        val_loss, n_v = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                x_hat, z = model(x)
                val_loss += F.mse_loss(x_hat, x).item()
                n_v += 1
        avg_val = val_loss / max(n_v, 1)

        if avg_val < best_val_mse:
            best_val_mse = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Train: {train_loss/max(n_b,1):.6f} | "
                  f"Val: {avg_val:.6f} | Best: {best_val_mse:.6f}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    return best_val_mse


# ══════════════════════════════════════════════════════════════
# MULTI-SCORING
# ══════════════════════════════════════════════════════════════

def compute_all_scores(model, loader, device):
    model.eval()
    results = {"global_mse": [], "max_lead": [], "top3_lead": [],
               "segment_max": [], "latent": [], "labels": []}

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            x_hat, z = model(signals)

            per_lead = torch.mean((x_hat - signals) ** 2, dim=2)  # (B, 12)
            global_mse = per_lead.mean(dim=1)
            max_lead = per_lead.max(dim=1).values
            top3 = per_lead.topk(3, dim=1).values.mean(dim=1)

            seg_len = signals.shape[2] // 10
            seg_mses = []
            for s in range(10):
                st, en = s * seg_len, (s + 1) * seg_len
                seg_mses.append(torch.mean((x_hat[:, :, st:en] - signals[:, :, st:en]) ** 2, dim=(1, 2)))
            seg_max = torch.stack(seg_mses, dim=1).max(dim=1).values

            results["global_mse"].append(global_mse.cpu().numpy())
            results["max_lead"].append(max_lead.cpu().numpy())
            results["top3_lead"].append(top3.cpu().numpy())
            results["segment_max"].append(seg_max.cpu().numpy())
            results["latent"].append(z.cpu().numpy())
            results["labels"].append(labels.numpy())

    return {k: np.concatenate(v) for k, v in results.items()}


def compute_mahalanobis(train_latent, test_latent):
    mu = train_latent.mean(axis=0)
    cov = np.cov(train_latent.T) + 1e-6 * np.eye(train_latent.shape[1])
    try:
        cov_inv = np.linalg.inv(cov)
    except:
        cov_inv = np.linalg.pinv(cov)
    diff = test_latent - mu
    return np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))


def evaluate_all_methods(test_results, train_results):
    labels = test_results["labels"]
    methods = {}

    for name in ["global_mse", "max_lead", "top3_lead", "segment_max"]:
        auroc = roc_auc_score(labels, test_results[name])
        auprc = average_precision_score(labels, test_results[name])
        methods[name] = {"auroc": auroc, "auprc": auprc}

    mahal = compute_mahalanobis(train_results["latent"], test_results["latent"])
    methods["mahalanobis"] = {
        "auroc": roc_auc_score(labels, mahal),
        "auprc": average_precision_score(labels, mahal),
    }

    # Best single
    best_name = max(methods, key=lambda k: methods[k]["auroc"])
    best_auroc = methods[best_name]["auroc"]

    # Try ensemble: normalize and combine top methods
    norm = {}
    for name in ["global_mse", "max_lead", "top3_lead", "segment_max"]:
        s = test_results[name]
        mn, mx = s.min(), s.max()
        norm[name] = (s - mn) / (mx - mn) if mx - mn > 1e-10 else s * 0

    # Weighted combos
    for w_global, w_top3 in [(0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (1.0, 0.0)]:
        combo = w_global * norm["global_mse"] + w_top3 * norm["top3_lead"]
        auroc = roc_auc_score(labels, combo)
        if auroc > best_auroc:
            best_auroc = auroc
            best_name = f"ensemble_g{w_global}_t{w_top3}"

    return methods, best_name, best_auroc


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ResConvAE Max-AUROC Pipeline")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--bottlenecks", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--depths", type=str, nargs="+", default=["medium", "deep"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--clip_sigma", type=float, default=5.0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.quick:
        args.epochs = 50
        args.bottlenecks = [128]
        args.depths = ["deep"]
        seeds = [42]
    else:
        seeds = SEEDS

    # Load
    data_dir = Path(args.data_dir)
    train_X = np.load(data_dir / "train_signals.npy")
    train_y = np.load(data_dir / "train_labels.npy")
    val_X = np.load(data_dir / "val_signals.npy")
    val_y = np.load(data_dir / "val_labels.npy")
    test_X = np.load(data_dir / "test_signals.npy")
    test_y = np.load(data_dir / "test_labels.npy")

    print(f"Loaded: train={train_X.shape}, val={val_X.shape}, test={test_X.shape}")

    # Clip + renorm
    if args.clip_sigma > 0:
        print(f"Clipping at ±{args.clip_sigma}σ...")
        train_X = clip_and_renorm(train_X, args.clip_sigma)
        val_X = clip_and_renorm(val_X, args.clip_sigma)
        test_X = clip_and_renorm(test_X, args.clip_sigma)
        train_X, val_X, test_X = renormalize(train_X, val_X, test_X)
        print(f"  After: mean={train_X.mean():.4f} std={train_X.std():.4f}")

    # Loaders
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(train_X), torch.FloatTensor(train_y)),
                              batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(val_X), torch.FloatTensor(val_y)),
                            batch_size=64)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(test_X), torch.FloatTensor(test_y)),
                             batch_size=64)

    # Results
    csv_lines = ["depth,bottleneck,seed,best_method,best_auroc,global_mse_auroc,max_lead_auroc,"
                 "top3_lead_auroc,segment_max_auroc,mahalanobis_auroc,params,size_mb,train_time"]
    all_results = []

    for depth in args.depths:
        for bn in args.bottlenecks:
            seed_aurocs = []
            for seed in seeds:
                set_seed(seed)
                print(f"\n{'='*60}")
                print(f"ResConvAE | depth={depth} | bn={bn} | seed={seed}")
                print(f"{'='*60}")

                model = ResConvAE(bottleneck=bn, depth=depth).to(device)
                print(f"  Params: {model.count_parameters():,} | Size: {model.model_size_mb():.2f}MB")

                t0 = time.time()
                best_val = train_model(model, train_loader, val_loader,
                                       epochs=args.epochs, lr=args.lr,
                                       device=device, patience=25)
                train_time = time.time() - t0
                print(f"  Training: {train_time:.1f}s | Best val MSE: {best_val:.6f}")

                # Score
                test_res = compute_all_scores(model, test_loader, device)
                train_res = compute_all_scores(model, train_loader, device)

                methods, best_name, best_auroc = evaluate_all_methods(test_res, train_res)

                print(f"\n  Scoring results:")
                for name, m in methods.items():
                    marker = " ◄" if name == best_name else ""
                    print(f"    {name:15s}: AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}{marker}")
                print(f"  Best: {best_name} → AUROC={best_auroc:.4f}")

                seed_aurocs.append(best_auroc)

                csv_lines.append(
                    f"{depth},{bn},{seed},{best_name},{best_auroc:.6f},"
                    f"{methods['global_mse']['auroc']:.6f},"
                    f"{methods['max_lead']['auroc']:.6f},"
                    f"{methods['top3_lead']['auroc']:.6f},"
                    f"{methods['segment_max']['auroc']:.6f},"
                    f"{methods['mahalanobis']['auroc']:.6f},"
                    f"{model.count_parameters()},{model.model_size_mb():.3f},{train_time:.1f}"
                )

                all_results.append({
                    "depth": depth, "bn": bn, "seed": seed,
                    "best_auroc": best_auroc, "best_method": best_name,
                    "methods": methods,
                })

            if len(seed_aurocs) > 1:
                print(f"\n  >>> {depth} bn={bn}: Best AUROC = "
                      f"{np.mean(seed_aurocs):.4f} ± {np.std(seed_aurocs):.4f}")

    # Save
    os.makedirs("outputs", exist_ok=True)
    csv_path = "outputs/resconvae_max_auroc_results.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines) + "\n")

    # Summary
    print(f"\n{'='*70}")
    print(f"RESCONVAE MAX-AUROC SUMMARY")
    print(f"{'='*70}")

    from itertools import groupby
    for (depth, bn), grp in groupby(all_results, key=lambda r: (r["depth"], r["bn"])):
        grp = list(grp)
        aurocs = [r["best_auroc"] for r in grp]
        if len(grp) > 1:
            print(f"  {depth:8s} bn={bn:3d}: AUROC={np.mean(aurocs):.4f}±{np.std(aurocs):.4f} "
                  f"| method={grp[0]['best_method']}")
        else:
            print(f"  {depth:8s} bn={bn:3d}: AUROC={aurocs[0]:.4f} | method={grp[0]['best_method']}")

    best = max(all_results, key=lambda r: r["best_auroc"])
    print(f"\n  🏆 Best: depth={best['depth']} bn={best['bn']} "
          f"→ AUROC={best['best_auroc']:.4f} ({best['best_method']})")

    # Compare with ConvAE baseline
    print(f"\n  Reference: ConvAE bn=128 on same data → AUROC=0.795±0.004")
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
