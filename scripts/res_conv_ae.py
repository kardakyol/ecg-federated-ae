"""
ResConvAE — Residual Convolutional Autoencoder for ECG Anomaly Detection
==========================================================================
Enhanced ConvAE with residual connections for better gradient flow
and deeper feature extraction.

Key improvements over vanilla ConvAE:
  1. Residual blocks (skip connections) → better gradient flow, deeper training
  2. Squeeze-and-Excitation (SE) → per-lead channel attention
  3. GroupNorm + inplace=False throughout → Opacus DP-SGD compatible
  4. Configurable depth: shallow (2 blocks) or deep (4 blocks)
  5. Gradual downsampling with strided convolutions

Compatible with existing project infrastructure:
  - Returns AEOutput(x_hat=...) from forward()
  - compute_loss() returns (mse,) tuple
  - get_parameters() / set_parameters() for Flower FL
  - GroupNorm for Opacus compatibility

USAGE:
    # As standalone test:
    !python res_conv_ae.py --test

    # Import in project:
    from models.res_conv_ae import ResConvAE
    model = ResConvAE(bottleneck=16, depth="medium")

    # Run ablation with this model:
    !python resconvae_ablation.py --data_dir data/ptb-xl-clean
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# ── Try project imports, fall back to standalone ──
try:
    from models.base import BaseAutoencoder, AEOutput
    HAS_BASE = True
except ImportError:
    HAS_BASE = False

    @dataclass
    class AEOutput:
        x_hat: torch.Tensor

    class BaseAutoencoder(nn.Module):
        def get_parameters(self):
            return [p.detach().cpu().clone() for p in self.parameters()]

        def set_parameters(self, params):
            for p, new_p in zip(self.parameters(), params):
                p.data = new_p.clone().to(p.device)

        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def model_size_mb(self):
            return sum(p.numel() * p.element_size() for p in self.parameters()) / (1024**2)


# ══════════════════════════════════════════════════════════════
# BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: learns per-channel importance weights.
    Helps the model focus on diagnostically relevant leads."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        # x: (B, C, T)
        b, c, _ = x.shape
        w = self.pool(x).view(b, c)        # (B, C)
        w = F.relu(self.fc1(w))             # (B, mid)
        w = torch.sigmoid(self.fc2(w))      # (B, C)
        return x * w.unsqueeze(-1)          # (B, C, T)


class ResBlock1d(nn.Module):
    """Residual block with 1D convolutions.

    Architecture:
        x → Conv → GN → ReLU → Conv → GN → (+x) → ReLU

    If in_ch != out_ch or stride > 1, a projection shortcut is used.
    """

    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1, use_se=True):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.gn1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act1 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=padding)
        self.gn2 = nn.GroupNorm(min(32, out_ch), out_ch)

        self.act_out = nn.ReLU(inplace=False)

        # Shortcut projection
        self.shortcut = nn.Identity()
        if in_ch != out_ch or stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.GroupNorm(min(32, out_ch), out_ch),
            )

        # Squeeze-and-Excitation
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        h = self.act1(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        h = self.se(h)
        return self.act_out(h + residual)


class ResBlockTranspose1d(nn.Module):
    """Residual block with transposed convolution for upsampling."""

    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1, output_padding=0, use_se=True):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride=stride,
                                         padding=padding, output_padding=output_padding)
        self.gn1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act1 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=padding)
        self.gn2 = nn.GroupNorm(min(32, out_ch), out_ch)

        self.act_out = nn.ReLU(inplace=False)

        # Shortcut
        self.shortcut = nn.Identity()
        if in_ch != out_ch or stride > 1:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=1, stride=stride,
                                    output_padding=output_padding),
                nn.GroupNorm(min(32, out_ch), out_ch),
            )

        self.se = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        h = self.act1(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        h = self.se(h)

        # Handle size mismatch from transposed conv
        if h.shape[-1] != residual.shape[-1]:
            min_len = min(h.shape[-1], residual.shape[-1])
            h = h[..., :min_len]
            residual = residual[..., :min_len]

        return self.act_out(h + residual)


# ══════════════════════════════════════════════════════════════
# RESCONVAE MODEL
# ══════════════════════════════════════════════════════════════

# Depth configurations: (channels, kernel_size, stride)
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


class ResConvAE(BaseAutoencoder):
    """Residual Convolutional Autoencoder for ECG anomaly detection.

    Args:
        bottleneck: Latent dimension size. Default 16.
        n_leads: Number of ECG leads. Default 12.
        seq_len: Timesteps per lead. Default 1000.
        depth: 'shallow', 'medium', or 'deep'.
        use_se: Use Squeeze-and-Excitation blocks.
    """

    def __init__(self, bottleneck=16, n_leads=12, seq_len=1000,
                 depth="medium", use_se=True):
        super().__init__()
        self.bottleneck = bottleneck
        self.n_leads = n_leads
        self.seq_len = seq_len
        self.depth_name = depth

        config = DEPTH_CONFIGS[depth]
        enc_config = config["encoder"]
        dec_config = config["decoder"]

        # ── Encoder ──
        enc_layers = []
        in_ch = n_leads
        for out_ch, ks, stride in enc_config:
            enc_layers.append(ResBlock1d(in_ch, out_ch, kernel_size=ks,
                                         stride=stride, use_se=use_se))
            in_ch = out_ch
        self.encoder = nn.Sequential(*enc_layers)

        # Compute encoded temporal dimension
        curr_len = seq_len
        for _, ks, stride in enc_config:
            padding = ks // 2
            curr_len = math.floor((curr_len + 2 * padding - ks) / stride) + 1

        self._enc_channels = enc_config[-1][0]
        self._enc_temporal = curr_len
        self._enc_flat_dim = self._enc_channels * self._enc_temporal

        # FC bottleneck
        self.enc_fc = nn.Linear(self._enc_flat_dim, bottleneck)

        # ── Decoder ──
        self.dec_fc = nn.Linear(bottleneck, self._enc_flat_dim)

        dec_layers = []
        in_ch = self._enc_channels
        for i, (out_ch, ks, stride) in enumerate(dec_config):
            is_last = (i == len(dec_config) - 1)
            dec_layers.append(ResBlockTranspose1d(
                in_ch, out_ch, kernel_size=ks, stride=stride,
                output_padding=1 if stride > 1 else 0,
                use_se=use_se and not is_last,  # no SE on final layer
            ))
            in_ch = out_ch

        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> AEOutput:
        """
        Args:
            x: (B, 12, 1000) channels-first ECG

        Returns:
            AEOutput with x_hat: (B, 12, 1000) — guaranteed same shape
        """
        # Encode
        h = self.encoder(x)                          # (B, C, T')
        h_flat = h.view(h.shape[0], -1)              # (B, C*T')
        z = F.relu(self.enc_fc(h_flat))              # (B, bottleneck)

        # Decode
        h = F.relu(self.dec_fc(z))                   # (B, C*T')
        h = h.view(h.size(0), self._enc_channels, self._enc_temporal)
        x_hat = self.decoder(h)                      # (B, 12, T'')

        # Guarantee output shape matches input
        if x_hat.shape[-1] != x.shape[-1]:
            x_hat = F.interpolate(x_hat, size=x.shape[-1],
                                   mode='linear', align_corners=False)

        return AEOutput(x_hat=x_hat)

    def compute_loss(self, x: torch.Tensor, output: AEOutput, **kwargs) -> tuple:
        """MSE reconstruction loss."""
        mse = F.mse_loss(output.x_hat, x)
        return (mse,)

    def __repr__(self):
        return (f"ResConvAE(bottleneck={self.bottleneck}, depth={self.depth_name}, "
                f"params={self.count_parameters():,})")


# ══════════════════════════════════════════════════════════════
# STANDALONE ABLATION
# ══════════════════════════════════════════════════════════════

def run_ablation():
    """Run bottleneck × depth ablation for ResConvAE."""
    import argparse
    import time
    import numpy as np
    from pathlib import Path

    parser = argparse.ArgumentParser(description="ResConvAE Ablation")
    parser.add_argument("--data_dir", type=str, default="data/ptb-xl-clean")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--bottlenecks", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--depths", type=str, nargs="+", default=["shallow", "medium", "deep"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--test", action="store_true", help="Just test forward pass")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Quick forward pass test
    if args.test:
        print("\n=== Forward Pass Test ===")
        for depth in ["shallow", "medium", "deep"]:
            for bn in [8, 16, 32, 64]:
                model = ResConvAE(bottleneck=bn, depth=depth).to(device)
                x = torch.randn(4, 12, 1000).to(device)
                out = model(x)
                loss = model.compute_loss(x, out)
                print(f"  {depth:8s} bn={bn:3d}: "
                      f"params={model.count_parameters():>10,} | "
                      f"size={model.model_size_mb():.2f}MB | "
                      f"out={out.x_hat.shape} | loss={loss[0].item():.4f}")
        print("\n✓ All forward passes OK")
        return

    # ── Load data ──
    SEEDS = [42, 123, 456]
    if args.quick:
        SEEDS = [42]
        args.epochs = 30

    try:
        from utils.dataset import load_splits, create_synthetic_data, create_dataloaders
        from utils.reproducibility import set_seed
        from evaluation.metrics import compute_metrics

        if args.synthetic:
            splits = create_synthetic_data(n_train=2000, n_val=500, n_test=500)
        else:
            splits = load_splits(args.data_dir)
        loaders = create_dataloaders(splits, batch_size=64)
        USE_PROJECT = True
    except ImportError:
        USE_PROJECT = False
        from torch.utils.data import DataLoader, TensorDataset

        def set_seed(seed):
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if args.synthetic:
            n_leads, seq_len = 12, 1000
            train_X = torch.randn(2000, n_leads, seq_len) * 0.5
            val_norm = torch.randn(300, n_leads, seq_len) * 0.5
            val_abn = torch.randn(200, n_leads, seq_len) * 0.5 + 0.5
            val_X = torch.cat([val_norm, val_abn])
            val_y = torch.cat([torch.zeros(300), torch.ones(200)])
            test_norm = torch.randn(300, n_leads, seq_len) * 0.5
            test_abn = torch.randn(200, n_leads, seq_len) * 0.5 + 0.5
            test_X = torch.cat([test_norm, test_abn])
            test_y = torch.cat([torch.zeros(300), torch.ones(200)])
        else:
            data_dir = Path(args.data_dir)
            train_X = torch.FloatTensor(np.load(data_dir / "train_signals.npy"))
            train_y = torch.FloatTensor(np.load(data_dir / "train_labels.npy"))
            val_X = torch.FloatTensor(np.load(data_dir / "val_signals.npy"))
            val_y = torch.FloatTensor(np.load(data_dir / "val_labels.npy"))
            test_X = torch.FloatTensor(np.load(data_dir / "test_signals.npy"))
            test_y = torch.FloatTensor(np.load(data_dir / "test_labels.npy"))

        loaders = {
            "train": DataLoader(TensorDataset(train_X, torch.zeros(len(train_X))),
                                batch_size=64, shuffle=True),
            "val": DataLoader(TensorDataset(val_X, val_y), batch_size=64),
            "test": DataLoader(TensorDataset(test_X, test_y), batch_size=64),
            "val_normal": DataLoader(
                TensorDataset(val_X[val_y == 0], val_y[val_y == 0]), batch_size=64),
        }

    print(f"\nAblation: bottlenecks={args.bottlenecks} × depths={args.depths} × seeds={SEEDS}")

    # ── Results ──
    csv_lines = ["depth,bottleneck,seed,auroc,auprc,f1,separation,params,size_mb,train_time_s"]
    results = []

    for depth in args.depths:
        for bn in args.bottlenecks:
            seed_aurocs = []
            for seed in SEEDS:
                set_seed(seed)
                print(f"\n{'─'*55}")
                print(f"ResConvAE | depth={depth} | bn={bn} | seed={seed}")
                print(f"{'─'*55}")

                model = ResConvAE(bottleneck=bn, depth=depth).to(device)
                print(f"  {model}")

                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=5)

                best_val_mse = float('inf')
                best_state = None
                no_improve = 0

                t0 = time.time()
                for epoch in range(1, args.epochs + 1):
                    model.train()
                    train_loss, n_b = 0.0, 0
                    for batch in loaders["train"]:
                        x = batch[0].to(device)
                        optimizer.zero_grad()
                        out = model(x)
                        loss = F.mse_loss(out.x_hat, x)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        train_loss += loss.item()
                        n_b += 1

                    avg_train = train_loss / max(n_b, 1)

                    model.eval()
                    val_loss, n_v = 0.0, 0
                    with torch.no_grad():
                        for batch in loaders["val"]:
                            x = batch[0].to(device)
                            out = model(x)
                            val_loss += F.mse_loss(out.x_hat, x).item()
                            n_v += 1
                    avg_val = val_loss / max(n_v, 1)
                    scheduler.step(avg_val)

                    if avg_val < best_val_mse:
                        best_val_mse = avg_val
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        no_improve = 0
                    else:
                        no_improve += 1

                    if epoch % 10 == 0 or epoch == 1:
                        print(f"  Epoch {epoch:3d} | Train: {avg_train:.6f} | "
                              f"Val: {avg_val:.6f} | Best: {best_val_mse:.6f}")

                    if no_improve >= 15:
                        print(f"  Early stopping at epoch {epoch}")
                        break

                train_time = time.time() - t0

                if best_state:
                    model.load_state_dict(best_state)
                    model.to(device)

                # Evaluate
                model.eval()
                all_scores, all_labels = [], []
                with torch.no_grad():
                    for signals, labels in loaders["test"]:
                        signals = signals.to(device)
                        out = model(signals)
                        mse = torch.mean((out.x_hat - signals) ** 2, dim=(1, 2))
                        all_scores.append(mse.cpu().numpy())
                        all_labels.append(labels.numpy())

                scores = np.concatenate(all_scores)
                labels = np.concatenate(all_labels)

                # Threshold
                norm_scores = []
                with torch.no_grad():
                    for signals, lbl in loaders["val_normal"]:
                        signals = signals.to(device)
                        out = model(signals)
                        mse = torch.mean((out.x_hat - signals) ** 2, dim=(1, 2))
                        norm_scores.append(mse.cpu().numpy())
                norm_scores = np.concatenate(norm_scores)
                threshold = float(np.percentile(norm_scores, 95))

                normal_s = scores[labels == 0]
                abnormal_s = scores[labels == 1]
                sep = (abnormal_s.mean() - normal_s.mean()) / max(normal_s.std(), 1e-8)

                from sklearn.metrics import roc_auc_score, average_precision_score
                auroc = roc_auc_score(labels, scores)
                auprc = average_precision_score(labels, scores)
                preds = (scores > threshold).astype(int)
                tp = ((preds == 1) & (labels == 1)).sum()
                fp = ((preds == 1) & (labels == 0)).sum()
                fn = ((preds == 0) & (labels == 1)).sum()
                prec = tp / max(tp + fp, 1)
                sens = tp / max(tp + fn, 1)
                f1 = 2 * prec * sens / max(prec + sens, 1e-8)

                print(f"  AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | "
                      f"F1: {f1:.4f} | Sep: {sep:.3f}")

                seed_aurocs.append(auroc)
                entry = {
                    "depth": depth, "bottleneck": bn, "seed": seed,
                    "auroc": auroc, "auprc": auprc, "f1": float(f1),
                    "separation": sep,
                    "params": model.count_parameters(),
                    "size_mb": model.model_size_mb(),
                    "train_time_s": train_time,
                }
                results.append(entry)
                csv_lines.append(
                    f"{depth},{bn},{seed},{auroc:.6f},{auprc:.6f},{f1:.6f},"
                    f"{sep:.4f},{model.count_parameters()},{model.model_size_mb():.3f},{train_time:.1f}"
                )

            if len(seed_aurocs) > 1:
                print(f"\n  >>> {depth} bn={bn}: AUROC = {np.mean(seed_aurocs):.4f} ± {np.std(seed_aurocs):.4f}")

    # Save
    import os
    os.makedirs("outputs", exist_ok=True)
    csv_path = "outputs/resconvae_ablation_results.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines) + "\n")

    # Summary
    print(f"\n{'='*70}")
    print(f"RESCONVAE ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Depth':>8s} | {'BN':>4s} | {'AUROC':>14s} | {'Params':>10s} | {'Size':>7s}")
    print(f"{'─'*8}-+-{'─'*4}-+-{'─'*14}-+-{'─'*10}-+-{'─'*7}")

    from itertools import groupby
    for (depth, bn), grp in groupby(results, key=lambda r: (r["depth"], r["bottleneck"])):
        grp = list(grp)
        aurocs = [r["auroc"] for r in grp]
        if len(grp) > 1:
            auroc_str = f"{np.mean(aurocs):.4f}±{np.std(aurocs):.4f}"
        else:
            auroc_str = f"{aurocs[0]:.4f}"
        print(f"{depth:>8s} | {bn:4d} | {auroc_str:>14s} | {grp[0]['params']:>10,} | {grp[0]['size_mb']:>6.2f}M")

    best = max(results, key=lambda r: r["auroc"])
    print(f"\n🏆 Best: depth={best['depth']} bn={best['bottleneck']} "
          f"→ AUROC={best['auroc']:.4f}")
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    run_ablation()
