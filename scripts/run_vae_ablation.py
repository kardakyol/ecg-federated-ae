"""
run_vae_ablation.py — Person C (Kaan), Sprint 3
================================================
VAE bottleneck ablation: latent_dim in {8, 16, 32, 64, 128}, 3 seeds each.

Sprint 3 updates:
  - CosineAnnealingWarmRestarts (from max_auroc_pipeline)
  - Gradient clipping max_norm=1.0
  - weight_decay=1e-5
  - patience=25
  - epochs=200 (was 100)
  - Added bn=8 to range
  - Default data_dir=data/ptb-xl-zscore

Usage:
    python scripts/run_vae_ablation.py --data_dir data/ptb-xl-zscore
    python scripts/run_vae_ablation.py --data_dir data/ptb-xl-zscore --bottlenecks 8 16 32
    python scripts/run_vae_ablation.py --synthetic --quick
"""

from __future__ import annotations
import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

SEEDS = [42, 123, 456]

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# VAE (inline — no import chain needed)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VAEConfig:
    in_channels: int = 12
    seq_len: int = 1000
    encoder_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    kernel_size: int = 7
    stride: int = 2
    dropout: float = 0.1
    latent_dim: int = 32


class _EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, dropout):
        super().__init__()
        p = (kernel - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=p, bias=False)
        self.norm = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act  = nn.LeakyReLU(0.2, inplace=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.norm(self.conv(x))))


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, is_last=False):
        super().__init__()
        p = (kernel - 1) // 2
        self.deconv  = nn.ConvTranspose1d(in_ch, out_ch, kernel, stride=stride,
                                           padding=p, output_padding=stride - 1, bias=False)
        self.is_last = is_last
        if not is_last:
            self.norm = nn.GroupNorm(min(32, out_ch), out_ch)
            self.act  = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        x = self.deconv(x)
        return x if self.is_last else self.act(self.norm(x))


class VAEModel(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        chs = [cfg.in_channels] + cfg.encoder_channels
        self.encoder = nn.ModuleList([
            _EncoderBlock(chs[i], chs[i+1], cfg.kernel_size, cfg.stride, cfg.dropout)
            for i in range(len(cfg.encoder_channels))
        ])
        t = cfg.seq_len
        for _ in cfg.encoder_channels:
            t = math.floor((t + 2 * ((cfg.kernel_size - 1) // 2) - cfg.kernel_size) / cfg.stride) + 1
        self._enc_t  = t
        self._flat   = cfg.encoder_channels[-1] * t
        self.fc_mu     = nn.Linear(self._flat, cfg.latent_dim)
        self.fc_logvar = nn.Linear(self._flat, cfg.latent_dim)
        self.fc_dec    = nn.Linear(cfg.latent_dim, self._flat)
        dec_chs = list(reversed(cfg.encoder_channels)) + [cfg.in_channels]
        self.decoder = nn.ModuleList([
            _DecoderBlock(dec_chs[i], dec_chs[i+1], cfg.kernel_size, cfg.stride,
                          is_last=(i == len(dec_chs) - 2))
            for i in range(len(dec_chs) - 1)
        ])

    def forward(self, x):
        h = x
        for blk in self.encoder:
            h = blk(h)
        h = h.flatten(1)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        lv     = torch.clamp(logvar, -20.0, 2.0)
        z      = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        h      = self.fc_dec(z).view(-1, self.cfg.encoder_channels[-1], self._enc_t)
        for blk in self.decoder:
            h = blk(h)
        if h.shape[-1] != self.cfg.seq_len:
            h = F.interpolate(h, size=self.cfg.seq_len, mode='linear', align_corners=False)
        return h, mu, logvar

    def loss(self, x, x_hat, mu, logvar, beta, kl_weight):
        mse = F.mse_loss(x_hat, x)
        lv  = torch.clamp(logvar, -20.0, 2.0)
        kl  = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / x.shape[0]
        return mse + beta * kl_weight * kl, mse

    def size_mb(self):
        return sum(p.numel() * p.element_size() for p in self.parameters()) / 1e6


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_dir: str, quick: bool = False) -> dict:
    d = Path(data_dir)
    train_x = torch.from_numpy(np.load(d / "train_signals.npy")).float()
    val_x   = torch.from_numpy(np.load(d / "val_signals.npy")).float()
    val_y   = torch.from_numpy(np.load(d / "val_labels.npy")).long()
    test_x  = torch.from_numpy(np.load(d / "test_signals.npy")).float()
    test_y  = torch.from_numpy(np.load(d / "test_labels.npy")).long()
    if quick:
        train_x, val_x, val_y = train_x[:500], val_x[:200], val_y[:200]
        test_x, test_y = test_x[:200], test_y[:200]
    val_normal_x = val_x[val_y == 0]
    return {
        "train":      TensorDataset(train_x),
        "val_normal": TensorDataset(val_normal_x),
        "test":       TensorDataset(test_x, test_y),
    }


def make_synthetic(quick: bool = False) -> dict:
    n = 300 if quick else 1000
    m = 100 if quick else 300
    tx = torch.randn(n, 12, 1000) * 0.1
    vn = torch.randn(m, 12, 1000) * 0.1
    te = torch.randn(m * 2, 12, 1000) * 0.1
    te[m:] += 2.0
    ty = torch.cat([torch.zeros(m), torch.ones(m)]).long()
    return {"train": TensorDataset(tx), "val_normal": TensorDataset(vn),
            "test":  TensorDataset(te, ty)}


def make_loaders(splits, bs=64) -> dict:
    return {k: DataLoader(v, batch_size=bs, shuffle=(k == "train"), drop_last=False)
            for k, v in splits.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Train / Evaluate (Sprint 3: cosine annealing + grad clip)
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, loaders, device, epochs, beta=0.5, patience=25):
    """Sprint 3 optimised training loop matching max_auroc_pipeline."""
    opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2, eta_min=1e-6)
    best, no_imp, best_state = float("inf"), 0, None

    for epoch in range(1, epochs + 1):
        kl_w = min(1.0, epoch / 20.0)
        model.train()
        for batch_idx, (x,) in enumerate(loaders["train"]):
            x = x.to(device)
            opt.zero_grad()
            x_hat, mu, lv = model(x)
            loss, _ = model.loss(x, x_hat, mu, lv, beta, kl_w)
            loss.backward()
            # Sprint 3: gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step(epoch + batch_idx / max(len(loaders["train"]), 1))

        model.eval()
        vl = []
        with torch.no_grad():
            for (x,) in loaders["val_normal"]:
                x = x.to(device)
                x_hat, mu, lv = model(x)
                _, mse = model.loss(x, x_hat, mu, lv, beta, 1.0)
                vl.append(mse.item())
        val_mse = float(np.mean(vl))

        if val_mse < best - 1e-6:
            best, no_imp = val_mse, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)


def evaluate_model(model, loaders, device) -> dict:
    model.eval()
    ns = []
    with torch.no_grad():
        for (x,) in loaders["val_normal"]:
            x = x.to(device)
            x_hat, _, _ = model(x)
            ns.extend(torch.mean((x_hat - x) ** 2, dim=(1, 2)).cpu().numpy())
    thr = float(np.percentile(ns, 95))

    sc, lb = [], []
    with torch.no_grad():
        for (x, y) in loaders["test"]:
            x = x.to(device)
            x_hat, _, _ = model(x)
            sc.extend(torch.mean((x_hat - x) ** 2, dim=(1, 2)).cpu().numpy())
            lb.extend(y.numpy())

    scores, labels = np.array(sc), np.array(lb)
    preds = (scores >= thr).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    try:
        auroc = float(roc_auc_score(labels, scores))
        auprc = float(average_precision_score(labels, scores))
    except Exception:
        auroc, auprc = 0.5, float(np.mean(labels))
    return dict(auroc=auroc, auprc=auprc, sensitivity=sens,
                specificity=spec, precision=prec, f1=f1)


# ─────────────────────────────────────────────────────────────────────────────
# CSV logging
# ─────────────────────────────────────────────────────────────────────────────

COLS = [
    "model", "setting", "beta", "epsilon", "precision_type", "seed",
    "auroc", "auprc", "sensitivity", "specificity", "precision_score", "f1",
    "model_size_mb", "flops_m", "inference_latency_ms", "peak_memory_mb",
    "training_time_s",
]

def append_csv(path: str, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    need_header = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if need_header:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in COLS})


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VAE Bottleneck Ablation — Sprint 3")
    parser.add_argument("--data_dir",    default="data/ptb-xl-zscore")
    parser.add_argument("--synthetic",   action="store_true")
    parser.add_argument("--quick",       action="store_true",
                        help="Fast test: 1 seed, 10 epochs, 2 configs")
    parser.add_argument("--bottlenecks", type=int, nargs="+",
                        help="Bottleneck sizes (default: 8 16 32 64 128)")
    parser.add_argument("--epochs",      type=int, default=None)
    parser.add_argument("--device",      type=str, default=None)
    args = parser.parse_args()

    device      = torch.device(args.device) if args.device else get_device()
    epochs      = args.epochs or (10 if args.quick else 200)
    seeds       = [SEEDS[0]] if args.quick else SEEDS
    bottlenecks = args.bottlenecks or [8, 16, 32, 64, 128]
    if args.quick:
        bottlenecks = bottlenecks[:2]
    out_csv = "outputs/vae_baselines.csv"

    print(f"Device      : {device}")
    print(f"Bottlenecks : {bottlenecks}")
    print(f"Seeds       : {seeds}")
    print(f"Epochs      : {epochs}")
    print(f"Output CSV  : {out_csv}")
    print("=" * 65)

    summary: dict = {}

    for bn in bottlenecks:
        aurocs = []
        print(f"\n[bottleneck = {bn}]")

        for seed in seeds:
            set_seed(seed)
            splits  = make_synthetic(args.quick) if args.synthetic else load_data(args.data_dir, args.quick)
            loaders = make_loaders(splits)
            model   = VAEModel(VAEConfig(latent_dim=bn)).to(device)

            t0 = time.time()
            train_model(model, loaders, device, epochs)
            elapsed = time.time() - t0

            m = evaluate_model(model, loaders, device)
            aurocs.append(m["auroc"])

            print(f"  seed={seed:3d} | AUROC={m['auroc']:.4f} | "
                  f"AUPRC={m['auprc']:.4f} | F1={m['f1']:.4f} | "
                  f"size={model.size_mb():.2f}MB | {elapsed:.0f}s")

            append_csv(out_csv, {
                "model":           "VAE",
                "setting":         f"centralised_bottleneck{bn}",
                "beta":            0.5,
                "precision_type":  "fp32",
                "seed":            seed,
                "auroc":           m["auroc"],
                "auprc":           m["auprc"],
                "sensitivity":     m["sensitivity"],
                "specificity":     m["specificity"],
                "precision_score": m["precision"],
                "f1":              m["f1"],
                "model_size_mb":   model.size_mb(),
                "training_time_s": elapsed,
            })

        mean_a = float(np.mean(aurocs))
        std_a  = float(np.std(aurocs))
        summary[bn] = (mean_a, std_a)
        print(f"  → AUROC: {mean_a:.4f} ± {std_a:.4f}")

    print("\n" + "=" * 65)
    print("SUMMARY — VAE Bottleneck Ablation")
    print(f"  {'Bottleneck':>12}  {'AUROC':>20}")
    print("  " + "-" * 36)
    for bn, (mean_a, std_a) in summary.items():
        print(f"  {bn:>12}  {mean_a:.4f} ± {std_a:.4f}")
    best_bn = max(summary, key=lambda k: summary[k][0])
    print(f"\n  Best: bottleneck={best_bn}  AUROC={summary[best_bn][0]:.4f}")
    print("=" * 65)
    print(f"\nResults appended to {out_csv}")


if __name__ == "__main__":
    main()