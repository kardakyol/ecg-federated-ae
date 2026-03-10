"""
run_perclass_breakdown.py — Person C (Kaan), Sprint 3
======================================================
Per-class anomaly detection breakdown for PTB-XL subcategories:
  MI (Myocardial Infarction), STTC (ST/T-wave Change),
  HYP (Hypertrophy), CD (Conduction Disturbance)

Loads saved checkpoints (or trains from scratch if not available),
then evaluates AUROC/AUPRC per superclass on the test set.

Results saved to:
  outputs/perclass_breakdown.csv

Prerequisites:
  1. Run extract_subclass_labels.py first to create test_subclass_labels.npy
  2. Have trained checkpoints in checkpoints/ OR use --train_first flag

Usage:
    python scripts/run_perclass_breakdown.py --data_dir data/ptb-xl
    python scripts/run_perclass_breakdown.py --data_dir data/ptb-xl --train_first
    python scripts/run_perclass_breakdown.py --synthetic --quick   # pipeline test
"""

from __future__ import annotations
import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score


# ─────────────────────────────────────────────────────────────────────────────
# Superclass constants
# ─────────────────────────────────────────────────────────────────────────────

SUPERCLASS_NAMES = {0: "NORM", 1: "MI", 2: "STTC", 3: "HYP", 4: "CD"}
ANOMALY_CLASSES  = [1, 2, 3, 4]   # NORM=0 is the normal class


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

SEEDS = [42, 123, 456]

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Inline model definitions (VAE, ConvAE, VanillaAE)
# ─────────────────────────────────────────────────────────────────────────────

# ── VAE ──

@dataclass
class VAEConfig:
    in_channels: int = 12
    seq_len: int = 1000
    encoder_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    kernel_size: int = 7
    stride: int = 2
    dropout: float = 0.1
    latent_dim: int = 32

class _EBlock(nn.Module):
    def __init__(self, ic, oc, k, s, d):
        super().__init__()
        p = (k-1)//2
        self.net = nn.Sequential(
            nn.Conv1d(ic, oc, k, s, p, bias=False),
            nn.GroupNorm(min(32, oc), oc),
            nn.LeakyReLU(0.2, False), nn.Dropout(d))
    def forward(self, x): return self.net(x)

class _DBlock(nn.Module):
    def __init__(self, ic, oc, k, s, last=False):
        super().__init__()
        p = (k-1)//2
        self.dc   = nn.ConvTranspose1d(ic, oc, k, s, p, output_padding=s-1, bias=False)
        self.last = last
        if not last:
            self.post = nn.Sequential(nn.GroupNorm(min(32, oc), oc), nn.LeakyReLU(0.2, False))
    def forward(self, x):
        x = self.dc(x)
        return x if self.last else self.post(x)

class VAENet(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        chs = [cfg.in_channels] + cfg.encoder_channels
        self.encoder = nn.ModuleList([_EBlock(chs[i], chs[i+1], cfg.kernel_size, cfg.stride, cfg.dropout)
                                      for i in range(len(cfg.encoder_channels))])
        t = cfg.seq_len
        for _ in cfg.encoder_channels:
            t = math.floor((t + 2*((cfg.kernel_size-1)//2) - cfg.kernel_size) / cfg.stride) + 1
        self._t = t; self._f = cfg.encoder_channels[-1] * t
        self.fc_mu  = nn.Linear(self._f, cfg.latent_dim)
        self.fc_lv  = nn.Linear(self._f, cfg.latent_dim)
        self.fc_dec = nn.Linear(cfg.latent_dim, self._f)
        dc = list(reversed(cfg.encoder_channels)) + [cfg.in_channels]
        self.decoder = nn.ModuleList([_DBlock(dc[i], dc[i+1], cfg.kernel_size, cfg.stride,
                                              last=(i == len(dc)-2))
                                      for i in range(len(dc)-1)])
    def forward(self, x):
        h = x
        for b in self.encoder: h = b(h)
        h = h.flatten(1)
        mu = self.fc_mu(h); lv = self.fc_lv(h)
        lvc = torch.clamp(lv, -20, 2)
        z = mu + torch.exp(0.5 * lvc) * torch.randn_like(mu)
        h = self.fc_dec(z).view(-1, self.cfg.encoder_channels[-1], self._t)
        for b in self.decoder: h = b(h)
        if h.shape[-1] != self.cfg.seq_len:
            h = F.interpolate(h, self.cfg.seq_len, mode='linear', align_corners=False)
        return h, mu, lv
    def anomaly_score(self, x):
        x_hat, _, _ = self.forward(x)
        return torch.mean((x_hat - x) ** 2, dim=(1, 2))
    def size_mb(self):
        return sum(p.numel() * p.element_size() for p in self.parameters()) / 1e6

# ── ConvAE ──

class ConvAENet(nn.Module):
    def __init__(self, bottleneck=32, n_leads=12, seq_len=1000):
        super().__init__()
        self.seq_len = seq_len
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv1d(n_leads, 32,  7, 2, 3, bias=False), nn.GroupNorm(32, 32),  nn.ReLU(False),
            nn.Conv1d(32,      64,  7, 2, 3, bias=False), nn.GroupNorm(32, 64),  nn.ReLU(False),
            nn.Conv1d(64,      128, 5, 2, 2, bias=False), nn.GroupNorm(32, 128), nn.ReLU(False),
            nn.Conv1d(128,     256, 5, 2, 2, bias=False), nn.GroupNorm(32, 256), nn.ReLU(False),
        )
        curr = seq_len
        for k, s, p in [(7,2,3),(7,2,3),(5,2,2),(5,2,2)]:
            curr = math.floor((curr + 2*p - k) / s) + 1
        self._t = curr
        self.enc_fc = nn.Sequential(nn.Linear(256 * curr, bottleneck), nn.ReLU(False))
        self.dec_fc = nn.Sequential(nn.Linear(bottleneck, 256 * curr), nn.ReLU(False))
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 5, 2, 2, output_padding=0, bias=False), nn.GroupNorm(32,128), nn.ReLU(False),
            nn.ConvTranspose1d(128, 64,  5, 2, 2, output_padding=1, bias=False), nn.GroupNorm(32,64),  nn.ReLU(False),
            nn.ConvTranspose1d(64,  32,  7, 2, 3, output_padding=1, bias=False), nn.GroupNorm(32,32),  nn.ReLU(False),
            nn.ConvTranspose1d(32,  n_leads, 7, 2, 3, output_padding=1, bias=False),
        )
    def forward(self, x):
        h = self.enc(x).flatten(1)
        z = self.enc_fc(h)
        h = self.dec_fc(z).view(-1, 256, self._t)
        x_hat = self.dec(h)
        if x_hat.shape[-1] != self.seq_len:
            x_hat = F.interpolate(x_hat, self.seq_len, mode='linear', align_corners=False)
        return x_hat
    def anomaly_score(self, x):
        return torch.mean((self.forward(x) - x) ** 2, dim=(1, 2))
    def size_mb(self):
        return sum(p.numel() * p.element_size() for p in self.parameters()) / 1e6

# ── VanillaAE ──

class VanillaAENet(nn.Module):
    def __init__(self, bottleneck=32, n_leads=12, seq_len=1000):
        super().__init__()
        d = n_leads * seq_len
        self.encoder = nn.Sequential(
            nn.Linear(d,   512), nn.ReLU(False),
            nn.Linear(512, 256), nn.ReLU(False),
            nn.Linear(256, 64),  nn.ReLU(False),
            nn.Linear(64,  bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 64),  nn.ReLU(False),
            nn.Linear(64,  256), nn.ReLU(False),
            nn.Linear(256, 512), nn.ReLU(False),
            nn.Linear(512, d),
        )
        self.n_leads = n_leads; self.seq_len = seq_len
    def forward(self, x):
        b = x.shape[0]
        return self.decoder(self.encoder(x.view(b, -1))).view(b, self.n_leads, self.seq_len)
    def anomaly_score(self, x):
        return torch.mean((self.forward(x) - x) ** 2, dim=(1, 2))
    def size_mb(self):
        return sum(p.numel() * p.element_size() for p in self.parameters()) / 1e6


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_dir: str, quick: bool = False) -> dict:
    d = Path(data_dir)
    train_x = torch.from_numpy(np.load(d / "train_signals.npy")).float()
    val_x   = torch.from_numpy(np.load(d / "val_signals.npy")).float()
    val_y   = torch.from_numpy(np.load(d / "val_labels.npy")).long()
    test_x  = torch.from_numpy(np.load(d / "test_signals.npy")).float()
    test_y  = torch.from_numpy(np.load(d / "test_labels.npy")).long()

    # Subclass labels
    sub_path = d / "test_subclass_labels.npy"
    if sub_path.exists():
        test_sub = torch.from_numpy(np.load(sub_path)).long()
    else:
        print(f"  WARNING: test_subclass_labels.npy not found.")
        print(f"  Run: python scripts/extract_subclass_labels.py --data_dir {data_dir}")
        print(f"  Per-class breakdown will be SKIPPED.")
        test_sub = None

    if quick:
        train_x = train_x[:500]
        val_x, val_y = val_x[:200], val_y[:200]
        test_x, test_y = test_x[:200], test_y[:200]
        if test_sub is not None:
            test_sub = test_sub[:200]

    val_normal_x = val_x[val_y == 0]
    return {
        "train_x":      train_x,
        "val_normal_x": val_normal_x,
        "test_x":       test_x,
        "test_y":       test_y,
        "test_sub":     test_sub,
    }


def make_synthetic(quick: bool = False) -> dict:
    n = 300 if quick else 1000
    m = 150 if quick else 500
    train_x      = torch.randn(n, 12, 1000) * 0.1
    val_normal_x = torch.randn(m // 2, 12, 1000) * 0.1
    # 4 anomaly classes + normal for test
    test_parts_x, test_parts_y, test_parts_s = [], [], []
    # normal
    test_parts_x.append(torch.randn(m // 5, 12, 1000) * 0.1)
    test_parts_y.append(torch.zeros(m // 5, dtype=torch.long))
    test_parts_s.append(torch.zeros(m // 5, dtype=torch.long))   # NORM=0
    for cls in range(1, 5):
        chunk = torch.randn(m // 5, 12, 1000) * 0.1
        chunk += cls * 0.5   # each class has different offset
        test_parts_x.append(chunk)
        test_parts_y.append(torch.ones(m // 5, dtype=torch.long))
        test_parts_s.append(torch.full((m // 5,), cls, dtype=torch.long))
    return {
        "train_x":      train_x,
        "val_normal_x": val_normal_x,
        "test_x":       torch.cat(test_parts_x),
        "test_y":       torch.cat(test_parts_y),
        "test_sub":     torch.cat(test_parts_s),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_generic(model, train_x, val_x, device, epochs, beta=0.5, patience=15):
    """Unified training loop for all 3 model types."""
    train_loader = DataLoader(TensorDataset(train_x), batch_size=32, shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_x),   batch_size=32)
    opt   = Adam(model.parameters(), lr=1e-3)
    sched = ReduceLROnPlateau(opt, patience=5, factor=0.5, min_lr=1e-5)
    best, no_imp, best_state = float("inf"), 0, None

    for epoch in range(1, epochs + 1):
        kl_w = min(1.0, epoch / 20.0)
        model.train()
        for (x,) in train_loader:
            x = x.to(device)
            opt.zero_grad()
            if isinstance(model, VAENet):
                x_hat, mu, lv = model(x)
                lvc = torch.clamp(lv, -20, 2)
                kl  = -0.5 * torch.sum(1 + lvc - mu.pow(2) - lvc.exp()) / x.shape[0]
                loss = F.mse_loss(x_hat, x) + beta * kl_w * kl
            else:
                x_hat = model(x)
                loss  = F.mse_loss(x_hat, x)
            loss.backward()
            opt.step()

        model.eval()
        vl = []
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                if isinstance(model, VAENet):
                    x_hat, _, _ = model(x)
                else:
                    x_hat = model(x)
                vl.append(F.mse_loss(x_hat, x).item())
        vm = float(np.mean(vl))
        sched.step(vm)
        if vm < best - 1e-6:
            best, no_imp = vm, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience: break

    if best_state: model.load_state_dict(best_state)


def get_scores(model, test_x, device, bs=64) -> np.ndarray:
    model.eval()
    all_scores = []
    loader = DataLoader(TensorDataset(test_x), batch_size=bs)
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            all_scores.extend(model.anomaly_score(x).cpu().numpy())
    return np.array(all_scores)


def get_threshold(model, val_x, device, bs=64, pct=95) -> float:
    model.eval()
    scores = []
    loader = DataLoader(TensorDataset(val_x), batch_size=bs)
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            scores.extend(model.anomaly_score(x).cpu().numpy())
    return float(np.percentile(scores, pct))


# ─────────────────────────────────────────────────────────────────────────────
# Per-class evaluation
# ─────────────────────────────────────────────────────────────────────────────

def per_class_auroc(scores, binary_labels, subclass_labels) -> dict:
    """
    Compute AUROC and AUPRC for each anomaly subclass vs NORM.
    Returns dict: class_name -> {auroc, auprc, n_pos, n_neg}
    """
    results = {}
    norm_mask = subclass_labels == 0

    for cls_int in ANOMALY_CLASSES:
        cls_name = SUPERCLASS_NAMES[cls_int]
        cls_mask = subclass_labels == cls_int

        # NORM samples + this class samples only
        mask    = norm_mask | cls_mask
        sc_sub  = scores[mask]
        lb_sub  = (subclass_labels[mask] == cls_int).astype(int)

        n_pos = int(lb_sub.sum())
        n_neg = int((lb_sub == 0).sum())

        if n_pos == 0 or n_neg == 0:
            results[cls_name] = {"auroc": float("nan"), "auprc": float("nan"),
                                 "n_pos": n_pos, "n_neg": n_neg}
            continue

        try:
            auroc = float(roc_auc_score(lb_sub, sc_sub))
            auprc = float(average_precision_score(lb_sub, sc_sub))
        except Exception:
            auroc, auprc = float("nan"), float("nan")

        results[cls_name] = {"auroc": auroc, "auprc": auprc,
                             "n_pos": n_pos, "n_neg": n_neg}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

COLS = ["model", "seed", "overall_auroc", "overall_auprc",
        "MI_auroc", "MI_auprc", "STTC_auroc", "STTC_auprc",
        "HYP_auroc", "HYP_auprc", "CD_auroc", "CD_auprc",
        "MI_n", "STTC_n", "HYP_n", "CD_n"]

def append_csv(path, row):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    need_hdr = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if need_hdr: w.writeheader()
        w.writerow({c: row.get(c, "") for c in COLS})


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "VAE":       lambda: VAENet(VAEConfig(latent_dim=32)),
    "ConvAE":    lambda: ConvAENet(bottleneck=32),
    "VanillaAE": lambda: VanillaAENet(bottleneck=32),
}

CHECKPOINT_MAP = {
    "VAE":       "checkpoints/vae_beta0.5_best.pt",
    "ConvAE":    "checkpoints/conv_ae_seed42.pt",
    "VanillaAE": "checkpoints/vanilla_ae_seed42.pt",
}


def run(args):
    device = torch.device(args.device) if args.device else get_device()
    print(f"Device: {device}")

    epochs = args.epochs or (5 if args.quick else 100)
    seeds  = [SEEDS[0]] if args.quick else SEEDS
    out    = "outputs/perclass_breakdown.csv"

    if args.synthetic:
        print("Using SYNTHETIC data")
        data = make_synthetic(args.quick)
    else:
        data = load_data(args.data_dir, args.quick)

    test_x   = data["test_x"]
    test_y   = data["test_y"].numpy()
    test_sub = data["test_sub"]

    if test_sub is None and not args.synthetic:
        print("\nERROR: Cannot run per-class breakdown without subclass labels.")
        print("Run first: python scripts/extract_subclass_labels.py --data_dir", args.data_dir)
        sys.exit(1)

    test_sub_np = test_sub.numpy() if test_sub is not None else None

    models_to_run = args.models if args.models else list(MODEL_CONFIGS.keys())
    if args.quick:
        models_to_run = models_to_run[:2]

    print(f"Models: {models_to_run}")
    print(f"Seeds:  {seeds}")
    print("=" * 70)

    for model_name in models_to_run:
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")

        seed_results = []

        for seed in seeds:
            set_seed(seed)
            model = MODEL_CONFIGS[model_name]().to(device)

            # Try to load checkpoint, otherwise train
            ckpt_path = Path(CHECKPOINT_MAP.get(model_name, ""))
            loaded = False
            if not args.train_first and ckpt_path.exists():
                try:
                    state = torch.load(ckpt_path, map_location=device)
                    model.load_state_dict(state, strict=False)
                    print(f"  [seed={seed}] Loaded checkpoint: {ckpt_path}")
                    loaded = True
                except Exception as e:
                    print(f"  [seed={seed}] Checkpoint load failed ({e}), training from scratch.")

            if not loaded:
                print(f"  [seed={seed}] Training {model_name} (epochs={epochs})...")
                t0 = time.time()
                train_generic(model, data["train_x"], data["val_normal_x"],
                              device, epochs)
                print(f"  [seed={seed}] Done in {time.time()-t0:.0f}s")

            # Get anomaly scores
            scores = get_scores(model, test_x, device)

            # Overall metrics
            try:
                overall_auroc = float(roc_auc_score(test_y, scores))
                overall_auprc = float(average_precision_score(test_y, scores))
            except Exception:
                overall_auroc = overall_auprc = float("nan")

            print(f"  [seed={seed}] Overall AUROC={overall_auroc:.4f}  AUPRC={overall_auprc:.4f}")

            # Per-class metrics
            if test_sub_np is not None:
                pc = per_class_auroc(scores, test_y, test_sub_np)
                for cls_name, res in pc.items():
                    auroc_str = f"{res['auroc']:.4f}" if not np.isnan(res['auroc']) else "N/A"
                    print(f"    {cls_name:6s}: AUROC={auroc_str}  n={res['n_pos']}")
            else:
                pc = {}

            row = {
                "model":          model_name,
                "seed":           seed,
                "overall_auroc":  overall_auroc,
                "overall_auprc":  overall_auprc,
            }
            for cls_name in ["MI", "STTC", "HYP", "CD"]:
                res = pc.get(cls_name, {})
                row[f"{cls_name}_auroc"] = res.get("auroc", "")
                row[f"{cls_name}_auprc"] = res.get("auprc", "")
                row[f"{cls_name}_n"]     = res.get("n_pos", "")
            append_csv(out, row)
            seed_results.append(overall_auroc)

        mean_a = float(np.mean([r for r in seed_results if not np.isnan(r)]))
        std_a  = float(np.std([r for r in seed_results if not np.isnan(r)]))
        print(f"\n  {model_name} overall AUROC: {mean_a:.4f} ± {std_a:.4f}")

    print(f"\n✓ Results saved to {out}")

    # Print summary table
    print("\n" + "=" * 70)
    print("PER-CLASS BREAKDOWN COMPLETE — see outputs/perclass_breakdown.csv")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Sprint 3 — Per-class breakdown for MI, STTC, HYP, CD (Person C)"
    )
    parser.add_argument("--data_dir",    default="data/ptb-xl")
    parser.add_argument("--synthetic",   action="store_true")
    parser.add_argument("--quick",       action="store_true",
                        help="Quick test: 1 seed, 5 epochs, 2 models")
    parser.add_argument("--train_first", action="store_true",
                        help="Always retrain instead of loading checkpoints")
    parser.add_argument("--models",      nargs="+",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Which models to evaluate (default: all 3)")
    parser.add_argument("--epochs",      type=int, default=None)
    parser.add_argument("--device",      type=str, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
