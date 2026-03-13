"""
run_perclass_breakdown.py — Person C (Kaan), Sprint 3
======================================================
Per-class anomaly detection breakdown: MI, STTC, HYP, CD.

Sprint 3 updates:
  - bottleneck=128 for all models
  - CosineAnnealingWarmRestarts + grad clip
  - patience=25, epochs=200
  - Default data_dir=data/ptb-xl-zscore

Usage:
    python scripts/run_perclass_breakdown.py --data_dir data/ptb-xl-zscore
    python scripts/run_perclass_breakdown.py --data_dir data/ptb-xl-zscore --train_first
    python scripts/run_perclass_breakdown.py --synthetic --quick
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score


SUPERCLASS_NAMES = {0: "NORM", 1: "MI", 2: "STTC", 3: "HYP", 4: "CD"}
ANOMALY_CLASSES  = [1, 2, 3, 4]
SEEDS = [42, 123, 456]
DEFAULT_BN = 128

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


# ── Inline models (same architecture, bn=128 default) ──

@dataclass
class VAEConfig:
    in_channels: int = 12; seq_len: int = 1000
    encoder_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    kernel_size: int = 7; stride: int = 2; dropout: float = 0.1
    latent_dim: int = DEFAULT_BN

class _EBlock(nn.Module):
    def __init__(self, ic, oc, k, s, d):
        super().__init__()
        p = (k-1)//2
        self.net = nn.Sequential(nn.Conv1d(ic, oc, k, s, p, bias=False),
            nn.GroupNorm(min(32, oc), oc), nn.LeakyReLU(0.2, False), nn.Dropout(d))
    def forward(self, x): return self.net(x)

class _DBlock(nn.Module):
    def __init__(self, ic, oc, k, s, last=False):
        super().__init__()
        p = (k-1)//2
        self.dc = nn.ConvTranspose1d(ic, oc, k, s, p, output_padding=s-1, bias=False)
        self.last = last
        if not last: self.post = nn.Sequential(nn.GroupNorm(min(32, oc), oc), nn.LeakyReLU(0.2, False))
    def forward(self, x):
        x = self.dc(x); return x if self.last else self.post(x)

class VAENet(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        chs = [cfg.in_channels] + cfg.encoder_channels
        self.encoder = nn.ModuleList([_EBlock(chs[i], chs[i+1], cfg.kernel_size, cfg.stride, cfg.dropout) for i in range(len(cfg.encoder_channels))])
        t = cfg.seq_len
        for _ in cfg.encoder_channels: t = math.floor((t + 2*((cfg.kernel_size-1)//2) - cfg.kernel_size) / cfg.stride) + 1
        self._t = t; self._f = cfg.encoder_channels[-1] * t
        self.fc_mu = nn.Linear(self._f, cfg.latent_dim); self.fc_lv = nn.Linear(self._f, cfg.latent_dim)
        self.fc_dec = nn.Linear(cfg.latent_dim, self._f)
        dc = list(reversed(cfg.encoder_channels)) + [cfg.in_channels]
        self.decoder = nn.ModuleList([_DBlock(dc[i], dc[i+1], cfg.kernel_size, cfg.stride, last=(i==len(dc)-2)) for i in range(len(dc)-1)])
    def forward(self, x):
        h = x
        for b in self.encoder: h = b(h)
        h = h.flatten(1); mu = self.fc_mu(h); lv = self.fc_lv(h)
        lvc = torch.clamp(lv, -20, 2); z = mu + torch.exp(0.5 * lvc) * torch.randn_like(mu)
        h = self.fc_dec(z).view(-1, self.cfg.encoder_channels[-1], self._t)
        for b in self.decoder: h = b(h)
        if h.shape[-1] != self.cfg.seq_len: h = F.interpolate(h, self.cfg.seq_len, mode='linear', align_corners=False)
        return h, mu, lv
    def anomaly_score(self, x):
        x_hat, _, _ = self.forward(x); return torch.mean((x_hat - x) ** 2, dim=(1, 2))
    def size_mb(self): return sum(p.numel() * p.element_size() for p in self.parameters()) / 1e6

class ConvAENet(nn.Module):
    def __init__(self, bottleneck=DEFAULT_BN, n_leads=12, seq_len=1000):
        super().__init__()
        self.seq_len = seq_len
        self.enc = nn.Sequential(
            nn.Conv1d(n_leads, 32, 7, 2, 3, bias=False), nn.GroupNorm(32, 32), nn.ReLU(False),
            nn.Conv1d(32, 64, 7, 2, 3, bias=False), nn.GroupNorm(32, 64), nn.ReLU(False),
            nn.Conv1d(64, 128, 5, 2, 2, bias=False), nn.GroupNorm(32, 128), nn.ReLU(False),
            nn.Conv1d(128, 256, 5, 2, 2, bias=False), nn.GroupNorm(32, 256), nn.ReLU(False))
        curr = seq_len
        for k, s, p in [(7,2,3),(7,2,3),(5,2,2),(5,2,2)]: curr = math.floor((curr+2*p-k)/s)+1
        self._t = curr
        self.enc_fc = nn.Sequential(nn.Linear(256*curr, bottleneck), nn.ReLU(False))
        self.dec_fc = nn.Sequential(nn.Linear(bottleneck, 256*curr), nn.ReLU(False))
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 5, 2, 2, output_padding=1, bias=False), nn.GroupNorm(32, 128), nn.ReLU(False),
            nn.ConvTranspose1d(128, 64, 5, 2, 2, output_padding=1, bias=False), nn.GroupNorm(32, 64), nn.ReLU(False),
            nn.ConvTranspose1d(64, 32, 7, 2, 3, output_padding=1, bias=False), nn.GroupNorm(32, 32), nn.ReLU(False),
            nn.ConvTranspose1d(32, n_leads, 7, 2, 3, output_padding=1, bias=False))
    def forward(self, x):
        h = self.enc(x).flatten(1); z = self.enc_fc(h)
        h = self.dec_fc(z).view(-1, 256, self._t); x_hat = self.dec(h)
        if x_hat.shape[-1] != self.seq_len: x_hat = F.interpolate(x_hat, self.seq_len, mode='linear', align_corners=False)
        return x_hat
    def anomaly_score(self, x): return torch.mean((self.forward(x) - x) ** 2, dim=(1, 2))
    def size_mb(self): return sum(p.numel() * p.element_size() for p in self.parameters()) / 1e6

class VanillaAENet(nn.Module):
    def __init__(self, bottleneck=DEFAULT_BN, n_leads=12, seq_len=1000):
        super().__init__()
        d = n_leads * seq_len
        self.encoder = nn.Sequential(nn.Linear(d, 512), nn.ReLU(False), nn.Linear(512, 256), nn.ReLU(False),
            nn.Linear(256, 64), nn.ReLU(False), nn.Linear(64, bottleneck))
        self.decoder = nn.Sequential(nn.Linear(bottleneck, 64), nn.ReLU(False), nn.Linear(64, 256), nn.ReLU(False),
            nn.Linear(256, 512), nn.ReLU(False), nn.Linear(512, d))
        self.n_leads = n_leads; self.seq_len = seq_len
    def forward(self, x):
        b = x.shape[0]; return self.decoder(self.encoder(x.view(b, -1))).view(b, self.n_leads, self.seq_len)
    def anomaly_score(self, x): return torch.mean((self.forward(x) - x) ** 2, dim=(1, 2))
    def size_mb(self): return sum(p.numel() * p.element_size() for p in self.parameters()) / 1e6


# ── Data ──

def load_data(data_dir: str, quick: bool = False) -> dict:
    d = Path(data_dir)
    train_x = torch.from_numpy(np.load(d / "train_signals.npy")).float()
    val_x = torch.from_numpy(np.load(d / "val_signals.npy")).float()
    val_y = torch.from_numpy(np.load(d / "val_labels.npy")).long()
    test_x = torch.from_numpy(np.load(d / "test_signals.npy")).float()
    test_y = torch.from_numpy(np.load(d / "test_labels.npy")).long()
    sub_path = d / "test_subclass_labels.npy"
    test_sub = torch.from_numpy(np.load(sub_path)).long() if sub_path.exists() else None
    if test_sub is not None and len(test_sub) != len(test_y):
        test_sub = test_sub[:len(test_y)]
    if test_sub is None:
        print(f"  WARNING: test_subclass_labels.npy not found.")
        print(f"  Run: python scripts/extract_subclass_labels.py --data_dir {data_dir}")
    if quick:
        train_x = train_x[:500]; val_x, val_y = val_x[:200], val_y[:200]
        test_x, test_y = test_x[:200], test_y[:200]
        if test_sub is not None: test_sub = test_sub[:200]
    return {"train_x": train_x, "val_normal_x": val_x[val_y == 0],
            "test_x": test_x, "test_y": test_y, "test_sub": test_sub}

def make_synthetic(quick=False):
    n = 300 if quick else 1000; m = 150 if quick else 500
    train_x = torch.randn(n, 12, 1000) * 0.1; val_normal_x = torch.randn(m//2, 12, 1000) * 0.1
    px, py, ps = [], [], []
    px.append(torch.randn(m//5, 12, 1000)*0.1); py.append(torch.zeros(m//5, dtype=torch.long)); ps.append(torch.zeros(m//5, dtype=torch.long))
    for c in range(1, 5):
        ch = torch.randn(m//5, 12, 1000)*0.1 + c*0.5
        px.append(ch); py.append(torch.ones(m//5, dtype=torch.long)); ps.append(torch.full((m//5,), c, dtype=torch.long))
    return {"train_x": train_x, "val_normal_x": val_normal_x, "test_x": torch.cat(px), "test_y": torch.cat(py), "test_sub": torch.cat(ps)}


# ── Training (Sprint 3: cosine annealing + grad clip) ──

def train_generic(model, train_x, val_x, device, epochs, beta=0.5, patience=25):
    train_loader = DataLoader(TensorDataset(train_x), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x), batch_size=64)
    opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2, eta_min=1e-6)
    best, no_imp, best_state = float("inf"), 0, None

    for epoch in range(1, epochs + 1):
        kl_w = min(1.0, epoch / 20.0)
        model.train()
        for batch_idx, (x,) in enumerate(train_loader):
            x = x.to(device); opt.zero_grad()
            if isinstance(model, VAENet):
                x_hat, mu, lv = model(x)
                lvc = torch.clamp(lv, -20, 2)
                kl = -0.5 * torch.sum(1 + lvc - mu.pow(2) - lvc.exp()) / x.shape[0]
                loss = F.mse_loss(x_hat, x) + beta * kl_w * kl
            else:
                x_hat = model(x); loss = F.mse_loss(x_hat, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step(epoch + batch_idx / max(len(train_loader), 1))

        model.eval(); vl = []
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                if isinstance(model, VAENet): x_hat, _, _ = model(x)
                else: x_hat = model(x)
                vl.append(F.mse_loss(x_hat, x).item())
        vm = float(np.mean(vl))
        if vm < best - 1e-6:
            best, no_imp = vm, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience: break

    if best_state: model.load_state_dict(best_state)


def get_scores(model, test_x, device, bs=64):
    model.eval(); all_s = []
    for (x,) in DataLoader(TensorDataset(test_x), batch_size=bs):
        with torch.no_grad(): x = x.to(device); all_s.extend(model.anomaly_score(x).cpu().numpy())
    return np.array(all_s)


# ── Per-class evaluation ──

def per_class_auroc(scores, binary_labels, subclass_labels):
    results = {}; norm_mask = subclass_labels == 0
    for cls_int in ANOMALY_CLASSES:
        cls_name = SUPERCLASS_NAMES[cls_int]; cls_mask = subclass_labels == cls_int
        mask = norm_mask | cls_mask; sc_sub = scores[mask]; lb_sub = (subclass_labels[mask] == cls_int).astype(int)
        n_pos, n_neg = int(lb_sub.sum()), int((lb_sub == 0).sum())
        if n_pos == 0 or n_neg == 0:
            results[cls_name] = {"auroc": float("nan"), "auprc": float("nan"), "n_pos": n_pos, "n_neg": n_neg}; continue
        try: auroc = float(roc_auc_score(lb_sub, sc_sub)); auprc = float(average_precision_score(lb_sub, sc_sub))
        except: auroc, auprc = float("nan"), float("nan")
        results[cls_name] = {"auroc": auroc, "auprc": auprc, "n_pos": n_pos, "n_neg": n_neg}
    return results


# ── CSV ──

COLS = ["model", "seed", "overall_auroc", "overall_auprc",
        "MI_auroc", "MI_auprc", "STTC_auroc", "STTC_auprc",
        "HYP_auroc", "HYP_auprc", "CD_auroc", "CD_auprc",
        "MI_n", "STTC_n", "HYP_n", "CD_n"]

def append_csv(path, row):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    need_hdr = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if need_hdr: w.writeheader()
        w.writerow({c: row.get(c, "") for c in COLS})


# ── Main ──

MODEL_CONFIGS = {
    "VAE":       lambda: VAENet(VAEConfig(latent_dim=DEFAULT_BN)),
    "ConvAE":    lambda: ConvAENet(bottleneck=DEFAULT_BN),
    "VanillaAE": lambda: VanillaAENet(bottleneck=DEFAULT_BN),
}

CHECKPOINT_MAP = {
    "VAE":       "checkpoints/vae_beta0.5_best.pt",
    "ConvAE":    "checkpoints/conv_ae_seed42.pt",
    "VanillaAE": "checkpoints/vanilla_ae_seed42.pt",
}


def run(args):
    device = torch.device(args.device) if args.device else get_device()
    print(f"Device: {device}")
    epochs = args.epochs or (10 if args.quick else 200)
    seeds = [SEEDS[0]] if args.quick else SEEDS
    out = "outputs/perclass_breakdown.csv"

    if args.synthetic: data = make_synthetic(args.quick)
    else: data = load_data(args.data_dir, args.quick)

    test_x, test_y = data["test_x"], data["test_y"].numpy()
    test_sub = data["test_sub"]
    if test_sub is None and not args.synthetic:
        print("\nERROR: Cannot run per-class breakdown without subclass labels.")
        print("Run first: python scripts/extract_subclass_labels.py --data_dir", args.data_dir)
        sys.exit(1)
    test_sub_np = test_sub.numpy() if test_sub is not None else None

    models_to_run = args.models if args.models else list(MODEL_CONFIGS.keys())
    if args.quick: models_to_run = models_to_run[:2]

    print(f"Models: {models_to_run}\nSeeds: {seeds}\nBottleneck: {DEFAULT_BN}")
    print("=" * 70)

    for model_name in models_to_run:
        print(f"\n{'='*70}\nModel: {model_name}\n{'='*70}")
        seed_results = []
        for seed in seeds:
            set_seed(seed); model = MODEL_CONFIGS[model_name]().to(device)
            ckpt_path = Path(CHECKPOINT_MAP.get(model_name, ""))
            loaded = False
            if not args.train_first and ckpt_path.exists():
                try:
                    state = torch.load(ckpt_path, map_location=device)
                    model.load_state_dict(state, strict=False)
                    print(f"  [seed={seed}] Loaded checkpoint: {ckpt_path}"); loaded = True
                except Exception as e:
                    print(f"  [seed={seed}] Checkpoint load failed ({e}), training.")
            if not loaded:
                print(f"  [seed={seed}] Training {model_name} (epochs={epochs})...")
                t0 = time.time()
                train_generic(model, data["train_x"], data["val_normal_x"], device, epochs)
                print(f"  [seed={seed}] Done in {time.time()-t0:.0f}s")

            scores = get_scores(model, test_x, device)
            try: overall_auroc = float(roc_auc_score(test_y, scores)); overall_auprc = float(average_precision_score(test_y, scores))
            except: overall_auroc = overall_auprc = float("nan")
            print(f"  [seed={seed}] Overall AUROC={overall_auroc:.4f}  AUPRC={overall_auprc:.4f}")

            pc = per_class_auroc(scores, test_y, test_sub_np) if test_sub_np is not None else {}
            for cn, res in pc.items():
                a = f"{res['auroc']:.4f}" if not np.isnan(res['auroc']) else "N/A"
                print(f"    {cn:6s}: AUROC={a}  n={res['n_pos']}")

            row = {"model": model_name, "seed": seed, "overall_auroc": overall_auroc, "overall_auprc": overall_auprc}
            for cn in ["MI", "STTC", "HYP", "CD"]:
                res = pc.get(cn, {}); row[f"{cn}_auroc"] = res.get("auroc", ""); row[f"{cn}_auprc"] = res.get("auprc", ""); row[f"{cn}_n"] = res.get("n_pos", "")
            append_csv(out, row); seed_results.append(overall_auroc)

        mean_a = float(np.mean([r for r in seed_results if not np.isnan(r)]))
        std_a = float(np.std([r for r in seed_results if not np.isnan(r)]))
        print(f"\n  {model_name} overall AUROC: {mean_a:.4f} ± {std_a:.4f}")

    print(f"\n✓ Results saved to {out}")

def main():
    parser = argparse.ArgumentParser(description="Sprint 3 — Per-class breakdown (Person C)")
    parser.add_argument("--data_dir", default="data/ptb-xl-zscore")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--train_first", action="store_true")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()