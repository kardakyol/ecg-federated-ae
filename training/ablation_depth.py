import argparse
import os
import time
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.dataset import load_splits, create_synthetic_data, create_dataloaders
from utils.reproducibility import SEEDS, set_seed, get_device, setup_logging
from utils.csv_logger import ResultLogger
from evaluation.metrics import compute_metrics, aggregate_seeds, format_aggregated

from models.base import BaseAutoencoder, AEOutput


logger = logging.getLogger(__name__)


# Vanilla AE depth variants (original: 12000 -> 512 -> 256 -> 64 -> bottleneck)
class VanillaAE_Shallow(BaseAutoencoder):
    """Shallow variant: 1 hidden layer per side (12000 -> 256 -> bottleneck)."""

    def __init__(self, bottleneck: int = 32, n_leads: int = 12, seq_len: int = 1000):
        super().__init__()
        self.n_leads = n_leads
        self.seq_len = seq_len
        self.input_dim = n_leads * seq_len

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, self.input_dim),
        )

    def forward(self, x: torch.Tensor) -> AEOutput:
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        z = self.encoder(x_flat)
        x_hat = self.decoder(z).view(batch_size, self.n_leads, self.seq_len)
        return AEOutput(x_hat=x_hat)

    def compute_loss(self, x, output, **kwargs):
        return (F.mse_loss(output.x_hat, x),)


class VanillaAE_Deep(BaseAutoencoder):
    """Deep variant: 5 hidden layers per side
    (12000 -> 2048 -> 1024 -> 512 -> 256 -> 64 -> bottleneck)."""

    def __init__(self, bottleneck: int = 32, n_leads: int = 12, seq_len: int = 1000):
        super().__init__()
        self.n_leads = n_leads
        self.seq_len = seq_len
        self.input_dim = n_leads * seq_len

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 2048),
            nn.ReLU(inplace=False),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 64),
            nn.ReLU(inplace=False),
            nn.Linear(64, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 64),
            nn.ReLU(inplace=False),
            nn.Linear(64, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=False),
            nn.Linear(2048, self.input_dim),
        )

    def forward(self, x: torch.Tensor) -> AEOutput:
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        z = self.encoder(x_flat)
        x_hat = self.decoder(z).view(batch_size, self.n_leads, self.seq_len)
        return AEOutput(x_hat=x_hat)

    def compute_loss(self, x, output, **kwargs):
        return (F.mse_loss(output.x_hat, x),)


# ConvAE depth variants (original: 4 conv blocks)
class ConvAE_Shallow(BaseAutoencoder):
    """Shallow ConvAE: 2 conv blocks instead of 4.
    (B, 12, 1000) -> (B, 32, 500) -> (B, 64, 250) -> FC -> bottleneck."""

    def __init__(self, bottleneck: int = 32, n_leads: int = 12, seq_len: int = 1000):
        super().__init__()
        self.n_leads = n_leads
        self.seq_len = seq_len

        # Encoder: 2 blocks
        self.enc_conv1 = nn.Conv1d(n_leads, 32, kernel_size=7, stride=2, padding=3)
        self.enc_gn1 = nn.GroupNorm(num_groups=min(32, 32), num_channels=32)
        self.enc_act1 = nn.ReLU(inplace=False)

        self.enc_conv2 = nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3)
        self.enc_gn2 = nn.GroupNorm(num_groups=min(32, 64), num_channels=64)
        self.enc_act2 = nn.ReLU(inplace=False)

        # Compute temporal dim after 2 conv blocks
        curr_len = seq_len
        for k, s, p in [(7, 2, 3), (7, 2, 3)]:
            curr_len = math.floor((curr_len + 2*p - k) / s) + 1
        self._enc_temporal = curr_len
        self._enc_flat_dim = 64 * self._enc_temporal

        self.enc_fc = nn.Linear(self._enc_flat_dim, bottleneck)
        self.enc_fc_act = nn.ReLU(inplace=False)

        # Decoder
        self.dec_fc = nn.Linear(bottleneck, self._enc_flat_dim)
        self.dec_fc_act = nn.ReLU(inplace=False)

        self.dec_conv1 = nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1)
        self.dec_gn1 = nn.GroupNorm(num_groups=min(32, 32), num_channels=32)
        self.dec_act1 = nn.ReLU(inplace=False)

        self.dec_conv2 = nn.ConvTranspose1d(32, n_leads, kernel_size=7, stride=2, padding=3, output_padding=1)

    def forward(self, x: torch.Tensor) -> AEOutput:
        h = self.enc_act1(self.enc_gn1(self.enc_conv1(x)))
        h = self.enc_act2(self.enc_gn2(self.enc_conv2(h)))
        h_flat = h.view(h.shape[0], -1)
        z = self.enc_fc_act(self.enc_fc(h_flat))

        h = self.dec_fc_act(self.dec_fc(z))
        h = h.view(-1, 64, self._enc_temporal)
        h = self.dec_act1(self.dec_gn1(self.dec_conv1(h)))
        x_hat = self.dec_conv2(h)

        if x_hat.shape[-1] != x.shape[-1]:
            x_hat = F.interpolate(x_hat, size=x.shape[-1], mode='linear', align_corners=False)
        return AEOutput(x_hat=x_hat)

    def compute_loss(self, x, output, **kwargs):
        return (F.mse_loss(output.x_hat, x),)


class ConvAE_Deep(BaseAutoencoder):
    """Deep ConvAE: 6 conv blocks instead of 4.
    Adds 2 extra blocks for finer-grained feature extraction."""

    def __init__(self, bottleneck: int = 32, n_leads: int = 12, seq_len: int = 1000):
        super().__init__()
        self.n_leads = n_leads
        self.seq_len = seq_len

        # Encoder: 6 blocks
        # (B, 12, 1000) -> (B, 16, 500)
        self.enc_conv1 = nn.Conv1d(n_leads, 16, kernel_size=7, stride=2, padding=3)
        self.enc_gn1 = nn.GroupNorm(num_groups=min(16, 16), num_channels=16)
        self.enc_act1 = nn.ReLU(inplace=False)

        # (B, 16, 500) -> (B, 32, 250)
        self.enc_conv2 = nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3)
        self.enc_gn2 = nn.GroupNorm(num_groups=min(32, 32), num_channels=32)
        self.enc_act2 = nn.ReLU(inplace=False)

        # (B, 32, 250) -> (B, 64, 125)
        self.enc_conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.enc_gn3 = nn.GroupNorm(num_groups=min(32, 64), num_channels=64)
        self.enc_act3 = nn.ReLU(inplace=False)

        # (B, 64, 125) -> (B, 128, 63)
        self.enc_conv4 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.enc_gn4 = nn.GroupNorm(num_groups=min(32, 128), num_channels=128)
        self.enc_act4 = nn.ReLU(inplace=False)

        # (B, 128, 63) -> (B, 256, 32)
        self.enc_conv5 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)
        self.enc_gn5 = nn.GroupNorm(num_groups=min(32, 256), num_channels=256)
        self.enc_act5 = nn.ReLU(inplace=False)

        # (B, 256, 32) -> (B, 512, 16)
        self.enc_conv6 = nn.Conv1d(256, 512, kernel_size=5, stride=2, padding=2)
        self.enc_gn6 = nn.GroupNorm(num_groups=min(32, 512), num_channels=512)
        self.enc_act6 = nn.ReLU(inplace=False)

        # Compute temporal dim
        curr_len = seq_len
        for k, s, p in [(7,2,3), (7,2,3), (5,2,2), (5,2,2), (5,2,2), (5,2,2)]:
            curr_len = math.floor((curr_len + 2*p - k) / s) + 1
        self._enc_temporal = curr_len
        self._enc_flat_dim = 512 * self._enc_temporal

        self.enc_fc = nn.Linear(self._enc_flat_dim, bottleneck)
        self.enc_fc_act = nn.ReLU(inplace=False)

        # Decoder
        self.dec_fc = nn.Linear(bottleneck, self._enc_flat_dim)
        self.dec_fc_act = nn.ReLU(inplace=False)

        self.dec_conv1 = nn.ConvTranspose1d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_gn1 = nn.GroupNorm(num_groups=min(32, 256), num_channels=256)
        self.dec_act1 = nn.ReLU(inplace=False)

        self.dec_conv2 = nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_gn2 = nn.GroupNorm(num_groups=min(32, 128), num_channels=128)
        self.dec_act2 = nn.ReLU(inplace=False)

        self.dec_conv3 = nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_gn3 = nn.GroupNorm(num_groups=min(32, 64), num_channels=64)
        self.dec_act3 = nn.ReLU(inplace=False)

        self.dec_conv4 = nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_gn4 = nn.GroupNorm(num_groups=min(32, 32), num_channels=32)
        self.dec_act4 = nn.ReLU(inplace=False)

        self.dec_conv5 = nn.ConvTranspose1d(32, 16, kernel_size=7, stride=2, padding=3, output_padding=1)
        self.dec_gn5 = nn.GroupNorm(num_groups=min(16, 16), num_channels=16)
        self.dec_act5 = nn.ReLU(inplace=False)

        self.dec_conv6 = nn.ConvTranspose1d(16, n_leads, kernel_size=7, stride=2, padding=3, output_padding=1)

    def forward(self, x: torch.Tensor) -> AEOutput:
        h = self.enc_act1(self.enc_gn1(self.enc_conv1(x)))
        h = self.enc_act2(self.enc_gn2(self.enc_conv2(h)))
        h = self.enc_act3(self.enc_gn3(self.enc_conv3(h)))
        h = self.enc_act4(self.enc_gn4(self.enc_conv4(h)))
        h = self.enc_act5(self.enc_gn5(self.enc_conv5(h)))
        h = self.enc_act6(self.enc_gn6(self.enc_conv6(h)))

        h_flat = h.view(h.shape[0], -1)
        z = self.enc_fc_act(self.enc_fc(h_flat))

        h = self.dec_fc_act(self.dec_fc(z))
        h = h.view(-1, 512, self._enc_temporal)

        h = self.dec_act1(self.dec_gn1(self.dec_conv1(h)))
        h = self.dec_act2(self.dec_gn2(self.dec_conv2(h)))
        h = self.dec_act3(self.dec_gn3(self.dec_conv3(h)))
        h = self.dec_act4(self.dec_gn4(self.dec_conv4(h)))
        h = self.dec_act5(self.dec_gn5(self.dec_conv5(h)))
        x_hat = self.dec_conv6(h)

        if x_hat.shape[-1] != x.shape[-1]:
            x_hat = F.interpolate(x_hat, size=x.shape[-1], mode='linear', align_corners=False)
        return AEOutput(x_hat=x_hat)

    def compute_loss(self, x, output, **kwargs):
        return (F.mse_loss(output.x_hat, x),)


MODEL_REGISTRY = {
    # VanillaAE 
    "vanilla_ae_shallow": VanillaAE_Shallow,    
    "vanilla_ae_default": None,         
    "vanilla_ae_deep": VanillaAE_Deep,   
    # ConvAE 
    "conv_ae_shallow": ConvAE_Shallow,       
    "conv_ae_default": None,       
    "conv_ae_deep": ConvAE_Deep,            
}

from models.vanilla_ae import VanillaAE
from models.conv_ae import ConvAE
MODEL_REGISTRY["vanilla_ae_default"] = VanillaAE
MODEL_REGISTRY["conv_ae_default"] = ConvAE

DEPTH_LABELS = {
    "vanilla_ae": {
        "shallow": ("vanilla_ae_shallow", "1 hidden layer"),
        "default": ("vanilla_ae_default", "3 hidden layers (original)"),
        "deep": ("vanilla_ae_deep", "5 hidden layers"),
    },
    "conv_ae": {
        "shallow": ("conv_ae_shallow", "2 conv blocks"),
        "default": ("conv_ae_default", "4 conv blocks (original)"),
        "deep": ("conv_ae_deep", "6 conv blocks"),
    },
}


def compute_anomaly_scores(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            output = model(signals)
            mse = ((output.x_hat - signals) ** 2).mean(dim=(1, 2))
            all_scores.append(mse.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)


def find_threshold(model, val_normal_loader, device, percentile=95):
    model.eval()
    scores = []
    with torch.no_grad():
        for signals, labels in val_normal_loader:
            signals = signals.to(device)
            output = model(signals)
            mse = ((output.x_hat - signals) ** 2).mean(dim=(1, 2))
            scores.append(mse.cpu().numpy())
    return float(np.percentile(np.concatenate(scores), percentile))


# Training Standard (matches ablation_bottleneck.py)
def train_single(model_key, bottleneck, loaders, seed, device,
                 epochs=200, lr=1e-3, weight_decay=1e-5, patience=25):
    set_seed(seed)
    ModelClass = MODEL_REGISTRY[model_key]
    model = ModelClass(bottleneck=bottleneck).to(device)

    # CosineAnnealingWarmRestarts (consistent with bottleneck ablation)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    size_mb = model.model_size_mb()
    logger.info(f"  {model_key} | bottleneck={bottleneck} | seed={seed} | "
                f"params={model.count_parameters():,} | size={size_mb:.2f} MB")

    best_val_mse = float("inf")
    no_improve = 0
    best_state = None
    train_start = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for batch_idx, (signals, labels) in enumerate(loaders["train"]):
            signals = signals.to(device)
            optimizer.zero_grad()
            output = model(signals)
            loss = model.compute_loss(signals, output)[0]
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(epoch + batch_idx / max(len(loaders["train"]), 1))
            epoch_losses.append(loss.item())

        # Early stopping on val MSE (not val AUROC)
        model.eval()
        val_mse_sum, n_val = 0.0, 0
        with torch.no_grad():
            for signals, labels in loaders["val"]:
                signals = signals.to(device)
                output = model(signals)
                val_mse_sum += F.mse_loss(output.x_hat, signals).item()
                n_val += 1
        avg_val_mse = val_mse_sum / max(n_val, 1)

        if avg_val_mse < best_val_mse:
            best_val_mse = avg_val_mse
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if (epoch + 1) % 25 == 0:
            logger.info(f"    Epoch {epoch+1}/{epochs} | Loss: {np.mean(epoch_losses):.6f} | "
                        f"Val MSE: {avg_val_mse:.6f} | Best: {best_val_mse:.6f}")

        if no_improve >= patience:
            logger.info(f"    Early stop at epoch {epoch+1}")
            break

    train_time = time.time() - train_start
    if best_state:
        model.load_state_dict(best_state)

    test_scores, test_labels = compute_anomaly_scores(model, loaders["test"], device)
    test_threshold = find_threshold(model, loaders["val_normal"], device)
    test_result = compute_metrics(test_labels, test_scores, test_threshold)

    logger.info(f"    -> Test {test_result} | time={train_time:.1f}s")
    return test_result, size_mb, train_time



def main():
    parser = argparse.ArgumentParser(description="Layer Depth Ablation")
    parser.add_argument("--data_dir", type=str, default="data/ptb-xl-zscore")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--model", type=str, default=None,
                        choices=["vanilla_ae", "conv_ae"])
    parser.add_argument("--depths", type=str, nargs="+", default=None,
                        choices=["shallow", "default", "deep"])
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--bottleneck", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    setup_logging()
    device = get_device()
    logger.info(f"Device: {device}")

    if args.synthetic:
        logger.info("Using SYNTHETIC data")
        splits = create_synthetic_data(n_train=2000, n_val=500, n_test=500)
    else:
        logger.info(f"Loading data from {args.data_dir}")
        splits = load_splits(args.data_dir)

    loaders = create_dataloaders(splits, batch_size=args.batch_size)

    models_to_run = [args.model] if args.model else ["vanilla_ae", "conv_ae"]
    depths = args.depths or ["shallow", "default", "deep"]
    seeds = args.seeds or SEEDS

    os.makedirs("outputs", exist_ok=True)
    csv_logger = ResultLogger("outputs/ablation_depth.csv",
                              extra_columns=["bottleneck", "depth", "depth_desc"])

    total_runs = len(models_to_run) * len(depths) * len(seeds)
    logger.info(f"Models: {models_to_run} | Depths: {depths} | Seeds: {seeds}")
    logger.info(f"Total runs: {total_runs}")

    for base_model in models_to_run:
        for depth in depths:
            model_key, depth_desc = DEPTH_LABELS[base_model][depth]

            logger.info(f"\n{'='*60}")
            logger.info(f"{base_model} | depth={depth} ({depth_desc})")
            logger.info(f"{'='*60}")

            results_per_seed = []
            for seed in seeds:
                result, size_mb, train_time = train_single(
                    model_key, args.bottleneck, loaders, seed, device,
                    epochs=args.epochs, lr=args.lr, patience=args.patience,
                )
                results_per_seed.append(result)

                csv_logger.log(
                    model=base_model,
                    setting="centralised",
                    bottleneck=args.bottleneck,
                    depth=depth,
                    depth_desc=depth_desc,
                    beta="",
                    epsilon="",
                    precision_type="fp32",
                    seed=seed,
                    auroc=result.auroc,
                    auprc=result.auprc,
                    sensitivity=result.sensitivity,
                    specificity=result.specificity,
                    precision_score=result.precision,
                    f1=result.f1,
                    model_size_mb=size_mb,
                    flops_m="",
                    inference_latency_ms="",
                    peak_memory_mb="",
                    training_time_s=train_time,
                )

            agg = aggregate_seeds(results_per_seed)
            logger.info(f"\n  Aggregated ({len(seeds)} seeds):")
            logger.info(format_aggregated(agg))

    logger.info(f"\nResults saved to outputs/ablation_depth.csv")


if __name__ == "__main__":
    main()