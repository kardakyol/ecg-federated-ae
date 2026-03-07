"""
Post-Training Quantisation (PTQ) Pipeline
==========================================
Owner: Ghadah (Person E) — Quantisation + Edge Deployment
Sprint: S2 (preparation), S3 (full experiments)

Usage:
    python quantisation/ptq.py --model vanilla_ae
    python quantisation/ptq.py --model conv_ae
    python quantisation/ptq.py --model vae
    python quantisation/ptq.py --model all

Model interface (from models/base.py — DO NOT MODIFY):
    output = model(x)            # AEOutput with .x_hat guaranteed
    size   = model.model_size_mb()
    params = model.count_parameters()

Outputs:
    outputs/quantisation_results.csv   — logged via ResultLogger
"""

import os
import time
import argparse
import torch
from utils.dataset import create_synthetic_data, create_dataloaders
from utils.reproducibility import set_seed
from utils.csv_logger import ResultLogger


# ---------------------------------------------------------------------------
# Size measurement
# ---------------------------------------------------------------------------

def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Measure the on-disk size of a model's state dict in megabytes.
    Uses a temporary file to get the real serialised size.
    """
    os.makedirs("outputs", exist_ok=True)
    tmp_path = os.path.join("outputs", "._tmp_model_size_check.pt")
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 ** 2)
    os.remove(tmp_path)
    return round(size_mb, 4)


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------

def measure_inference_latency_ms(
    model: torch.nn.Module,
    x: torch.Tensor,
    n_warmup: int = 20,
    n_runs: int = 200,
) -> float:
    """
    Measure mean inference latency in milliseconds.

    Args:
        model:    Any model extending BaseAutoencoder.
        x:        A single-sample batch — shape (1, 12, 1000).
        n_warmup: Warm-up passes before timing starts.
        n_runs:   Number of timed forward passes.

    Returns:
        Mean latency in milliseconds (float).
    """
    model.eval()
    x = x.cpu()
    with torch.inference_mode():
        for _ in range(n_warmup):
            model(x)

        start = time.perf_counter()
        for _ in range(n_runs):
            model(x)
        end = time.perf_counter()

    latency_ms = (end - start) / n_runs * 1000
    return round(latency_ms, 4)


# ---------------------------------------------------------------------------
# Quantisation
# ---------------------------------------------------------------------------

def apply_dynamic_quantisation(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply PyTorch dynamic post-training quantisation (FP32 -> INT8).

    Dynamic quantisation targets Linear layers only.
    Conv1d requires static quantisation or ONNX Runtime (Sprint 3).
    Weights are quantised statically; activations are quantised on-the-fly.

    Args:
        model: A trained BaseAutoencoder subclass (eval mode).

    Returns:
        Quantised model (CPU-only, torch.qint8).
    """
    model.eval()
    quantised = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={torch.nn.Linear},
        dtype=torch.qint8,
    )
    return quantised


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def _make_temp_ae() -> torch.nn.Module:
    """
    Temporary placeholder model used when the real model is not yet implemented.
    Remove once Shardul and Kaan finish their implementations.
    """
    import torch.nn as nn
    import torch.nn.functional as F
    from models.base import BaseAutoencoder, AEOutput

    class _TempAE(BaseAutoencoder):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(12000, 32)
            self.fc2 = nn.Linear(32, 12000)

        def forward(self, x):
            b = x.shape[0]
            z = F.relu(self.fc1(x.view(b, -1)))
            x_hat = self.fc2(z).view(b, 12, 1000)
            return AEOutput(x_hat=x_hat)

        def compute_loss(self, x, output, **kwargs):
            return (F.mse_loss(output.x_hat, x),)

    return _TempAE()


def load_model(model_name: str) -> torch.nn.Module:
    """
    Load a model by name.  Only imports from the shared models/ directory.
    Falls back to a temporary placeholder if the real model is not yet implemented.

    Args:
        model_name: One of 'vanilla_ae', 'conv_ae', 'vae'.

    Returns:
        Instantiated model in eval mode.
    """
    if model_name == "vanilla_ae":
        try:
            from models.vanilla_ae import VanillaAE
            return VanillaAE()
        except (ImportError, AttributeError):
            print(f"[WARNING] VanillaAE not implemented yet — using TempAE placeholder.")
            return _make_temp_ae()

    elif model_name == "conv_ae":
        try:
            from models.conv_ae import ConvAE
            return ConvAE()
        except (ImportError, AttributeError):
            print(f"[WARNING] ConvAE not implemented yet — using TempAE placeholder.")
            return _make_temp_ae()

    elif model_name == "vae":
        try:
            from models.vae import VAE
            return VAE()
        except (ImportError, AttributeError):
            print(f"[WARNING] VAE not implemented yet — using TempAE placeholder.")
            return _make_temp_ae()

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            "Choose from: vanilla_ae, conv_ae, vae"
        )


# ---------------------------------------------------------------------------
# Single model PTQ run
# ---------------------------------------------------------------------------

def run_ptq_single(
    model_name: str,
    logger: ResultLogger,
    seed: int = 42,
) -> None:
    """
    Run the full PTQ pipeline for one model and one seed.

    Steps:
        1. Load model.
        2. Measure FP32 size and latency.
        3. Apply dynamic quantisation.
        4. Measure INT8 size and latency.
        5. Log both results via ResultLogger.

    Args:
        model_name: 'vanilla_ae', 'conv_ae', or 'vae'.
        logger:     Shared ResultLogger instance.
        seed:       Random seed for reproducibility.
    """
    set_seed(seed)

    # --- build a single-sample synthetic batch (no real data needed) ---
    splits = create_synthetic_data(n_train=64, n_val=32, n_test=32)
    loaders = create_dataloaders(splits, batch_size=1)
    x_sample, _ = next(iter(loaders["test"]))   # shape: (1, 12, 1000)
    x_sample = x_sample.cpu()

    # --- FP32 baseline ---
    model_fp32 = load_model(model_name).cpu()
    model_fp32.eval()

    fp32_size_mb   = get_model_size_mb(model_fp32)
    fp32_latency   = measure_inference_latency_ms(model_fp32, x_sample)
    fp32_params    = model_fp32.count_parameters()

    # --- INT8 quantised ---
    model_int8     = apply_dynamic_quantisation(model_fp32)
    int8_size_mb   = get_model_size_mb(model_int8)
    int8_latency   = measure_inference_latency_ms(model_int8, x_sample)

    # --- sanity check: INT8 output must have x_hat with correct shape ---
    with torch.inference_mode():
        out = model_int8(x_sample)
        assert hasattr(out, "x_hat"), "INT8 model output missing x_hat"
        assert out.x_hat.shape == x_sample.shape, (
            f"Shape mismatch: got {out.x_hat.shape}, expected {x_sample.shape}"
        )

    # --- derived metrics ---
    size_reduction_pct    = round((1 - int8_size_mb / fp32_size_mb) * 100, 2)
    speedup_ratio         = round(fp32_latency / int8_latency, 3) if int8_latency > 0 else None

    # --- console summary ---
    print(f"\n{'=' * 52}")
    print(f"  Model : {model_name}   |   Seed : {seed}")
    print(f"{'=' * 52}")
    print(f"  {'':12s}  {'Size (MB)':>10}  {'Latency (ms)':>14}")
    print(f"  {'FP32':12s}  {fp32_size_mb:>10.4f}  {fp32_latency:>14.4f}")
    print(f"  {'INT8':12s}  {int8_size_mb:>10.4f}  {int8_latency:>14.4f}")
    print(f"  {'Reduction':12s}  {size_reduction_pct:>9.1f}%  "
          f"  speedup x{speedup_ratio}")
    print(f"  Parameters : {fp32_params:,}")
    print(f"{'=' * 52}\n")

    # --- log FP32 row ---
    logger.log(
        model=model_name,
        setting="centralised",
        precision_type="fp32",
        seed=seed,
        model_size_mb=fp32_size_mb,
        inference_latency_ms=fp32_latency,
        auroc=None, auprc=None, sensitivity=None,
        specificity=None, precision_score=None,
        f1=None, epsilon=None, training_time_s=None,
    )

    # --- log INT8 row ---
    logger.log(
        model=model_name,
        setting="centralised",
        precision_type="int8",
        seed=seed,
        model_size_mb=int8_size_mb,
        inference_latency_ms=int8_latency,
        auroc=None, auprc=None, sensitivity=None,
        specificity=None, precision_score=None,
        f1=None, epsilon=None, training_time_s=None,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-Training Quantisation pipeline — Ghadah (Person E)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vanilla_ae",
        choices=["vanilla_ae", "conv_ae", "vae", "all"],
        help="Model to quantise. Use 'all' to run all three.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42). Use SEEDS=[42,123,456] for full runs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/quantisation_results.csv",
        help="Path to output CSV file.",
    )
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    logger = ResultLogger(args.output)

    models_to_run = (
        ["vanilla_ae", "conv_ae", "vae"]
        if args.model == "all"
        else [args.model]
    )

    for model_name in models_to_run:
        run_ptq_single(model_name, logger, seed=args.seed)

    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
