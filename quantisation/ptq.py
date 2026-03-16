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

Notes on measurements:
    - FLOPs are measured on FP32 architecture only. INT8 dynamic
      quantisation does not change the model architecture, only the
      numerical precision of weights. Both rows share the same FLOPs.
    - peak_memory_mb stores a process RSS delta estimate (before vs
      after inference) measured via psutil. This is not a true
      instantaneous hardware peak profiler. In the report, describe
      it as: "process RSS delta during inference".
    - Model size is measured as on-disk size of the saved state dict
      (via get_model_size_mb), NOT via model.model_size_mb() which
      returns an in-memory estimate (params x 4 bytes). On-disk
      measurement is used because it captures actual INT8 compression
      after quantisation, which in-memory estimation does not reflect.
"""

import gc
import os
import time
import argparse
import importlib
import torch
import psutil

from utils.dataset import create_synthetic_data, create_dataloaders
from utils.reproducibility import set_seed
from utils.csv_logger import ResultLogger


# ---------------------------------------------------------------------------
# Size measurement
# ---------------------------------------------------------------------------

def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Measure the on-disk size of a model's state dict in megabytes.

    Saves the state dict to a unique temporary file, reads its size,
    then deletes it. Using os.getpid() in the filename avoids conflicts
    if the script is ever run in parallel.

    Note: we use on-disk size rather than model.model_size_mb() because
    the base class estimates size as (params x 4 bytes), which does not
    reflect actual INT8 compression. On-disk measurement captures the
    real storage difference between FP32 and INT8 models.
    """
    os.makedirs("outputs", exist_ok=True)
    tmp_path = os.path.join(
        "outputs", f"._tmp_model_size_check_{os.getpid()}.pt"
    )
    try:
        torch.save(model.state_dict(), tmp_path)
        size_mb = os.path.getsize(tmp_path) / (1024 ** 2)
        return round(size_mb, 4)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------

def measure_inference_latency_ms(
    model: torch.nn.Module,
    x: torch.Tensor,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> float:
    """
    Measure mean inference latency in milliseconds.

    Runs n_warmup passes first (results discarded) to eliminate
    cold-start bias from CPU caching and memory paging. Then times
    n_runs passes and returns the mean.

    Both model.eval() and torch.inference_mode() are required:
    - eval()             disables dropout and batchnorm training behaviour
    - inference_mode()   disables gradient tracking for maximum speed
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
# FLOPs measurement
# ---------------------------------------------------------------------------

def measure_flops_m(model: torch.nn.Module, x: torch.Tensor) -> float:
    """
    Measure FLOPs (floating point operations) in millions using ptflops.

    FLOPs are measured on the FP32 model only. Dynamic INT8 quantisation
    changes the numerical precision of weights but does NOT change the
    model architecture, so FP32 and INT8 share the same FLOPs count.

    ptflops counts MACs (multiply-accumulate operations). Each MAC
    equals one multiply + one add = 2 FLOPs, so we multiply by 2.

    Returns 0.0 with a warning if ptflops is unavailable or fails.
    """
    try:
        from ptflops import get_model_complexity_info

        model.eval()
        input_shape = tuple(x.shape[1:])  # (12, 1000) — no batch dimension

        macs, _ = get_model_complexity_info(
            model,
            input_shape,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )

        flops_m = round((macs * 2) / 1e6, 4) if macs else 0.0
        return flops_m

    except Exception as e:
        print(
            f"[WARNING] FLOPs measurement failed for "
            f"{model.__class__.__name__}: {e}"
        )
        return 0.0


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------

def measure_peak_memory_mb(model: torch.nn.Module, x: torch.Tensor) -> float:
    """
    Estimate process-level memory increase during a single forward pass.

    Measures the OS-level Resident Set Size (RSS) before and after
    inference using psutil. The delta is the memory increase caused
    by one forward pass.

    gc.collect() is called before the baseline reading to flush Python
    garbage and reduce noise in the measurement.

    IMPORTANT: This is a process RSS delta estimate, not a true
    instantaneous hardware peak profiler. Small values or 0.0 are
    expected for lightweight models — this is normal OS behaviour.
    In the paper, describe as: "process RSS delta during inference".
    """
    model.eval()
    x = x.cpu()

    gc.collect()
    process = psutil.Process(os.getpid())
    baseline_mb = process.memory_info().rss / (1024 ** 2)

    with torch.inference_mode():
        _ = model(x)

    after_mb = process.memory_info().rss / (1024 ** 2)
    return round(max(after_mb - baseline_mb, 0.0), 4)


# ---------------------------------------------------------------------------
# Quantisation
# ---------------------------------------------------------------------------

def apply_dynamic_quantisation(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply PyTorch dynamic post-training quantisation (FP32 -> INT8).

    Targets Linear layers only. Weights are quantised statically to
    INT8; activations are quantised dynamically on each forward pass.

    Conv1d layers are NOT quantised by this method — they require
    static quantisation or ONNX Runtime (planned for Sprint 3).

    Note: torch.quantization.quantize_dynamic is deprecated in PyTorch
    2.10+. It remains functional for now but will need migration to
    torchao in a future sprint.
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
    Temporary placeholder autoencoder used when the real model is
    not yet available. Remove once Shardul and Kaan push their models.

    Inherits count_parameters() and model_size_mb() from BaseAutoencoder,
    so the full pipeline interface is satisfied without extra methods here.
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
    Load a model by name from the shared models/ directory.
    Falls back to _TempAE if the real model is not yet implemented.

    Args:
        model_name: One of 'vanilla_ae', 'conv_ae', 'vae'.

    Returns:
        Instantiated model ready for eval mode.
    """
    _MODELS = {
        "vanilla_ae": ("models.vanilla_ae", "VanillaAE"),
        "conv_ae":    ("models.conv_ae",    "ConvAE"),
        "vae":        ("models.vae",        "VAE"),
    }

    if model_name not in _MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            "Choose from: vanilla_ae, conv_ae, vae"
        )

    module_path, class_name = _MODELS[model_name]
    try:
        # importlib imported at top of file
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls()
    except (ImportError, AttributeError, TypeError) as e:
        print(
            f"[WARNING] Failed to load {class_name}: {e} "
            "— using TempAE placeholder."
        )
        return _make_temp_ae()


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
        1. Create a synthetic single-sample batch.
        2. Load FP32 model and measure: size, latency, FLOPs, RSS delta.
        3. Apply dynamic INT8 quantisation.
        4. Measure INT8: size, latency, RSS delta.
           (INT8 FLOPs = FP32 FLOPs — architecture unchanged.)
        5. Sanity check: INT8 output x_hat shape must match input.
        6. Log FP32 and INT8 rows to CSV via ResultLogger.

    If the model fails at any step, a warning is printed and the
    pipeline moves on to the next model without crashing.
    KeyboardInterrupt is re-raised so Ctrl+C always stops the program.

    Args:
        model_name: 'vanilla_ae', 'conv_ae', or 'vae'.
        logger:     Shared ResultLogger instance.
        seed:       Random seed for reproducibility.
    """
    set_seed(seed)

    try:
        # single-sample batch — shape (1, 12, 1000)
        splits = create_synthetic_data(n_train=64, n_val=32, n_test=32)
        loaders = create_dataloaders(splits, batch_size=1)
        x_sample, _ = next(iter(loaders["test"]))
        x_sample = x_sample.cpu()

        # ----------------------------------------------------------------
        # FP32 baseline
        # ----------------------------------------------------------------
        model_fp32 = load_model(model_name).cpu()
        model_fp32.eval()

        fp32_size_mb  = get_model_size_mb(model_fp32)
        fp32_latency  = measure_inference_latency_ms(model_fp32, x_sample)
        fp32_flops_m  = measure_flops_m(model_fp32, x_sample)
        fp32_peak_mem = measure_peak_memory_mb(model_fp32, x_sample)
        fp32_params   = model_fp32.count_parameters()

        # ----------------------------------------------------------------
        # INT8 quantised
        # ----------------------------------------------------------------
        model_int8 = apply_dynamic_quantisation(model_fp32)

        int8_size_mb  = get_model_size_mb(model_int8)
        int8_latency  = measure_inference_latency_ms(model_int8, x_sample)
        int8_flops_m  = fp32_flops_m   # same architecture — FLOPs unchanged
        int8_peak_mem = measure_peak_memory_mb(model_int8, x_sample)

        # ----------------------------------------------------------------
        # Sanity check
        # ----------------------------------------------------------------
        with torch.inference_mode():
            out = model_int8(x_sample)
            assert hasattr(out, "x_hat"), \
                "INT8 model output is missing x_hat attribute"
            assert out.x_hat.shape == x_sample.shape, (
                f"Shape mismatch: got {out.x_hat.shape}, "
                f"expected {x_sample.shape}"
            )

        # ----------------------------------------------------------------
        # Derived metrics
        # ----------------------------------------------------------------
        if fp32_size_mb > 0:
            size_reduction_pct = round(
                (1 - int8_size_mb / fp32_size_mb) * 100, 2
            )
        else:
            size_reduction_pct = 0.0

        speedup_ratio = (
            round(fp32_latency / int8_latency, 3)
            if int8_latency > 0 else None
        )
        speedup_text = f"x{speedup_ratio}" if speedup_ratio is not None else "N/A"

        # ----------------------------------------------------------------
        # Console summary
        # ----------------------------------------------------------------
        print(f"\n{'=' * 68}")
        print(f"  Model : {model_name}   |   Seed : {seed}")
        print(f"{'=' * 68}")
        print(
            f"  {'':12s}  {'Size (MB)':>10}  {'Latency (ms)':>13}"
            f"  {'FLOPs (M)':>10}  {'RSS δ (MB)':>10}"
        )
        print(
            f"  {'FP32':12s}  {fp32_size_mb:>10.4f}  {fp32_latency:>13.4f}"
            f"  {fp32_flops_m:>10.4f}  {fp32_peak_mem:>10.4f}"
        )
        print(
            f"  {'INT8':12s}  {int8_size_mb:>10.4f}  {int8_latency:>13.4f}"
            f"  {int8_flops_m:>10.4f}  {int8_peak_mem:>10.4f}"
        )
        print(
            f"  {'Reduction':12s}  {size_reduction_pct:>9.1f}%"
            f"    speedup {speedup_text}"
        )
        print(f"  Parameters : {fp32_params:,}")
        print(f"{'=' * 68}\n")

        # ----------------------------------------------------------------
        # Log to CSV
        # peak_memory_mb stores process RSS delta estimate — see docstring
        # ----------------------------------------------------------------
        logger.log(
            model=model_name,
            setting="centralised",
            precision_type="fp32",
            seed=seed,
            model_size_mb=fp32_size_mb,
            inference_latency_ms=fp32_latency,
            flops_m=fp32_flops_m,
            peak_memory_mb=fp32_peak_mem,   # RSS delta estimate
            auroc=None, auprc=None, sensitivity=None,
            specificity=None, precision_score=None,
            f1=None, epsilon=None, training_time_s=None,
        )

        logger.log(
            model=model_name,
            setting="centralised",
            precision_type="int8",
            seed=seed,
            model_size_mb=int8_size_mb,
            inference_latency_ms=int8_latency,
            flops_m=int8_flops_m,
            peak_memory_mb=int8_peak_mem,   # RSS delta estimate
            auroc=None, auprc=None, sensitivity=None,
            specificity=None, precision_score=None,
            f1=None, epsilon=None, training_time_s=None,
        )

    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"[WARNING] {model_name} failed with error: {e} — skipping.")


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
        help="Random seed (default: 42). Use 42, 123, 456 for full runs.",
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