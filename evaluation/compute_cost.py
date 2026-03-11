"""
SHARED COMPUTATION COST METRICS
Measures FLOPs (via ptflops), inference latency, and peak memory.

Returned dict keys match STANDARD_COLUMNS in utils/csv_logger.py:
    flops_m, inference_latency_ms, peak_memory_mb

USAGE:
    from evaluation.compute_cost import compute_all_costs
    costs = compute_all_costs(model, device)
    logger.log(model="conv_ae", **costs)

WHO USES THIS:
    Ghadah   - quantisation efficiency table
    Everyone - ablation study, paper Table II (computation efficiency)
"""
from __future__ import annotations

import logging
import time
import tracemalloc
from typing import Dict, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 12, 1000),
) -> Dict[str, float]:
    """Compute FLOPs and parameter count using ptflops.

    Returns dict with:
        flops_m:  mega-FLOPs (MAC-based, ×2 for FLOPs)
        params_m: million trainable parameters
    """
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        logger.warning(
            "ptflops not installed — install with: pip install -e '.[edge]'. "
            "Returning zeros for FLOPs."
        )
        return {"flops_m": 0.0, "params_m": 0.0}

    was_training = model.training
    model.eval()

    input_res = input_shape[1:]  # ptflops expects (C, T) not (B, C, T)

    try:
        macs, params = get_model_complexity_info(
            model,
            input_res,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        flops_m = (macs * 2) / 1e6
        params_m = params / 1e6
    except Exception as e:
        logger.warning(f"ptflops failed: {e}. Returning zeros.")
        flops_m, params_m = 0.0, 0.0

    if was_training:
        model.train()

    return {"flops_m": round(flops_m, 4), "params_m": round(params_m, 4)}


def measure_inference_time(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> Dict[str, float]:
    """Benchmark forward-pass latency with proper warmup and GPU sync.

    Returns dict with:
        inference_latency_ms:     mean latency in milliseconds
        inference_latency_std_ms: std of latency across runs
    """
    device = input_tensor.device
    use_cuda = device.type == "cuda"
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for _ in range(n_warmup):
            model(input_tensor)
            if use_cuda:
                torch.cuda.synchronize(device)

        timings = []
        for _ in range(n_runs):
            if use_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            model(input_tensor)
            if use_cuda:
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

    if was_training:
        model.train()

    mean_ms = sum(timings) / len(timings)
    var_ms = sum((t - mean_ms) ** 2 for t in timings) / len(timings)
    std_ms = var_ms ** 0.5

    return {
        "inference_latency_ms": round(mean_ms, 4),
        "inference_latency_std_ms": round(std_ms, 4),
    }


def measure_peak_memory(
    model: nn.Module,
    input_tensor: torch.Tensor,
) -> Dict[str, float]:
    """Measure peak memory during a single forward pass.

    Uses torch.cuda memory stats on CUDA, tracemalloc on CPU/MPS.

    Returns dict with:
        peak_memory_mb: peak memory allocated in megabytes
    """
    device = input_tensor.device
    was_training = model.training
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        with torch.no_grad():
            model(input_tensor)
        torch.cuda.synchronize(device)
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_mb = peak_bytes / (1024 ** 2)
    else:
        tracemalloc.start()
        with torch.no_grad():
            model(input_tensor)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak_bytes / (1024 ** 2)

    if was_training:
        model.train()

    return {"peak_memory_mb": round(peak_mb, 4)}


def compute_all_costs(
    model: nn.Module,
    device: torch.device | str = "cpu",
    input_shape: Tuple[int, ...] = (1, 12, 1000),
) -> Dict[str, float]:
    """Convenience wrapper: FLOPs + inference time + peak memory.

    Returns flat dict with keys matching STANDARD_COLUMNS:
        flops_m, params_m, inference_latency_ms,
        inference_latency_std_ms, peak_memory_mb, model_size_mb
    """
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)

    results: Dict[str, float] = {}
    results.update(compute_flops(model, input_shape))

    dummy = torch.randn(*input_shape, device=device)
    results.update(measure_inference_time(model, dummy))
    results.update(measure_peak_memory(model, dummy))

    if hasattr(model, "model_size_mb"):
        results["model_size_mb"] = round(model.model_size_mb(), 4)

    return results
