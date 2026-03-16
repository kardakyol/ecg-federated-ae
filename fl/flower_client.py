"""
RAHEEB: Flower client and server configuration.
Optimized for Sprint 2 completion and Sprint 3 stability.
"""

import flwr as fl
import torch
import numpy as np 
import time
import platform
import gc
from pathlib import Path
from torch.utils.data import Subset
from models.base import BaseAutoencoder, AEOutput
from utils.dataset import load_splits, create_dataloaders
from evaluation.metrics import compute_metrics
from evaluation.compute_cost import compute_all_costs
from fl.model_factory import get_model
from privacy.dp_sgd import make_private, get_epsilon
from opacus.accountants.utils import get_noise_multiplier
import psutil
import os
import io
from sklearn.metrics import roc_auc_score

def get_model_size_mb(model):
    """Accurately measures model size including quantized PackedParams."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 * 1024)

class ECGClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, model_type: str = "vanilla", alpha: float = 0.5):
        self.client_id = client_id
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize model form factory
        self.model = get_model(model_type).to(self.device)

        # Data loading logic
        data_dir = Path("data/ptb-xl-zscore")
        all_splits = load_splits(data_dir)
        indices_path = data_dir / "client_splits" / f"client_{client_id}_indices.npy"

        if client_id == "dry-run" or not indices_path.exists():
            print(f"[!] {client_id}: Shard not found or dry-run. Using full training set.")
            self.splits = {
                "train": all_splits["train"],
                "val": all_splits["val"],
                "test": all_splits["test"]
            }
        else:
            client_indices = np.load(indices_path)
            train_subset = Subset(all_splits["train"], client_indices)
            train_subset.labels = all_splits["train"].labels[client_indices]
            self.splits = {
                "train": train_subset,
                "val": all_splits["val"],
                "test": all_splits["test"]
            }
            
        self.loaders = create_dataloaders(self.splits, batch_size=32)
        self.last_train_time = 0.0
        # Track Privacy Engine state
        self.privacy_engine = None

        self.test_subclass = None
        sub_path = Path("data/ptb-xl/test_subclass_labels.npy")
        if sub_path.exists():
            self.test_subclass = np.load(sub_path)
            # Ensure it matches test set length if using a subset
            test_len = len(self.splits["test"])
            self.test_subclass = self.test_subclass[:test_len]

    def get_parameters(self, config):
        model_to_use = self.model._module if hasattr(self.model, "_module") else self.model
        return model_to_use.get_parameters()

    def set_parameters(self, parameters):
        model_to_use = self.model._module if hasattr(self.model, "_module") else self.model
        model_to_use.set_parameters(parameters) 

    def fit(self, parameters, config):
        train_loader=self.loaders["train"]
        if len(train_loader) == 0:
            print(f"[!] Client {self.client_id}: No training data. Skipping round.")
            return parameters, 0, {"train_loss": 0.0, "epsilon": 0.0}
        
        # Standard Federated Training
        start_time = time.perf_counter()
        self.set_parameters(parameters)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        train_loader = self.loaders["train"]
        self.privacy_engine = None

        #──── DP Configuration ───────────────────────
        target_epsilon = config.get("epsilon", float('inf'))
        noise_multiplier = 0.0

        if target_epsilon < float('inf'):
            dataset_size = len(train_loader.dataset)
            sample_rate=train_loader.batch_size / dataset_size if dataset_size > 0 else 1.0

            # Calculate the noise required to reach target_epsilon
            # sample_rate = batch_size / total_samples
            try:
                noise_multiplier = get_noise_multiplier(
                    target_epsilon=target_epsilon,
                    target_delta=1e-5,
                    sample_rate=sample_rate,
                    epochs=config.get("local_epochs", 1)
                )
                self.model, optimizer, train_loader, self.privacy_engine = make_private(
                    model=self.model,
                    optimizer=optimizer,
                    dataloader=train_loader,
                    noise_multiplier=noise_multiplier,
                    max_grad_norm=1.0
                )
            except Exception as e:
                print(f"[!] DP Math error: {e}. Falling back to non-private for this client.")

        # Baseline: No DP wrapping
        self.model.train()
        epochs = config.get("local_epochs", 1)
        total_loss = 0.0
        num_batches = 0

        target_model = self.model._module if hasattr(self.model, "_module") else self.model
        for epoch in range(epochs):
            for batch in train_loader:
                x = batch[0].to(self.device)
                optimizer.zero_grad()
                output=self.model(x)
                loss, *_ = target_model.compute_loss(x, output)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        
        self.last_train_time = time.perf_counter() - start_time
        avg_loss = total_loss / max(num_batches, 1)

        # Calculate final epsilon
        actual_eps = get_epsilon(self.privacy_engine, delta=1e-5) if self.privacy_engine else float('inf')

        torch.cuda.empty_cache()
        gc.collect()

        return self.get_parameters(config={}), len(self.loaders["train"].dataset), {
            "train_loss": float(avg_loss),
            "training_time_s": float(self.last_train_time),
            "epsilon": float(actual_eps),
            "noise_multiplier": float(noise_multiplier)
        }
    
    def evaluate(self, parameters, config):
        if len(self.loaders["train"]) == 0 or len(self.loaders["val"]) == 0:
            return 0.0, 0, {"val_loss": 0.0, "status": "no_data"}
        
        self.set_parameters(parameters)
        target_model = self.model._module if hasattr(self.model, "_module") else self.model
        
        # ── Post Global fine tuning ──────────────────────────────────────────────
        self.model.train()
        ft_optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-5)
        fine_tune_epochs = 1

        for _ in range(fine_tune_epochs):
            for batch in self.loaders["train"]:
                x = batch[0].to(self.device)
                ft_optimizer.zero_grad()
                output=self.model(x)
                loss, *_ = target_model.compute_loss(x, output)
                loss.backward()
                ft_optimizer.step()

        arch_costs = compute_all_costs(target_model, device=self.device)
        # ── Model Quantization Phase ──────────────────────────────────────────────
        precision_type = config.get("precision_type", "fp32")
        eval_device = self.device
        if precision_type == "int8":
            # Setting ARM optimization engine for running on Raspeberry PI
            if platform.machine() in ['armv7l', 'aarch64']:
                torch.backends.quantized.engine = 'qnnpack'
                print("[INFO] Using QNNPACK engine for Raspberry PI")
            # Applying Quantization
            target_model = target_model.to("cpu")
            from quantisation.ptq import apply_dynamic_quantisation
            eval_model = apply_dynamic_quantisation(target_model.to("cpu"))
            eval_device = torch.device("cpu")
            print("[INFO] int8 detected: Running evaluation on CPU.")
        else:
            eval_model = target_model

        # ── Inference loop ──────────────────────────────────────────────
        eval_model.eval()
        # Reset CUDA stats at the start of evaluation to get a "per-round" peak
        if torch.cuda.is_available() and eval_device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device=eval_device)

        if precision_type == "int8":
            from evaluation.compute_cost import measure_inference_time, measure_peak_memory

            dummy_input = torch.randn((1, 12, 1000), device=eval_device)
            latency_metrics = measure_inference_time(eval_model, dummy_input)
            memory_metrics = measure_peak_memory(eval_model, dummy_input)

            final_costs = {
                "flops_m": arch_costs["flops_m"],
                "params_m": arch_costs["params_m"],
                "inference_latency_ms": latency_metrics["inference_latency_ms"],
                "inference_latency_std_ms": latency_metrics.get("inference_latency_std_ms", 0.0),
                "peak_memory_mb": memory_metrics["peak_memory_mb"]
            }
        else:
            final_costs = compute_all_costs(eval_model, device=eval_device)
        
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.loaders["test"]:
                x = batch[0].to(eval_device)
                y = batch[1] if len(batch) > 1 else torch.zeros(x.shape[0])
                output = eval_model(x)
                x_hat = output.x_hat if hasattr(output, 'x_hat') else output[0]
                score = torch.mean((x_hat.to(eval_device) - x)**2, dim=(1,2))
                all_scores.extend(score.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        scores_arr = np.concatenate([np.atleast_1d(s) for s in all_scores])
        labels_arr = np.concatenate([np.atleast_1d(l) for l in all_labels])
        threshold = np.percentile(scores_arr, 95)
        
        if len(np.unique(labels_arr)) < 2:
            return 0.0, len(self.loaders["test"].dataset), {"status": "no_labels"}

        metrics = compute_metrics(labels_arr, scores_arr, threshold)
        result_dict = metrics.to_dict()
        result_dict["auroc"] = float(metrics.auroc)

        if self.test_subclass is not None:
            norm_mask = (self.test_subclass == 0)
            for cls_idx, cls_name in {1: "MI", 2: "STTC", 3: "HYP", 4: "CD"}.items():
                cls_mask = (self.test_subclass == cls_idx)
                mask = norm_mask | cls_mask
                if np.sum(cls_mask) > 0:
                    try:
                        sub_scores = scores_arr[mask]
                        sub_labels = (self.test_subclass[mask] == cls_idx).astype(int)
                        result_dict[f"{cls_name}_auroc"] = float(roc_auc_score(sub_labels, sub_scores))
                    except:
                        result_dict[f"{cls_name}_auroc"] = 0.0

        # ── System Tracking ───────────────────────────────────────────────────
        result_dict.update(final_costs)
        result_dict["model_size_mb"] = get_model_size_mb(eval_model)
        
        # System RAM (For Raspberry Pi)
        process = psutil.Process(os.getpid())
        peak_mem=process.memory_info().rss
        result_dict["peak_memory_mb"] = float(round(peak_mem / (1024**2), 2))
        
        # For GPU VRAM (For Simulation run)
        if torch.cuda.is_available() and eval_device.type == 'cuda':
            result_dict["peak_memory_mb"] = float(round(torch.cuda.max_memory_allocated(device=eval_device) / (1024**2), 2))
        
        result_dict.update({
            "training_time_s": float(self.last_train_time),
            "epsilon": str(config.get("epsilon", "N/A")),
            "precision_type": config.get("precision_type", "fp32"),
            "evaluation_type": "personalized_1_epoch"
        })
        if precision_type == "int8":
            del eval_model

        self.model.to(self.device)
        gc.collect()
        torch.cuda.empty_cache()
        
        return float(np.mean(scores_arr)), len(self.loaders["val"].dataset), result_dict

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flower ECG Client")
    parser.add_argument("--client_id", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="conv", choices=["vanilla", "conv", "vae"])
    parser.add_argument("--server_address", type=str, default="localhost:8080",
                        help="IP of the server (use default for laptop simulation)")

    args = parser.parse_args()
    # Start the client
    fl.client.start_numpy_client(
        server_address=args.server_address, 
        client=ECGClient(client_id=args.client_id, model_type=args.model_type),
        grpc_max_message_length=1024*1024*1024
    )