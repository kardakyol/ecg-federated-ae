"""
RAHEEB: Flower client and server configuration.
Optimized for Sprint 2 completion and Sprint 3 stability.
"""

import flwr as fl
import torch
import numpy as np 
import time
from pathlib import Path
from torch.utils.data import Subset
from models.base import BaseAutoencoder, AEOutput
from utils.dataset import load_splits, create_dataloaders
from evaluation.metrics import compute_metrics
from evaluation.compute_cost import compute_all_costs
from fl.model_factory import get_model
from privacy.dp_sgd import make_private, get_epsilon
from opacus.accountants.utils import get_noise_multiplier


class ECGClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, model_type: str = "vanilla", alpha: float = 0.5):
        self.client_id = client_id
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize model form factory
        self.model = get_model(model_type).to(self.device)

        # Data loading logic
        data_dir = Path("data/ptb-xl")
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
        target_epsilon = config.get("epsilon", None)
        noise_multiplier = 0.0

        if target_epsilon is not None:
            target_epsilon = float(target_epsilon)

        if target_epsilon is not None and target_epsilon < float('inf'):
            dataset_size = len(train_loader.dataset)
            sample_rate=train_loader.batch_size / dataset_size if dataset_size > 0 else 1.0

            # Calculate the noise required to reach target_epsilon
            # sample_rate = batch_size / total_samples
            try:
                if sample_rate >= 1.0:
                    raise ValueError("Sample rate >= 1.0 is mathematically impossible for DP.")
                noise_multiplier = get_noise_multiplier(
                    target_epsilon=target_epsilon,
                    target_delta=1e-5,
                    sample_rate=sample_rate,
                    epochs=config.get("local_epochs", 1)
                )
            except (ValueError, ZeroDivisionError, OverflowError):
                # FALLBACK: If the specific epsilon is impossible for this small dataset,
                # we use a high constant noise to allow the simulation to continue.
                print(f"[!] Client {self.client_id}: DP Math failed. Falling back to safe noise.")
                noise_multiplier = 2.0 if target_epsilon <= 1.0 else 1.0
            
            self.model, optimizer, train_loader, self.privacy_engine = make_private(
                model=self.model,
                optimizer=optimizer,
                dataloader=train_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=1.0
            )

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
        ft_optimizer=torch.optim.Adam(self.model.parameters(), lr=0.0001)
        fine_tune_epochs = 5

        for _ in range(fine_tune_epochs):
            for batch in self.loaders["train"]:
                x = batch[0].to(self.device)
                ft_optimizer.zero_grad()
                output=self.model(x)
                loss, *_ = target_model.compute_loss(x, output)
                loss.backward()
                ft_optimizer.step()
        
        # ── Model Quantization Phase ──────────────────────────────────────────────
        precision_type = config.get("precision_type", "fp32")
        eval_device = self.device
        if precision_type == "int8":
            from quantisation.ptq import apply_dynamic_quantisation
            self.model = self.model.to("cpu")
            self.model = apply_dynamic_quantisation(self.model)
            eval_device = torch.device("cpu")
            target_model=self.model

        # ── Inference loop ──────────────────────────────────────────────
        self.model.eval()
        all_scores = []
        all_labels = []
        start_inf = time.time()
        with torch.no_grad():
            for batch in self.loaders["val"]:
                x = batch[0].to(eval_device)
                y = batch[1] if len(batch) > 1 else torch.zeros(x.shape[0])
                output = self.model(x)
                x_hat = output.x_hat if hasattr(output, 'x_hat') else output[0]
                score = torch.mean((x_hat.to(eval_device) - x)**2, dim=(1,2))
                all_scores.extend(score.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        total_inf_time = (time.time() - start_inf) * 1000
        avg_latency = total_inf_time / max(len(self.loaders["val"].dataset), 1)

        scores_arr = np.concatenate([np.atleast_1d(s) for s in all_scores])
        labels_arr = np.concatenate([np.atleast_1d(l) for l in all_labels])
        threshold = np.percentile(scores_arr, 95)
        
        if len(np.unique(labels_arr)) < 2:
            return float(np.mean(scores_arr)), len(self.loaders["val"].dataset), {
                "val_loss": float(np.mean(scores_arr)),
                "epsilon": str(config.get("epsilon", "inf")),
            }

        metrics = compute_metrics(labels_arr, scores_arr, threshold)
        costs = compute_all_costs(target_model, device=eval_device)
        result_dict = metrics.to_dict()
        result_dict.update(costs)
        result_dict.update({
            "training_time_s": float(self.last_train_time),
            "epsilon": str(config.get("epsilon", "N/A")),
            "precision_type": config.get("precision_type", "fp32"),
        })

        # record memory usage for DP efficiency analysis
        if torch.cuda.is_available() and eval_device.type == 'cuda':
            result_dict["peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
        
        return float(metrics.auroc), len(self.loaders["val"].dataset), result_dict