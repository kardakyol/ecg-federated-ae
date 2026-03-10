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
from fl.model_factory import get_model


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

    def get_parameters(self, config):
        return self.model.get_parameters()

    def set_parameters(self, parameters):
        self.model.set_parameters(parameters) 

    def fit(self, parameters, config):
        # Standard Federated Training
        start_time = time.time()
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        epochs = config.get("local_epochs", 1)
        total_loss = 0.0
        num_batches = 0

        for epoch in range(epochs):
            for batch in self.loaders["train"]:
                x = batch[0].to(self.device)
                optimizer.zero_grad()
                loss, *_ = self.model.compute_loss(x, self.model(x))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        
        self.last_train_time = time.time() - start_time
        avg_loss = total_loss / max(num_batches, 1)

        return self.get_parameters(config={}), len(self.loaders["train"].dataset), {
            "train_loss": float(avg_loss),
            "train_time": self.last_train_time
        }
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # Post Global fine tuning
        self.model.train()
        ft_optimizer=torch.optim.Adam(self.model.parameters(), lr=0.0001)
        fine_tune_epochs = 5

        for _ in range(fine_tune_epochs):
            for batch in self.loaders["train"]:
                x = batch[0].to(self.device)
                ft_optimizer.zero_grad()
                loss, *_ = self.model.compute_loss(x, self.model(x))
                loss.backward()
                ft_optimizer.step()
        
        self.model.eval()
        all_scores = []
        all_labels = []
        start_inf = time.time()

        with torch.no_grad():
            for batch in self.loaders["val"]:
                x = batch[0].to(self.device)
                y = batch[1] if len(batch) > 1 else torch.zeros(x.shape[0])
                output = self.model(x)
                score = torch.mean((output.x_hat - x)**2, dim=(1,2))
                all_scores.extend(score.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        total_inf_time = (time.time() - start_inf) * 1000
        avg_latency = total_inf_time / len(self.loaders["val"].dataset)

        scores_arr = np.array(all_scores)
        labels_arr = np.array(all_labels)
        threshold = np.percentile(scores_arr, 95)
        
        if len(np.unique(labels_arr)) < 2:
            return float(np.mean(scores_arr)), len(self.loaders["val"].dataset), {
                "val_loss": float(np.mean(scores_arr)),
            }

        metrics = compute_metrics(labels_arr, scores_arr, threshold)
        result_dict = metrics.to_dict()
        result_dict.update({
            "training_time_s": self.last_train_time,
            "inference_latency_ms": avg_latency,
            "model_size_mb": self.model.model_size_mb(),
            "precision_score": result_dict.get("precision", 0.0),
            "flops_m": 0.0,
            "epsilon": "N/A"
        })

        # record memory usage for DP efficiency analysis
        if torch.cuda.is_available():
            result_dict["peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
        
        return float(metrics.auroc), len(self.loaders["val"].dataset), result_dict