"""
RAHEEB: Flower client for federated ECG anomaly detection.

Model interface (from models/base.py):
    model.get_parameters() -> List[np.ndarray]
    model.set_parameters(List[np.ndarray])
    model.forward(x) -> AEOutput (with .x_hat)
    model.compute_loss(x, output) -> (total_loss, ...)

Data loading:
    from utils.dataset import load_splits, create_dataloaders
"""

import flwr as fl
import torch
import numpy as np

from models.base import BaseAutoencoder, AEOutput
from utils.dataset import create_synthetic_data, create_dataloaders
from evaluation.metrics import compute_metrics
from fl.model_factory import get_model


class ECGClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, model_type: str = "vanilla",
                 data_splits=None, batch_size: int = 32):
        self.client_id = client_id
        self.model_type = model_type
        self.model = get_model(model_type)

        if data_splits is not None:
            self.splits = data_splits
        else:
            self.splits = create_synthetic_data(n_train=200, n_val=50, n_test=50)

        self.loaders = create_dataloaders(self.splits, batch_size=batch_size)

    def get_parameters(self, config):
        return self.model.get_parameters()

    def set_parameters(self, parameters):
        self.model.set_parameters(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        epochs = config.get("local_epochs", 5)
        beta = config.get("beta", 0.5)
        total_loss = 0.0
        num_batches = 0

        # KL annealing for VAE: ramp kl_weight from 0 to 1 over epochs
        is_vae = self.model_type == "vae"

        for epoch in range(epochs):
            self.model.train()
            kl_weight = min(1.0, (epoch + 1) / max(epochs, 1)) if is_vae else 1.0

            for batch in self.loaders["train"]:
                x = batch[0]
                optimizer.zero_grad()
                output = self.model(x)

                if is_vae:
                    loss, *_ = self.model.compute_loss(
                        x, output, beta=beta, kl_weight=kl_weight
                    )
                else:
                    loss, *_ = self.model.compute_loss(x, output)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        return self.get_parameters(config={}), len(self.loaders["train"].dataset), {
            "train_loss": float(avg_loss),
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch in self.loaders["val"]:
                x = batch[0]
                y = batch[1] if len(batch) > 1 else torch.zeros(x.shape[0])
                output = self.model(x)
                score = torch.mean((output.x_hat - x) ** 2, dim=(1, 2))
                all_scores.extend(score.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        scores_arr = np.array(all_scores)
        labels_arr = np.array(all_labels)
        threshold = float(np.percentile(scores_arr, 95))

        # Only compute full metrics if we have both classes present
        if len(np.unique(labels_arr)) < 2:
            return float(np.mean(scores_arr)), len(self.loaders["val"].dataset), {
                "val_loss": float(np.mean(scores_arr)),
            }

        metrics = compute_metrics(labels_arr, scores_arr, threshold)

        return float(metrics.auroc), len(self.loaders["val"].dataset), metrics.to_dict()
