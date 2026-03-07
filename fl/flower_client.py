"""
RAHEEB: Flower client and server configuration.

Model interface (from models/base.py):
    model.get_parameters() -> List[np.ndarray]
    model.set_parameters(List[np.ndarray])
    model.forward(x) -> AEOutput (with .x_hat)
    model.compute_loss(x, output) -> (total_loss, ...)

Data loading:
    from utils.dataset import load_splits, create_dataloaders
"""
# TODO: Raheeb implementation

import flwr as fl
import torch
import numpy as np 
from models.base import BaseAutoencoder, AEOutput
from utils.dataset import create_synthetic_data, create_dataloaders
from evaluation.metrics import compute_metrics
from fl.model_factory import get_model, DummyAE


class ECGClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, model_type: str = "vanilla"):
        self.client_id = client_id

        # Uncomment this line for using actual model once ready
        self.model = get_model(model_type)

        self.splits = create_synthetic_data(n_train=200, n_val=50, n_test=50)
        # For Sprint 2 TODO: Load Ghouse's Dirichlet shards
        # from utils.dataset import load_splits
        # self.splits = load_splits(f"data/ptb-xl/client_{client_id}")
        self.loaders = create_dataloaders(self.splits, batch_size=32)

    def get_parameters(self, config):
        return self.model.get_parameters()

    def set_parameters(self, parameters):
        self.model.set_parameters(parameters) 

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        epochs = config.get("local_epochs", 1)
        total_loss = 0.0
        num_batches = 0

        for epoch in range(epochs):
            for batch in self.loaders["train"]:
                x = batch[0]
                optimizer.zero_grad()
                loss, *_ = self.model.compute_loss(x, self.model(x))
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
                score = torch.mean((output.x_hat - x)**2, dim=(1,2))
                all_scores.extend(score.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        scores_arr = np.array(all_scores)
        labels_arr = np.array(all_labels)
        threshold = np.percentile(scores_arr, 95)
        
        if len(np.unique(labels_arr)) < 2:
            return float(np.mean(scores_arr)), len(self.loaders["val"].dataset), {
                "val_loss": float(np.mean(scores_arr)),
            }

        metrics = compute_metrics(np.array(all_labels), scores_arr, threshold)
        
        return float(metrics.auroc), len(self.loaders["val"].dataset), metrics.to_dict()