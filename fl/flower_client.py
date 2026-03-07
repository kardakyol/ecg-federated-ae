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

class DummyAE(BaseAutoencoder):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(12 * 1000, 32),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Linear(32, 12 * 1000)

    def forward(self, x):
        batch_size = x.shape[0]
        z = self.encoder(x.view(batch_size, -1))
        x_hat = self.decoder(z).view(batch_size, 12, 1000)
        return AEOutput(x_hat=x_hat)
    
    def compute_loss(self, x, output, **kwargs):
        loss = torch.nn.functional.mse_loss(output.x_hat, x)
        return (loss, )
    
class ECGClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, model_type: str = "vanilla"):
        self.client_id = client_id

        # Uncomment this line for using actual model once ready
        #self.model = get_model(model_type)
        self.model = DummyAE()

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
        for epoch in range(epochs):
            for batch in self.loaders["train"]:
                x = batch[0]
                optimizer.zero_grad()
                loss, *_ = self.model.compute_loss(x, self.model(x))
                loss.backward()
                optimizer.step()
        
        return self.get_parameters(config={}), len(self.loaders["train"].dataset), {}
    
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
        
        threshold = np.percentile(all_scores, 95)
        metrics = compute_metrics(np.array(all_labels), np.array(all_scores), threshold)
        
        return float(metrics.auroc), len(self.loaders["val"].dataset), metrics.to_dict()