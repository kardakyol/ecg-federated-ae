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
from models.base import BaseAutoencoder, AEOutput
from utils.dataset import create_synthetic_data, create_dataloaders

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
    def __init__(self, client_id: str):
        self.client_id = client_id

        self.model = DummyAE()

        # For Sprint 2 TODO: Replace DummyAE() with Model Factory
        # from fl.model_factory import get_model
        # self.model = get_model(config["model_type"])

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

        # For Sprint 1: local Epoch 1
        # For Sprint 2: TODO: increase it to 5

        for epoch in range(1):
            for batch in self.loaders["train"]:
                x = batch[0]
                optimizer.zero_grad()
                loss, *_ = self.model.compute_loss(x, self.model(x))
                loss.backward()
                optimizer.step()
        
        return self.get_parameters(config={}), len(self.loaders["train"].dataset), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss=0.0
        with torch.no_grad():
            for batch in self.loaders["val"]:
                x = batch[0]
                loss += self.model.compute_loss(x, self.model(x))[0].item()
        
        # For Sprint 2 TODO: Integrate evaluation/metrics.py here for AUROC/AUPRC
        return loss / len(self.loaders["val"]), len(self.loaders["val"].dataset), {}