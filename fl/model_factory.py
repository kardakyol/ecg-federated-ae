import torch 
import torch.nn as nn
from models.base import BaseAutoencoder, AEOutput
#from models.vanilla_ae import VanillaAE
#from models.conv_ae import ConvAE
#from models.vae import VAE

class DummyAE(BaseAutoencoder):
    """Minimal AE for FL pipeline testing. NOT for real experiments."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(12 * 1000, 32),
            nn.ReLU()
        )
        self.decoder = nn.Linear(32, 12 * 1000)

    def forward(self, x):
        batch_size = x.shape[0]
        z = self.encoder(x.view(batch_size, -1))
        x_hat = self.decoder(z).view(batch_size, 12, 1000)
        return AEOutput(x_hat=x_hat)

    def compute_loss(self, x, output, **kwargs):
        loss = nn.functional.mse_loss(output.x_hat, x)
        return (loss,)

def get_model(model_name: str):
    name = model_name.lower()

    if name == "vanilla":
        #return VanillaAE()
        return DummyAE()
    elif name == "conv":
        #return ConvAE()
        return DummyAE()
    elif name == "vae":
        #return VAE()
        return DummyAE()
    
    raise ValueError(f"Model type '{model_name}' not recognised.")

