import torch 
import torch.nn as nn
from models.base import BaseAutoencoder, AEOutput
from models.vanilla_ae import VanillaAE
from models.conv_ae import ConvAE
from models.vae import VAE

def get_model(model_name: str) -> BaseAutoencoder:
    """
    Factory function to initialize models. 
    Centralized point for Sprint 3 ablation study architecture swaps.
    """
    name = model_name.lower().strip()

    if name == "vanilla":
        return VanillaAE()
    elif name == "conv":
        return ConvAE()
    elif name == "vae":
        return VAE()
    
    raise ValueError(f"Model type '{model_name}' not recognised.")