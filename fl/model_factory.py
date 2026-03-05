import torch 
from fl.flower_client import DummyAE
#from models.vanilla_ae import VanillaAE
#from models.conv_ae import ConvAE
#from models.vae import VAE

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

