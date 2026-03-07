from dataclasses import dataclass, field
from typing import List


@dataclass
class AEConfig:
    """Vanilla AE and Conv AE training Hyperparameters."""

    bottleneck: int = 32                      
    n_leads: int = 12                       
    seq_len: int = 1000                       

    batch_size: int = 128                     
    epochs: int = 50                         
    lr: float = 1e-3                            
    weight_decay: float = 1e-5                  
    scheduler_patience: int = 5                 
    scheduler_factor: float = 0.5              

    data_dir: str = "data/ptb-xl"             

    checkpoint_dir: str = "checkpoints"         
    output_dir: str = "outputs"                
    figure_dir: str = "outputs/figures"         

    bottleneck_ablation: List[int] = field(default_factory=lambda: [16, 32, 64, 128])