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
