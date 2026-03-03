"""
HILAL: Opacus DP-SGD integration.

CRITICAL - Opacus constraints:
    BatchNorm CANNOT be used (no per-sample gradients).
    inplace=True CANNOT be used.
    Shardul and Kaan: use GroupNorm, inplace=False.

Model interface (from models/base.py):
    loss, *_ = model.compute_loss(x, output)
    loss.backward()  # Opacus hooks into gradients here
"""
# TODO: Hilal implementation
