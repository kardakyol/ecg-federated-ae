"""
GHADAH: Post-Training Quantisation pipeline.

Model interface (from models/base.py):
    output = model(x)          # .x_hat guaranteed
    size = model.model_size_mb()
    params = model.count_parameters()

Metrics:
    from evaluation.metrics import compute_metrics
    from utils.csv_logger import ResultLogger
"""
# TODO: Ghadah implementation
