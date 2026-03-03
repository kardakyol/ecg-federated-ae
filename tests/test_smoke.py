"""
SMOKE TEST - everyone runs this after cloning:
    pip install -e ".[dev]"
    pytest tests/test_smoke.py -v

If all pass, your environment is correctly set up.
"""
import numpy as np
import pytest


class TestSharedUtils:
    def test_seeds_constant(self):
        from utils.reproducibility import SEEDS
        assert SEEDS == [42, 123, 456]

    def test_synthetic_data_shape(self):
        from utils.dataset import create_synthetic_data
        splits = create_synthetic_data(n_train=50, n_val=20, n_test=20)
        assert splits["train"].signals.shape[1] == 12
        assert splits["train"].signals.shape[2] == 1000

    def test_train_loader_normal_only(self):
        from utils.dataset import create_synthetic_data, create_dataloaders
        splits = create_synthetic_data(n_train=100, n_val=30, n_test=30)
        loaders = create_dataloaders(splits, batch_size=16, num_workers=0)
        for x, y in loaders["train"]:
            assert (y == 0).all(), "Train must be normal-only"
            break

    def test_csv_logger(self, tmp_path):
        from utils.csv_logger import ResultLogger
        lg = ResultLogger(tmp_path / "test.csv")
        lg.log(model="vae", seed=42, auroc=0.92)
        import csv
        with open(tmp_path / "test.csv") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["model"] == "vae"


class TestSharedMetrics:
    def test_perfect_separation(self):
        from evaluation.metrics import compute_metrics
        result = compute_metrics(
            np.array([0,0,0,1,1,1]), np.array([0.1,0.2,0.3,0.7,0.8,0.9]), 0.5)
        assert result.auroc == 1.0
        assert result.sensitivity == 1.0

    def test_aggregate(self):
        from evaluation.metrics import MetricsResult, aggregate_seeds
        agg = aggregate_seeds([
            MetricsResult(auroc=0.90, f1=0.85),
            MetricsResult(auroc=0.92, f1=0.87),
        ])
        assert abs(agg["auroc"]["mean"] - 0.91) < 0.001


class TestBaseAutoencoder:
    def test_cannot_instantiate(self):
        from models.base import BaseAutoencoder
        with pytest.raises(TypeError):
            BaseAutoencoder()

    def test_concrete_model(self):
        import torch
        from models.base import BaseAutoencoder, AEOutput

        class Dummy(BaseAutoencoder):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(12000, 12000)
            def forward(self, x):
                return AEOutput(x_hat=self.fc(x.view(x.size(0),-1)).view_as(x))
            def compute_loss(self, x, output, **kw):
                return (torch.nn.functional.mse_loss(output.x_hat, x),)

        m = Dummy()
        x = torch.randn(2, 12, 1000)
        out = m(x)
        assert out.x_hat.shape == (2, 12, 1000)
        loss, = m.compute_loss(x, out)
        assert loss.item() > 0
        params = m.get_parameters()
        assert isinstance(params[0], np.ndarray)
        m.set_parameters(params)
        assert m.model_size_mb() > 0
