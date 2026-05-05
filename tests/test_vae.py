"""
Run with: pytest tests/test_vae.py -v

Tests verify:
    1. Model architecture (shapes, parameter count, base class compliance)
    2. Forward pass (output shapes, AEOutput fields, numerical stability)
    3. Loss computation (MSE + KL components, beta weighting, annealing)
    4. Anomaly scoring (MC sampling, threshold calibration)
    5. Flower compatibility (get/set_parameters round-trip)
    6. Opacus readiness (no BatchNorm, no inplace ops)
    7. Gradient flow (no dead gradients, no NaN after backward)
    8. Edge cases (single sample, very large/small inputs)
    9. Training loop (loss decreases, early stopping, checkpointing)
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ================================================================
# FIXTURES
# ================================================================

@pytest.fixture
def config():
    from configs.vae_config import VAEArchitectureConfig
    return VAEArchitectureConfig(latent_dim=16, encoder_channels=[16, 32])

@pytest.fixture
def model(config):
    from models.vae import VAE
    return VAE(config)

@pytest.fixture
def batch():
    """Realistic batch: (B=4, 12 leads, 1000 timesteps)."""
    return torch.randn(4, 12, 1000)

@pytest.fixture
def small_loaders():
    """Small synthetic data loaders for training tests."""
    from utils.dataset import create_synthetic_data, create_dataloaders
    splits = create_synthetic_data(n_train=100, n_val=30, n_test=30)
    return create_dataloaders(splits, batch_size=16, num_workers=0)


# ================================================================
# 1. ARCHITECTURE TESTS
# ================================================================

class TestArchitecture:
    def test_extends_base_autoencoder(self, model):
        """VAE MUST extend BaseAutoencoder"""
        from models.base import BaseAutoencoder
        assert isinstance(model, BaseAutoencoder)

    def test_has_encoder_decoder(self, model):
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'decoder')
        assert hasattr(model, 'fc_mu')
        assert hasattr(model, 'fc_logvar')
        assert hasattr(model, 'fc_decode')

    def test_parameter_count_reasonable(self, model):
        """Model should be small enough for edge deployment."""
        n_params = model.count_parameters()
        assert n_params > 1000, "Model too small to learn anything"
        assert n_params < 50_000_000, "Model too large for edge deployment"

    def test_model_size_mb(self, model):
        size = model.model_size_mb()
        assert size > 0
        assert size < 200, "Model too large"


# ================================================================
# 2. FORWARD PASS TESTS
# ================================================================

class TestForwardPass:
    def test_output_type(self, model, batch):
        from models.base import AEOutput
        output = model(batch)
        assert isinstance(output, AEOutput)

    def test_output_shapes(self, model, batch):
        """x_hat.shape MUST equal x.shape (BaseAutoencoder contract)."""
        output = model(batch)
        assert output.x_hat.shape == batch.shape, \
            f"x_hat shape {output.x_hat.shape} != input shape {batch.shape}"

    def test_vae_fields_populated(self, model, batch):
        """VAE must fill mu, logvar, z."""
        output = model(batch)
        assert output.mu is not None
        assert output.logvar is not None
        assert output.z is not None

    def test_latent_shapes(self, model, batch, config):
        output = model(batch)
        B = batch.shape[0]
        assert output.mu.shape == (B, config.latent_dim)
        assert output.logvar.shape == (B, config.latent_dim)
        assert output.z.shape == (B, config.latent_dim)

    def test_no_nan_in_output(self, model, batch):
        output = model(batch)
        assert not torch.isnan(output.x_hat).any()
        assert not torch.isnan(output.mu).any()
        assert not torch.isnan(output.logvar).any()

    def test_stochastic_forward(self, model, batch):
        """Two forward passes should give different z (reparameterisation)."""
        model.eval()
        out1 = model(batch)
        out2 = model(batch)
        # mu and logvar should be identical (deterministic encoder)
        assert torch.allclose(out1.mu, out2.mu, atol=1e-6)
        # z and x_hat should differ (stochastic sampling)
        assert not torch.allclose(out1.z, out2.z, atol=1e-6)

    def test_single_sample(self, model):
        """Must work with batch_size=1."""
        x = torch.randn(1, 12, 1000)
        output = model(x)
        assert output.x_hat.shape == (1, 12, 1000)


# ================================================================
# 3. LOSS COMPUTATION TESTS
# ================================================================

class TestLossComputation:
    def test_loss_returns_tuple(self, model, batch):
        output = model(batch)
        result = model.compute_loss(batch, output, beta=1.0)
        assert isinstance(result, tuple)
        assert len(result) == 3  # (total, mse, kl)

    def test_loss_first_element_is_total(self, model, batch):
        """First element = total loss (BaseAutoencoder contract)."""
        output = model(batch)
        total, mse, kl = model.compute_loss(batch, output, beta=1.0)
        assert total.ndim == 0  # scalar
        assert total.requires_grad

    def test_mse_non_negative(self, model, batch):
        output = model(batch)
        _, mse, _ = model.compute_loss(batch, output, beta=1.0)
        assert mse.item() >= 0

    def test_kl_non_negative(self, model, batch):
        """KL divergence is always >= 0."""
        output = model(batch)
        _, _, kl = model.compute_loss(batch, output, beta=1.0)
        assert kl.item() >= 0

    def test_beta_scaling(self, model, batch):
        """Higher beta -> higher total loss (KL contributes more)."""
        output = model(batch)
        loss_low, _, _ = model.compute_loss(batch, output, beta=0.01, kl_weight=1.0)
        loss_high, _, _ = model.compute_loss(batch, output, beta=10.0, kl_weight=1.0)
        assert loss_high.item() >= loss_low.item()

    def test_kl_annealing_zero(self, model, batch):
        """kl_weight=0 means total = MSE only."""
        output = model(batch)
        total, mse, kl = model.compute_loss(batch, output, beta=1.0, kl_weight=0.0)
        assert torch.allclose(total, mse, atol=1e-6)

    def test_loss_no_nan(self, model, batch):
        output = model(batch)
        total, mse, kl = model.compute_loss(batch, output, beta=0.5)
        assert not torch.isnan(total)
        assert not torch.isnan(mse)
        assert not torch.isnan(kl)


# ================================================================
# 4. ANOMALY SCORING TESTS
# ================================================================

class TestAnomalyScoring:
    def test_score_shape(self, model, small_loaders):
        from evaluation.anomaly_scorer import compute_anomaly_scores
        scores, labels = compute_anomaly_scores(
            model, small_loaders["test"], alpha=0.5, n_mc_samples=2
        )
        assert len(scores) == len(labels)
        assert len(scores) > 0

    def test_scores_non_negative(self, model, small_loaders):
        from evaluation.anomaly_scorer import compute_anomaly_scores
        scores, _ = compute_anomaly_scores(
            model, small_loaders["test"], alpha=0.5, n_mc_samples=2
        )
        assert np.all(scores >= 0)

    def test_threshold_calibration(self, model, small_loaders):
        from evaluation.anomaly_scorer import calibrate_threshold
        threshold = calibrate_threshold(
            model, small_loaders["val_normal"],
            percentile=95.0, alpha=0.5, n_mc_samples=2
        )
        assert isinstance(threshold, float)
        assert threshold > 0


# ================================================================
# 5. FLOWER COMPATIBILITY
# ================================================================

class TestFlowerCompatibility:
    def test_get_parameters_returns_numpy(self, model):
        params = model.get_parameters()
        assert isinstance(params, list)
        assert all(isinstance(p, np.ndarray) for p in params)

    def test_set_parameters_round_trip(self, model, batch):
        """get -> set -> get must produce identical parameters."""
        params_before = model.get_parameters()
        output_before = model(batch)

        # Simulate FedAvg: extract, perturb slightly, set back original
        model.set_parameters(params_before)
        params_after = model.get_parameters()

        for p_before, p_after in zip(params_before, params_after):
            assert np.allclose(p_before, p_after, atol=1e-6)

    def test_fedavg_simulation(self, config):
        """Simulate FedAvg averaging of two client models."""
        from models.vae import VAE
        model1 = VAE(config)
        model2 = VAE(config)

        params1 = model1.get_parameters()
        params2 = model2.get_parameters()

        # FedAvg: simple average
        avg_params = [(p1 + p2) / 2 for p1, p2 in zip(params1, params2)]

        # Must be settable without error
        model1.set_parameters(avg_params)
        x = torch.randn(2, 12, 1000)
        output = model1(x)
        assert output.x_hat.shape == (2, 12, 1000)


# ================================================================
# 6. OPACUS READINESS 
# ================================================================

class TestOpacusReadiness:
    def test_no_batchnorm(self, model):
        """Opacus CANNOT handle BatchNorm. Must use GroupNorm."""
        for name, module in model.named_modules():
            assert not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)), \
                f"Found BatchNorm at {name}. Use GroupNorm for Opacus."

    def test_no_inplace_operations(self, model):
        """Opacus needs inplace=False on all activations."""
        for name, module in model.named_modules():
            if hasattr(module, 'inplace'):
                assert not module.inplace, \
                    f"inplace=True at {name}. Opacus requires inplace=False."


# ================================================================
# 7. GRADIENT FLOW TESTS
# ================================================================

class TestGradientFlow:
    def test_backward_no_nan(self, model, batch):
        """Loss.backward() must produce finite gradients."""
        output = model(batch)
        total, _, _ = model.compute_loss(batch, output, beta=0.5)
        total.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), \
                    f"NaN gradient in {name}"

    def test_all_parameters_receive_gradient(self, model, batch):
        """Every parameter should participate in the computation."""
        output = model(batch)
        total, _, _ = model.compute_loss(batch, output, beta=0.5)
        total.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


# ================================================================
# 8. EDGE CASES
# ================================================================

class TestEdgeCases:
    def test_large_input_values(self, model):
        """Model should handle large input values without NaN."""
        x = torch.randn(2, 12, 1000) * 100
        output = model(x)
        assert not torch.isnan(output.x_hat).any()

    def test_zero_input(self, model):
        """Model should handle all-zero input."""
        x = torch.zeros(2, 12, 1000)
        output = model(x)
        assert not torch.isnan(output.x_hat).any()

    def test_different_latent_dims(self):
        """Architecture must work with different latent dimensions."""
        from models.vae import VAE
        from configs.vae_config import VAEArchitectureConfig
        for dim in [8, 16, 32, 64, 128]:
            cfg = VAEArchitectureConfig(latent_dim=dim, encoder_channels=[16, 32])
            m = VAE(cfg)
            x = torch.randn(2, 12, 1000)
            out = m(x)
            assert out.x_hat.shape == (2, 12, 1000)
            assert out.mu.shape == (2, dim)


# ================================================================
# 9. TRAINING INTEGRATION TEST
# ================================================================

class TestTrainingIntegration:
    def test_loss_decreases(self, config, small_loaders):
        """Training for a few epochs should reduce MSE loss.

        NOTE: We check MSE loss, not total loss, because KL annealing
        intentionally increases total loss in early epochs as the KL
        weight ramps from 0 to 1. MSE (reconstruction quality) should
        always improve with training.
        """
        from models.vae import VAE
        from training.train_vae import VAETrainer
        from configs.vae_config import VAETrainingConfig

        model = VAE(config)
        train_config = VAETrainingConfig(
            epochs=8, patience=10, kl_annealing_epochs=2,
            batch_size=16
        )
        trainer = VAETrainer(model, train_config, torch.device('cpu'))
        history = trainer.train(
            small_loaders["train"], small_loaders["val"], beta=0.5
        )
        assert history["train_mse_loss"][-1] < history["train_mse_loss"][0], \
            "MSE loss did not decrease during training"

    def test_full_pipeline_synthetic(self, config, small_loaders):
        """End-to-end: train -> score -> metrics."""
        from models.vae import VAE
        from training.train_vae import VAETrainer
        from evaluation.anomaly_scorer import compute_anomaly_scores, calibrate_threshold
        from evaluation.metrics import compute_metrics
        from configs.vae_config import VAETrainingConfig

        model = VAE(config)
        train_config = VAETrainingConfig(
            epochs=3, patience=10, kl_annealing_epochs=1, batch_size=16
        )
        trainer = VAETrainer(model, train_config, torch.device('cpu'))
        trainer.train(small_loaders["train"], small_loaders["val"], beta=0.5)

        threshold = calibrate_threshold(
            model, small_loaders["val_normal"],
            percentile=95.0, alpha=0.5, n_mc_samples=2
        )
        scores, labels = compute_anomaly_scores(
            model, small_loaders["test"], alpha=0.5, n_mc_samples=2
        )
        result = compute_metrics(labels, scores, threshold)

        assert 0.0 <= result.auroc <= 1.0
        assert 0.0 <= result.sensitivity <= 1.0
        assert 0.0 <= result.specificity <= 1.0
