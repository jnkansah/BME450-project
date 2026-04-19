"""
Tests for DrowsyCNN architecture correctness.
"""
import os
import tempfile

import pytest
import torch
import torch.nn as nn

from src.model import DrowsyCNN
from src.config import IMG_SIZE, MODELS_DIR


@pytest.fixture
def model():
    return DrowsyCNN(num_classes=2)


@pytest.fixture
def dummy_batch():
    return torch.randn(4, 1, IMG_SIZE[0], IMG_SIZE[1])


class TestDrowsyCNN:
    def test_output_shape(self, model, dummy_batch):
        out = model(dummy_batch)
        assert out.shape == (4, 2)

    def test_output_dtype(self, model, dummy_batch):
        out = model(dummy_batch)
        assert out.dtype == torch.float32

    def test_single_image(self, model):
        img = torch.randn(1, 1, IMG_SIZE[0], IMG_SIZE[1])
        out = model(img)
        assert out.shape == (1, 2)

    def test_predict_proba_sums_to_one(self, model, dummy_batch):
        probs = model.predict_proba(dummy_batch)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_predict_proba_non_negative(self, model, dummy_batch):
        probs = model.predict_proba(dummy_batch)
        assert (probs >= 0).all()

    def test_predict_proba_no_grad(self, model, dummy_batch):
        probs = model.predict_proba(dummy_batch)
        assert not probs.requires_grad

    def test_parameter_count(self, model):
        """Model should have a reasonable number of trainable parameters."""
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert 100_000 < n < 10_000_000, f"Unexpected param count: {n}"

    def test_gradient_flows(self, model, dummy_batch):
        out = model(dummy_batch)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No grad for {name}"

    def test_batchnorm_train_vs_eval(self, model, dummy_batch):
        """BatchNorm should produce different outputs in train vs eval mode."""
        model.train()
        out_train = model(dummy_batch).detach()
        model.eval()
        with torch.no_grad():
            out_eval = model(dummy_batch)
        # Not identical (BN running stats vs batch stats differ at init)
        assert not torch.allclose(out_train, out_eval)

    def test_dropout_train_vs_eval(self, model, dummy_batch):
        """Outputs under dropout (train) should vary across runs."""
        model.train()
        out1 = model(dummy_batch).detach()
        out2 = model(dummy_batch).detach()
        # With p=0.5 dropout, outputs should differ (extremely rarely same)
        assert not torch.allclose(out1, out2)

    def test_save_and_load(self, model, dummy_batch, tmp_path):
        path = str(tmp_path / "test_model.pt")
        torch.save(model.state_dict(), path)
        model2 = DrowsyCNN(num_classes=2)
        model2.load_state_dict(torch.load(path, map_location="cpu"))
        model2.eval()
        model.eval()
        with torch.no_grad():
            out1 = model(dummy_batch)
            out2 = model2(dummy_batch)
        assert torch.allclose(out1, out2)

    def test_features_module_exists(self, model):
        assert hasattr(model, "features")
        assert isinstance(model.features, nn.Sequential)

    def test_classifier_module_exists(self, model):
        assert hasattr(model, "classifier")
        assert isinstance(model.classifier, nn.Sequential)

    def test_wrong_input_channels_raises(self, model):
        """Model expects 1-channel input; 3-channel should error."""
        rgb = torch.randn(1, 3, IMG_SIZE[0], IMG_SIZE[1])
        with pytest.raises(Exception):
            model(rgb)

    def test_wrong_spatial_size_raises(self, model):
        """Wrong spatial size should cause a dimension mismatch."""
        big = torch.randn(1, 1, 128, 128)
        with pytest.raises(Exception):
            model(big)

    def test_saved_models_loadable(self):
        """If trained models exist in models/, they should be loadable."""
        for task in ("eye", "mouth", "drowsy"):
            path = os.path.join(MODELS_DIR, f"{task}_best.pt")
            if not os.path.exists(path):
                pytest.skip(f"Model {task}_best.pt not yet trained")
            m = DrowsyCNN(num_classes=2)
            m.load_state_dict(torch.load(path, map_location="cpu"))
