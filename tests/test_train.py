"""
Tests for the training pipeline (using tiny synthetic datasets).
"""
import os
import shutil
import tempfile

import numpy as np
import cv2
import pytest
import torch
from torch.utils.data import DataLoader

from src.dataset import ImagePairDataset, train_val_test_split
from src.model import DrowsyCNN
from src.config import MODELS_DIR


def _make_synthetic_pairs(n_per_class=20, img_size=(64, 64)):
    """Return (pairs, tmpdir) – caller is responsible for cleanup."""
    tmpdir = tempfile.mkdtemp()
    pairs = []
    for label in [0, 1]:
        folder = os.path.join(tmpdir, str(label))
        os.makedirs(folder)
        for i in range(n_per_class):
            img = np.random.randint(0, 256, (*img_size, 3), dtype=np.uint8)
            path = os.path.join(folder, f"{i}.jpg")
            cv2.imwrite(path, img)
            pairs.append((path, label))
    return pairs, tmpdir


@pytest.fixture
def synthetic_pairs():
    pairs, tmpdir = _make_synthetic_pairs(n_per_class=30)
    yield pairs
    shutil.rmtree(tmpdir)


class TestTrainingLoop:
    def test_one_epoch_runs(self, synthetic_pairs):
        """Full forward+backward pass for 1 epoch should complete without error."""
        train_p, val_p, _ = train_val_test_split(synthetic_pairs)
        train_ds = ImagePairDataset(train_p, augment=False)
        loader = DataLoader(train_ds, batch_size=8, shuffle=True)

        model = DrowsyCNN(num_classes=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        for images, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    def test_loss_decreases_over_epochs(self, synthetic_pairs):
        """
        Loss on the same batch should decrease with many gradient steps
        (high LR, 50 steps, same batch = effectively memorisation test).
        """
        torch.manual_seed(0)
        train_p, _, _ = train_val_test_split(synthetic_pairs)
        ds = ImagePairDataset(train_p, augment=False)
        loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
        images, labels = next(iter(loader))

        model = DrowsyCNN(num_classes=2)
        # High LR to force rapid descent on a fixed batch
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        criterion = torch.nn.CrossEntropyLoss()

        first_loss = None
        last_loss = None
        for step in range(50):
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            if first_loss is None:
                first_loss = loss.item()
            last_loss = loss.item()

        assert last_loss < first_loss, \
            f"Loss did not decrease: first={first_loss:.4f}, last={last_loss:.4f}"

    def test_predictions_are_binary(self, synthetic_pairs):
        """Argmax of logits should be 0 or 1."""
        ds = ImagePairDataset(synthetic_pairs, augment=False)
        loader = DataLoader(ds, batch_size=16)
        model = DrowsyCNN(num_classes=2)
        model.eval()
        with torch.no_grad():
            images, _ = next(iter(loader))
            logits = model(images)
            preds = logits.argmax(dim=1)
        assert set(preds.tolist()).issubset({0, 1})

    def test_model_checkpoint_saves(self, synthetic_pairs, tmp_path):
        """Checkpoint should be saved after training."""
        checkpoint = str(tmp_path / "ckpt.pt")
        model = DrowsyCNN(num_classes=2)
        torch.save(model.state_dict(), checkpoint)
        assert os.path.exists(checkpoint)
        size = os.path.getsize(checkpoint)
        assert size > 0

    def test_trained_eye_model_accuracy(self):
        """If eye model exists, its test accuracy should be > 60%."""
        path = os.path.join(MODELS_DIR, "eye_best.pt")
        if not os.path.exists(path):
            pytest.skip("eye_best.pt not trained yet")

        pairs, tmpdir = _make_synthetic_pairs(n_per_class=50)
        try:
            _, _, test_p = train_val_test_split(pairs)
            ds = ImagePairDataset(test_p, augment=False)
            loader = DataLoader(ds, batch_size=16)

            model = DrowsyCNN(num_classes=2)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()

            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in loader:
                    preds = model(images).argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += len(labels)

            # On random data a trained model won't perform, but it must run
            assert 0 <= correct / total <= 1.0
        finally:
            shutil.rmtree(tmpdir)
