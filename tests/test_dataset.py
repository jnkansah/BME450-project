"""
Tests for dataset discovery and data pipeline.
"""
import os
import tempfile
import shutil

import numpy as np
import cv2
import pytest
import torch

from src.dataset import (discover_eye_dataset, discover_mouth_dataset,
                          discover_drowsy_dataset,
                          train_val_test_split, ImagePairDataset,
                          get_transform, RAW_DIR)
from src.config import IMG_SIZE


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_raw_dir():
    """
    Creates a temp directory mimicking the downloaded dataset structure,
    with 10 synthetic images per class.
    """
    tmpdir = tempfile.mkdtemp()
    structure = {
        "eye": [("Open_Eyes", 0), ("Closed_Eyes", 1)],
        "mouth": [("no_yawn", 0), ("yawn", 1)],
        "drowsy": [("Non Drowsy", 0), ("Drowsy", 1)],
    }
    for task, class_dirs in structure.items():
        for dirname, label in class_dirs:
            folder = os.path.join(tmpdir, task, dirname)
            os.makedirs(folder, exist_ok=True)
            for i in range(10):
                img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(folder, f"img_{i:03d}.jpg"), img)
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def tiny_pairs():
    """Return 20 synthetic (path, label) pairs using real temp files."""
    tmpdir = tempfile.mkdtemp()
    pairs = []
    for label in [0, 1]:
        for i in range(10):
            img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            path = os.path.join(tmpdir, f"img_{label}_{i}.jpg")
            cv2.imwrite(path, img)
            pairs.append((path, label))
    yield pairs
    shutil.rmtree(tmpdir)


# ──────────────────────────────────────────────────────────────────────────────
# Discovery tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDiscovery:
    def test_eye_dataset_finds_images(self, synthetic_raw_dir):
        pairs = discover_eye_dataset(synthetic_raw_dir)
        assert len(pairs) == 20

    def test_eye_dataset_labels(self, synthetic_raw_dir):
        pairs = discover_eye_dataset(synthetic_raw_dir)
        labels = {label for _, label in pairs}
        assert labels == {0, 1}

    def test_mouth_dataset_finds_images(self, synthetic_raw_dir):
        pairs = discover_mouth_dataset(synthetic_raw_dir)
        assert len(pairs) == 20

    def test_mouth_dataset_labels(self, synthetic_raw_dir):
        pairs = discover_mouth_dataset(synthetic_raw_dir)
        labels = {label for _, label in pairs}
        assert labels == {0, 1}

    def test_drowsy_dataset_finds_images(self, synthetic_raw_dir):
        pairs = discover_drowsy_dataset(synthetic_raw_dir)
        assert len(pairs) == 20

    def test_drowsy_dataset_labels(self, synthetic_raw_dir):
        pairs = discover_drowsy_dataset(synthetic_raw_dir)
        labels = {label for _, label in pairs}
        assert labels == {0, 1}

    def test_empty_dir_returns_empty(self):
        tmpdir = tempfile.mkdtemp()
        try:
            assert discover_eye_dataset(tmpdir) == []
            assert discover_mouth_dataset(tmpdir) == []
            assert discover_drowsy_dataset(tmpdir) == []
        finally:
            shutil.rmtree(tmpdir)

    def test_real_raw_dir_has_eye_data(self):
        """Integration: real downloaded data should yield eye images."""
        if not os.path.isdir(RAW_DIR):
            pytest.skip("Raw data not downloaded")
        pairs = discover_eye_dataset(RAW_DIR)
        assert len(pairs) > 0, "Expected eye images in real raw dir"

    def test_real_raw_dir_has_mouth_data(self):
        if not os.path.isdir(RAW_DIR):
            pytest.skip("Raw data not downloaded")
        pairs = discover_mouth_dataset(RAW_DIR)
        assert len(pairs) > 0, "Expected mouth/yawn images in real raw dir"

    def test_real_raw_dir_has_drowsy_data(self):
        if not os.path.isdir(RAW_DIR):
            pytest.skip("Raw data not downloaded")
        pairs = discover_drowsy_dataset(RAW_DIR)
        assert len(pairs) > 0, "Expected drowsy images in real raw dir"


# ──────────────────────────────────────────────────────────────────────────────
# Split tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSplit:
    def test_split_sizes(self, tiny_pairs):
        train, val, test = train_val_test_split(tiny_pairs)
        assert len(train) + len(val) + len(test) == len(tiny_pairs)

    def test_split_proportions(self, tiny_pairs):
        n = len(tiny_pairs)
        train, val, test = train_val_test_split(tiny_pairs, train=0.8, val=0.1)
        assert abs(len(train) / n - 0.8) < 0.05
        assert abs(len(val)   / n - 0.1) < 0.05

    def test_split_no_overlap(self, tiny_pairs):
        train, val, test = train_val_test_split(tiny_pairs)
        train_paths = {p for p, _ in train}
        val_paths   = {p for p, _ in val}
        test_paths  = {p for p, _ in test}
        assert not (train_paths & val_paths)
        assert not (train_paths & test_paths)
        assert not (val_paths & test_paths)

    def test_reproducible_with_same_seed(self, tiny_pairs):
        t1, v1, te1 = train_val_test_split(tiny_pairs, seed=42)
        t2, v2, te2 = train_val_test_split(tiny_pairs, seed=42)
        assert [p for p, _ in t1] == [p for p, _ in t2]


# ──────────────────────────────────────────────────────────────────────────────
# Dataset / DataLoader tests
# ──────────────────────────────────────────────────────────────────────────────

class TestImagePairDataset:
    def test_len(self, tiny_pairs):
        ds = ImagePairDataset(tiny_pairs)
        assert len(ds) == len(tiny_pairs)

    def test_item_shape(self, tiny_pairs):
        ds = ImagePairDataset(tiny_pairs)
        img, label = ds[0]
        assert img.shape == (1, IMG_SIZE[0], IMG_SIZE[1])

    def test_item_dtype(self, tiny_pairs):
        ds = ImagePairDataset(tiny_pairs)
        img, label = ds[0]
        assert img.dtype == torch.float32

    def test_label_type(self, tiny_pairs):
        ds = ImagePairDataset(tiny_pairs)
        _, label = ds[0]
        assert label.dtype == torch.int64

    def test_label_values(self, tiny_pairs):
        ds = ImagePairDataset(tiny_pairs)
        labels = {ds[i][1].item() for i in range(len(ds))}
        assert labels == {0, 1}

    def test_normalized_range(self, tiny_pairs):
        """Pixels after normalize(mean=0.5, std=0.5) should be in [-1, 1]."""
        ds = ImagePairDataset(tiny_pairs)
        img, _ = ds[0]
        assert img.min() >= -1.5 and img.max() <= 1.5

    def test_missing_file_returns_blank(self, tiny_pairs):
        """A pair with a non-existent path should return a zero tensor, not crash."""
        bad_pairs = [("/nonexistent/img.jpg", 0)]
        ds = ImagePairDataset(bad_pairs)
        img, label = ds[0]
        assert img.shape == (1, IMG_SIZE[0], IMG_SIZE[1])

    def test_augmented_dataset_same_shape(self, tiny_pairs):
        ds = ImagePairDataset(tiny_pairs, augment=True)
        img, _ = ds[0]
        assert img.shape == (1, IMG_SIZE[0], IMG_SIZE[1])

    def test_transform_no_augment(self):
        transform = get_transform(augment=False)
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        from PIL import Image
        pil = Image.fromarray(img)
        tensor = transform(img)
        assert tensor.shape == (1, IMG_SIZE[0], IMG_SIZE[1])
