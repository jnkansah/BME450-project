"""
Dataset discovery and PyTorch Dataset classes.

Scans the raw data directory to build a unified eye-state and
mouth-state dataset from all downloaded sources.

Label convention
----------------
Eye CNN  : 0 = open, 1 = closed
Mouth CNN: 0 = no_yawn, 1 = yawn
Drowsy   : 0 = alert, 1 = drowsy  (DDD dataset)
"""
import os
import glob
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import IMG_SIZE, SEED, RAW_DIR

# ──────────────────────────────────────────────────────────────────────────────
# Folder-name → label mappings  (lower-cased folder name patterns)
# ──────────────────────────────────────────────────────────────────────────────
EYE_OPEN_KEYWORDS   = {"open", "open_eye", "openeye", "open_eyes", "opened"}
EYE_CLOSED_KEYWORDS = {"closed", "close", "closed_eye", "closedeye",
                        "closed_eyes", "close_eye", "closeeye"}
YAWN_KEYWORDS       = {"yawn", "yawning"}
NO_YAWN_KEYWORDS    = {"no_yawn", "noyawn", "notayawn", "not_yawn",
                        "no yawn", "no-yawn"}
DROWSY_KEYWORDS     = {"drowsy", "drowsiness", "fatigue"}
ALERT_KEYWORDS      = {"non_drowsy", "nondrowsy", "alert", "awake",
                        "non-drowsy", "active", "nodrowsy", "non drowsy"}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _label_from_folder(name: str, mapping_pairs):
    """Return label int if folder name matches any keyword set, else -1."""
    n = name.lower().strip()
    for keywords, label in mapping_pairs:
        # exact match first, then substring containment
        if n in keywords:
            return label
        if any(k in n for k in keywords):
            return label
    return -1


def _collect_images(root: str, mapping_pairs, min_per_class=0):
    """
    Walk root recursively.  For each leaf folder whose name matches a keyword,
    collect (path, label) pairs.

    Returns list of (path, label).
    """
    buckets = defaultdict(list)
    for dirpath, dirnames, filenames in os.walk(root):
        folder = os.path.basename(dirpath)
        label = _label_from_folder(folder, mapping_pairs)
        if label < 0:
            continue
        for f in filenames:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                buckets[label].append(os.path.join(dirpath, f))

    # balance classes if requested
    if min_per_class > 0:
        n = min(len(v) for v in buckets.values() if v)
        for k in buckets:
            random.shuffle(buckets[k])
            buckets[k] = buckets[k][:n]

    pairs = []
    for label, paths in buckets.items():
        pairs.extend((p, label) for p in paths)
    return pairs


def discover_eye_dataset(raw_dir: str = RAW_DIR):
    mapping = [
        (EYE_OPEN_KEYWORDS,   0),
        (EYE_CLOSED_KEYWORDS, 1),
    ]
    return _collect_images(raw_dir, mapping)


def discover_mouth_dataset(raw_dir: str = RAW_DIR):
    mapping = [
        (NO_YAWN_KEYWORDS, 0),
        (YAWN_KEYWORDS,    1),
    ]
    return _collect_images(raw_dir, mapping)


def discover_drowsy_dataset(raw_dir: str = RAW_DIR):
    mapping = [
        (ALERT_KEYWORDS,  0),
        (DROWSY_KEYWORDS, 1),
    ]
    return _collect_images(raw_dir, mapping)


def train_val_test_split(pairs, train=0.80, val=0.10, seed=SEED):
    rng = random.Random(seed)
    data = list(pairs)
    rng.shuffle(data)
    n = len(data)
    t = int(n * train)
    v = int(n * (train + val))
    return data[:t], data[t:v], data[v:]


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────

def get_transform(augment: bool):
    ops = [
        transforms.ToPILImage(),
        transforms.Resize(IMG_SIZE),
        transforms.Grayscale(num_output_channels=1),
    ]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
    return transforms.Compose(ops)


class ImagePairDataset(Dataset):
    def __init__(self, pairs, augment: bool = False):
        self.pairs = pairs
        self.transform = get_transform(augment)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        img = cv2.imread(path)
        if img is None:
            # Return a blank image on corrupt file
            img = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img)
        return tensor, torch.tensor(label, dtype=torch.long)
