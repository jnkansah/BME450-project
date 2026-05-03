"""
Train CNN models for eye-state, mouth-state, and drowsiness classification.

Usage:
    python -m src.train --task eye   --epochs 20
    python -m src.train --task mouth --epochs 20
    python -m src.train --task drowsy --epochs 20
    python -m src.train --task all   --epochs 20   (trains all three)
"""
import argparse
import os
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from src.config import (BATCH_SIZE, EPOCHS, LR, MODELS_DIR, SEED,
                         RAW_DIR)
from src.dataset import (discover_eye_dataset, discover_mouth_dataset,
                          discover_drowsy_dataset,
                          train_val_test_split, ImagePairDataset)
from src.model import DrowsyCNN

torch.manual_seed(SEED)
os.makedirs(MODELS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _count_classes(pairs):
    counts = {}
    for _, label in pairs:
        counts[label] = counts.get(label, 0) + 1
    return counts


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return avg_loss, accuracy, all_preds, all_labels


def train_one_task(task: str, epochs: int, lr: float = LR, batch_size: int = BATCH_SIZE, raw_dir: str = RAW_DIR, max_samples: int = 0):
    """Train a DrowsyCNN for the given task. Returns final test accuracy."""
    print(f"\n{'='*60}")
    print(f"  Training task: {task.upper()} (LR={lr}, Batch={batch_size}, MaxSamples={max_samples if max_samples > 0 else 'All'})")
    print(f"{'='*60}")

    # --- Discover data ---
    if task == "eye":
        pairs = discover_eye_dataset(raw_dir, max_per_class=max_samples)
        class_names = ["open", "closed"]
    elif task == "mouth":
        pairs = discover_mouth_dataset(raw_dir, max_per_class=max_samples)
        class_names = ["no_yawn", "yawn"]
    elif task == "drowsy":
        pairs = discover_drowsy_dataset(raw_dir, max_per_class=max_samples)
        class_names = ["alert", "drowsy"]
    else:
        raise ValueError(f"Unknown task: {task}")

    if not pairs:
        print(f"  [WARNING] No images found for task '{task}'. Skipping.")
        return None

    class_counts = _count_classes(pairs)
    print(f"  Total images: {len(pairs)}")
    for k, v in sorted(class_counts.items()):
        print(f"    class {k} ({class_names[k]}): {v}")

    train_p, val_p, test_p = train_val_test_split(pairs)
    print(f"  Split → train: {len(train_p)}, val: {len(val_p)}, test: {len(test_p)}")

    train_ds = ImagePairDataset(train_p, augment=True)
    val_ds   = ImagePairDataset(val_p,   augment=False)
    test_ds  = ImagePairDataset(test_p,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                               num_workers=0)

    device = torch.device("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available()
                           else "cpu")
    print(f"  Device: {device}")

    model = DrowsyCNN(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_path = os.path.join(MODELS_DIR, f"{task}_best.pt")
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_acc={val_acc:.4f} | "
              f"{elapsed:.1f}s")

    # --- Test evaluation ---
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, device)
    print(f"\n  Test accuracy: {test_acc:.4f}")
    
    # Only print report if both classes are present in the test set
    unique_labels = sorted(list(set(labels)))
    if len(unique_labels) == len(class_names):
        print(classification_report(labels, preds, target_names=class_names,
                                     zero_division=0))
    else:
        print(f"  [INFO] Skipping classification report: test set only contains labels {unique_labels}")

    # Save history
    hist_path = os.path.join(MODELS_DIR, f"{task}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"  Model saved → {best_path}")
    return test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all",
                        choices=["eye", "mouth", "drowsy", "all"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--raw_dir", default=RAW_DIR)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max images per class to use for training (0 = all)")
    args = parser.parse_args()

    tasks = ["eye", "mouth", "drowsy"] if args.task == "all" else [args.task]
    results = {}
    for task in tasks:
        acc = train_one_task(task, args.epochs, LR, BATCH_SIZE, args.raw_dir, args.max_samples)
        if acc is not None:
            results[task] = acc

    print("\n\n=== FINAL RESULTS ===")
    for task, acc in results.items():
        print(f"  {task:10s}: {acc:.4f}")


if __name__ == "__main__":
    main()
