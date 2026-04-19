"""
CNN architecture for drowsiness-related binary classification.

One model class is shared for three tasks:
  - Eye state    (open=0, closed=1)
  - Mouth state  (no_yawn=0, yawn=1)
  - Drowsy state (alert=0, drowsy=1)

Architecture: lightweight custom CNN suited for 64×64 grayscale crops.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DrowsyCNN(nn.Module):
    """
    5-layer CNN for binary classification of 64×64 grayscale images.

    Conv block structure: Conv → BN → ReLU → MaxPool
    Classifier:           Flatten → FC(256) → Dropout → FC(2)
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1  64 → 32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2  32 → 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3  16 → 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4  8 → 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # 256 × 4 × 4 = 4096
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def predict_proba(self, x):
        """Return class probabilities (softmax of logits)."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
