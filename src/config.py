"""Central configuration for the drowsiness detection system."""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Image settings
IMG_SIZE = (64, 64)        # resize all crops to this
CHANNELS = 1               # grayscale

# EAR / MAR thresholds (rule-based fallback + test assertions)
EAR_THRESH = 0.25          # below → eye closed
MAR_THRESH = 0.60          # above → yawning
HEAD_PITCH_THRESH = 20.0   # degrees nose-down → nodding

# Drowsiness logic (frame-based)
CONSEC_FRAMES_EAR = 20     # EAR below thresh for N frames → drowsy
CONSEC_FRAMES_MAR = 15     # MAR above thresh for N frames → yawning

# CNN training
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
# TEST_SPLIT = 0.10 (remainder)

# Random seed
SEED = 42
