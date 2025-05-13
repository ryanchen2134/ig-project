"""
Configuration settings for the project
"""
import os
from pathlib import Path
import torch

# Project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")

# Ensure directories exist
for directory in [DATA_DIR, MODEL_DIR, CHECKPOINT_DIR, LOG_DIR, UPLOAD_FOLDER]:
    os.makedirs(directory, exist_ok=True)

# Model settings
DEFAULT_EMBEDDING_DIM = 128
DEFAULT_CLIP_MODEL = "ViT-B/32"  # Options: "ViT-B/32", "ViT-B/16", "ViT-L/14"
DEFAULT_BACKBONE = "clip"         # Legacy option: "resnet18"
SONG_DATABASE_PATH = os.path.join(DATA_DIR, "song_database.pt")
TRAINED_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

# Training settings
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100
PATIENCE = 10
TEMPERATURE = 0.07
HARD_NEGATIVE_WEIGHT = 0.3
MIXUP_ALPHA = 0.2
TRAIN_TEST_SPLIT = 0.2
VAL_TEST_SPLIT = 0.5

# Server settings
HOST = "0.0.0.0"
PORT = 8080
DEBUG = True

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')