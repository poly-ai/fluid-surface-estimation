import os
from definitions import DATA_RAW_DIR, TRAINED_MODEL_DIR

# Configuration file for driver program in main.py
# Ensure these settings are correct before beginning training...

# Dataset
CREATE_DATASET = False
DATASET_FILENAME = ["wave-omni.npy","wave-circ.npy","wave-shallow.npy", "wave-von-karman.npy"]
FRAMES_PER_EXAMPLE = 20

# Dataset augmentation
NUM_AFFINE_AUG = 2
NUM_SUM_AUG = 2
RANDOM_SEED = 0

# Model
USE_PRETRAINED_MODEL = False
TRAINED_MODEL = "ConvLSTM/pretrained-model.pt"  # Must exist within /models
SAVED_MODEL = "ConvLSTM/model-1.pt"  # Must exist within /models

# Training
NUM_EPOCHS = 10
PRED_FRAMES = 50  # Choose the number of frames to be predicted
VISUAL_FREQ = 10  # Visaulize every VISUAL_FRAMES


# Filepaths (for convenience)
# NOTE: no need to modify this. Simply change values above
DATASET_FILEPATH = [os.path.join(DATA_RAW_DIR, name) for name in DATASET_FILENAME]
TRAINED_MODEL_FILEPATH = os.path.join(TRAINED_MODEL_DIR, TRAINED_MODEL)
SAVED_MODEL_FILEPATH = os.path.join(TRAINED_MODEL_DIR, SAVED_MODEL)
