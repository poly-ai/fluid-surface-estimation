import os
from definitions import DATA_RAW_DIR, TRAINED_MODEL_DIR
from src.data.wave_adv_omi import IMAGE_DIMENSION

# Configuration file for driver program in main.py
# Ensure these settings are correct before beginning training...

# Dataset generation
IMAGE_DIMENSION = 64
NUM_FRAMES = 1000
OMNI_WAVE_NUM_DIRECTIONS = 12       # Divide range [0°, 360°) 
OMNI_WAVE_SPEEDS = [0.5, 1, 2, 4]

# Dataset
CREATE_DATASET = False
DATASET_FILENAME = "wave-shallow.npy"
FRAMES_PER_EXAMPLE = 20

# Dataset augmentation
NUM_AFFINE_AUG = 2
NUM_SUM_AUG = 2

# Model
USE_PRETRAINED_MODEL = True
TRAINED_MODEL = "ConvLSTM/pretrained-model.pt"  # Must exist within /models
SAVED_MODEL = "ConvLSTM/model-1.pt"  # Must exist within /models

# Training
NUM_EPOCHS = 10
PRED_FRAMES = 50  # Choose the number of frames to be predicted
VISUAL_FREQ = 10  # Visaulize every VISUAL_FRAMES


# Filepaths (for convenience)
# NOTE: no need to modify this. Simply change values above
DATASET_FILEPATH = os.path.join(DATA_RAW_DIR, DATASET_FILENAME)
TRAINED_MODEL_FILEPATH = os.path.join(TRAINED_MODEL_DIR, TRAINED_MODEL)
SAVED_MODEL_FILEPATH = os.path.join(TRAINED_MODEL_DIR, SAVED_MODEL)
