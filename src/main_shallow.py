import os
import pathlib
import numpy as np
import random
import torch
import torch.nn as nn
from enum import Enum
import argparse
from torch.optim import Adam
from definitions import DATA_RAW_DIR, TRAINED_MODEL_DIR
import config
from data.make_dataset import make_cfd_wave_dataset, make_omni_wave_dataset, make_cir_wave_dataset
from data.augmentation import aug_random_affine_norm, aug_add_random_pairs, augment
from models.RL_CNN.cnn_deep import *
from models.RL_CNN.model_train import *
from models.RL_CNN.resnet18 import ResNet18

#from visualization.create_animation import *

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------

### Configuration ###
NUM_TOTAL_VIDEOS = 80
NUM_TRAIN_VIDEOS = 60
NUM_TEST_VIDEOS  = 10
NUM_VALID_VIDEOS  = 10

# Training Setup 
# Label for training
TRAIN_ID = "cnn-unnorm-shallow-1"
# bool: Load pre-trained model
ReTrain = False
# If Retrain == true: set the pre-trained model path
MODEL_LOAD_PATH = os.path.join(TRAINED_MODEL_DIR, 'RL/12.00.pt')

TARGET_FRAME = 200
isSave = True        # Set this to True if you want to Save Model and History
NUM_EPOCH = 50000    # number of epochs
RENDER_FREQ = 10     # print info every () frames
RENDER_EPOCH = 50  # save train history every () epochs
STOP_CRITERIA = 10000

# Augmentation
NUM_AFFINE_AUG = 0  
NUM_SUM_AUG = 0 
RANDOM_SEED = 0

# RL Hyper Parameter 
WEIGHT_PLAY = 20
#############################################

MODEL_BEST_SAVE = os.path.join(TRAINED_MODEL_DIR, 'RL/'+TRAIN_ID+'.pt')             # best model
MODEL_LAST_SAVE = os.path.join(TRAINED_MODEL_DIR, 'RL/'+TRAIN_ID+'_last.pt')   # last model
TRAIN_HIST = os.path.join(TRAINED_MODEL_DIR, 'RL/'+TRAIN_ID+'.npy')

print("Best Model Save path: ", MODEL_BEST_SAVE)
print("Last Model Save path: ", MODEL_LAST_SAVE)
print("Train History Save path: ", TRAIN_HIST)


#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
def main():
    # Create datasets
    '''
    if config.CREATE_DATASET or not os.path.exists(config.DATASET_FILEPATH[0]):
        print(f"Creating raw dataset {config.DATASET_FILENAME[0]}")
        make_omni_wave_dataset(
            output_filepath=config.DATASET_FILEPATH[0],
            image_dimension=64,
            num_frames=1000,
            wave_freq=1,
        )

    if config.CREATE_DATASET or not os.path.exists(config.DATASET_FILEPATH[1]):
        print(f"Creating raw dataset {config.DATASET_FILENAME[1]}")
        make_cir_wave_dataset(
            output_filepath=config.DATASET_FILEPATH[1],
            image_dimension=64,
            num_frames=1000,
        )
    '''

    if config.CREATE_DATASET or not os.path.exists(config.DATASET_FILEPATH[2]):
        print(f"Creating raw dataset {config.DATASET_FILENAME[2]}")
        make_cfd_wave_dataset(output_filepath=config.DATASET_FILEPATH[2], num_videos = NUM_TOTAL_VIDEOS, slice=False)
     
    # Load data (0: Omni, 1: Circ, 2: Shallow)
    print("Loading dataset")
    dataset = np.load(config.DATASET_FILEPATH[2])
    print("raw dataset load successfully")
    print("raw dataset shape: ", dataset.shape)


    # Separate test data from dataset
    dataset = dataset[0:NUM_TOTAL_VIDEOS-NUM_TEST_VIDEOS]
    
    # Min and Max Check
    print("Min and Max check. Max: ", np.max(dataset), " Min: ", np.min(dataset))
    
    # Convert to PyTorch Tensor
    dataset = torch.from_numpy(np.float32(dataset))

    # Pick Training Data
    pick = np.arange(NUM_TRAIN_VIDEOS)
    input_data_train = dataset[pick]
    input_data_valid = dataset[NUM_TRAIN_VIDEOS:]
    print("train data shape: ", input_data_train.shape)
    print("valid data shape: ", input_data_valid.shape)

    # Configure for CUDA
    print("Configuring PyTorch GPU usage")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    # Create CNN Policy, Optimizer, Criterion Fcn
    # frames_in=10, H_in=64, W_in=64,
    #             kernels_1=30, pad_1=1, kern_sz_1=5, stride_1=1,
    #             pool_sz_1=2,
    #             kernels_2=60, pad_2=1, kern_sz_2=5, stride_2=1,
    #             pool_sz_2=2,
    #             h_3=4096, h_4=4096, frames_out=1)
    policy = RL_CNN_DEEP(H_in=88,W_in=58,h_6=88*58).to(device)
    policy.apply(init_weights)
    optim = Adam(policy.parameters(), lr=1e-4)
    criterion = nn.MSELoss(reduction='mean')

    # Train
    train_RLCNN(policy, criterion, optim, input_data_train, MODEL_LOAD_PATH, MODEL_BEST_SAVE, MODEL_LAST_SAVE, TRAIN_HIST, STOP_CRITERIA,
                ReTrain, NUM_TRAIN_VIDEOS, TARGET_FRAME, RENDER_FREQ, RENDER_EPOCH, NUM_EPOCH, device, WEIGHT_PLAY, isSave,
                input_data_valid, NUM_VALID_VIDEOS)


if __name__ == '__main__':
    main()
