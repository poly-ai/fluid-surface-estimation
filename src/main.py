import os
import pathlib
import numpy as np
import torch
import torch.nn as nn
from enum import Enum
import argparse
from torch.optim import Adam
from definitions import DATA_RAW_DIR, PRE_TRAINED_MODEL_DIR
from data.normalization import normalize
from data.make_dataset import make_cfd_wave_dataset, make_omni_wave_dataset
from data.augmentation import aug_random_affine_norm, aug_add_random_pairs
from models.ConvLSTM.Seq2Seq import Seq2Seq
from models.ConvLSTM.prepare_data import prepare_data
from models.ConvLSTM.model_train import train_model
from models.ConvLSTM.model_predict import predict_model

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------

# Note to Julien - these configurations were commented out and essentially 
# replaced with the dataset_path argument. Do with it as you see fit.

# Datasets
# CREATE_DATASET = False
# DATASET_FILENAME = 'wave-sine-omni.npy'
# DATASET_FILENAME = 'wave-shallow.npy'
# Options:  wave-sine-omni.npy
#           wave-shallow.npy

FRAMES_PER_EXAMPLE = 20

# Pretrained models
USE_PRETRAINED_MODEL = True
PRETRAINED_MODEL_FILEPATH = os.path.join(PRE_TRAINED_MODEL_DIR, 'ConvLSTM/pretrained-model.pt')
SAVED_MODEL_FILEPATH = os.path.join(PRE_TRAINED_MODEL_DIR, 'ConvLSTM/model-1.pt')

# Augmentation
NUM_AFFINE_AUG = 2
NUM_SUM_AUG = 2

# Training
NUM_EPOCHS = 10

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
def main():

    # Command-line arguments
    parser = argparse.ArgumentParser(description="Train and test various ML models for fluid simulation.")
    parser.add_argument("datapath", type=pathlib.Path, help="Relative or absolute path to dataset.")
    args = parser.parse_args()

    # Create datasets
    dataset_path = args.datapath.resolve()

    # If dataset not found, prompt to create a new one on specified type
    if not dataset_path.exists():
        print(f"Dataset not found. Generating raw dataset '{dataset_path.name}'")
        print("Types of waves:")
        print("    [0] Omnidirectional Sine Wave")
        print("    [1] CFD Wave")

        wave_type = input("Select a type of wave to generate: ")

        if wave_type == "0":
            print(f"Generating omnidirectional sine wave at '{dataset_path}'")
            make_omni_wave_dataset(output_filepath=dataset_path,
                                image_dimension=64,
                                num_frames=1000,
                                wave_freq=1)
        elif wave_type == "1":
            print(f"Generating CFD wave at '{dataset_path}'")
            make_cfd_wave_dataset(output_filepath=dataset_path)
        else:
            print("Error: Wave type was not 0 or 1")
            return 1

    # PyTorch device config
    print("Configuring PyTorch GPU usage")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    # Load data
    print("Loading dataset")
    dataset = np.load(dataset_path) # (8,100,64,64)
    dataset = np.float32(dataset)
    data_to_predict = dataset[1,:,:,:]  # Will predict on first video
    print("data to predict shape: ", data_to_predict.shape)

    # Reshape data
    frames_per_video = dataset.shape[1]
    frames_to_remove = frames_per_video % FRAMES_PER_EXAMPLE
    dataset = dataset[:, 0:frames_per_video-frames_to_remove, :, :]
    dataset = np.reshape(dataset, (-1, FRAMES_PER_EXAMPLE, 64, 64))

    # Normalize dataset pre-augmentation
    dataset = normalize(dataset)

    ### Data Augmentation ###

    print("Augmenting data")
    orig_dataset_size = dataset.shape[0]

    # Python list containing all augmentations, to contatenate at end
    aug_list = []
    aug_list.append(dataset.copy())

    # Normalized random-affine augmentations
    for _ in range(0, NUM_AFFINE_AUG):
        aug_list.append(aug_random_affine_norm(dataset))     # Random affine
    dataset = np.vstack(aug_list)

    # Random-pair-sum augmentations, inside normalized random-affine augmentations
    dataset = np.vstack((dataset, aug_random_affine_norm(aug_add_random_pairs(dataset, orig_dataset_size * NUM_SUM_AUG))))
    
    # Normalization
    print("dataset shape before DataLoaders:", dataset.shape)
    print("Normalization 0~1 check. Max: ", np.max(dataset), " Min: ", np.min(dataset))
    #########################################

    # Setup
    train_loader, val_loader, num_examples = prepare_data(dataset, device)
    print(f"Loaded dataset {dataset_path.name}")

    # Init model (input video frames are grayscale => single channel)
    print("Initializing the model")
    model = Seq2Seq(num_channels=1,
                    num_kernels=64,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                    activation="relu",
                    frame_size=(64, 64),
                    num_layers=3).to(device)

    # Setup Optimier and Loss Function
    optim = Adam(model.parameters(), lr=1e-4)   # Select optimizer
    criterion = nn.MSELoss(reduction='sum')     # Select Loss Function

    # Initialize Best Val Loss
    best_val_loss = 100000000                   # Set a high value

    # Use pre-trained Model (Highly suggested using the pre-trained Model to reduce the possibiliy of nan Error during training)
    #pretrained_model = torch.load(os.path.join(DATA_DIR_PATH, 'sin-omni-V2.pth'), map_location=torch.device(device))
    if USE_PRETRAINED_MODEL:
        print(f"Loading pretrained model from {PRETRAINED_MODEL_FILEPATH}")
        model.load_state_dict(torch.load(PRETRAINED_MODEL_FILEPATH, map_location=torch.device(device)))
        checkpoint = torch.load(PRETRAINED_MODEL_FILEPATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint['val_loss']
        del checkpoint

    model.train()                               # Set model to training mode    
    optim = Adam(model.parameters(), lr=1e-5)   # Select optimizer
    criterion = nn.MSELoss(reduction='sum')     # Select Loss Function

    # Train the model
    print("Best Loss before Training: ",best_val_loss)
    print("Training the model")
    train_model(model       =model,
                train_loader=train_loader,
                val_loader  =val_loader,
                optim       =optim,
                criterion   =criterion,
                num_examples=num_examples,
                device      =device,
                num_epochs  =NUM_EPOCHS,
                save_path   = SAVED_MODEL_FILEPATH,
                best_loss   = best_val_loss
                )

    # Make a prediction
    print("Making prediction")
    predict_model(model, data_to_predict, device)

if __name__ == '__main__':
    main()
