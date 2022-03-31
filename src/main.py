import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from definitions import DATA_RAW_DIR, PRE_TRAINED_MODEL_DIR
from data.make_dataset import make_omni_wave_dataset
from models.conv_LSTM.Seq2Seq import Seq2Seq
from models.conv_LSTM.prepare_data import prepare_data
from models.conv_LSTM.model_train import train_model
from models.conv_LSTM.model_predict import predict_model

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------
# Datasets
# Options:  wave-sine-omni.npy
#           wave-shallow.npy
CREATE_DATASET = False
DATASET_FILENAME = 'wave-sine-omni.npy' 

# Pretrained models
USE_PRETRAINED_MODEL = False
PRE_TRAINED_MODEL_FILENAME = '../wave-sine-omni.pth'


#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
def main():

    # Create datasets
    dataset_path = os.path.join(DATA_RAW_DIR, DATASET_FILENAME)
    if CREATE_DATASET or not os.path.exists(dataset_path):
        print(f"Creating raw dataset {DATASET_FILENAME}")
        make_omni_wave_dataset(output_filepath=dataset_path,
                               image_dimension=64, 
                               num_frames=100)
    
    # TODO: Augment data
    print("Creating augmented data")
    # TODO:

    # PyTorch device config
    print("Configuring PyTorch GPU usage")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    # Load and prepare data
    print("Loading dataset")
    dataset = np.load(os.path.join(DATA_RAW_DIR, DATASET_FILENAME))
    dataset = np.float32(dataset)
    data_to_predict = dataset[0,:,:,:]  # Will predict on first example
    train_loader, val_loader, num_examples = prepare_data(dataset, device)
    print(f"Loaded dataset {DATASET_FILENAME}")

    # Init model (input video frames are grayscale => single channel)
    print("Initializing the model")
    model = Seq2Seq(num_channels=1, 
                    num_kernels=64, 
                    kernel_size=(3, 3), 
                    padding=(1, 1), 
                    activation="relu", 
                    frame_size=(64, 64), 
                    num_layers=3).to(device)

    # Use pre-trained Model (Highly suggested using the pre-trained Model to reduce the possibiliy of nan Error during training)
    #pretrained_model = torch.load(os.path.join(DATA_DIR_PATH, 'sin-omni-V2.pth'), map_location=torch.device(device))
    if USE_PRETRAINED_MODEL:
        model.load_state_dict(torch.load(os.path.join(PRE_TRAINED_MODEL_DIR, PRE_TRAINED_MODEL_FILENAME), map_location=torch.device(device)))
    
    model.train()                               # Set model to training mode    
    optim = Adam(model.parameters(), lr=1e-5)   # Select optimizer
    criterion = nn.MSELoss(reduction='sum')     # Select Loss Function
    
    # Train the model
    print("Training the model")
    train_model(model, train_loader, val_loader, optim, criterion, num_examples, device)

    # Make a prediction
    print("Making prediction")
    predict_model(model, data_to_predict, device)

if __name__ == '__main__':
    main()