import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from definitions import DATA_RAW_DIR, PRE_TRAINED_MODEL_DIR
from data.make_dataset import make_omni_wave_dataset
from data.augmentation import aug_random_affine_norm
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
FRAMES_PER_EXAMPLE = 20

# Pretrained models
USE_PRETRAINED_MODEL = True 
PRE_TRAINED_MODEL_FILENAME = 'convLSTM/pretrained-model.pt'
SAVED_MODEL_FILENAME = 'convLSTM/model1.pt'

# Training
NUM_EPOCHS = 10


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
                               num_frames=1000,
                               wave_freq=1)
    
    # TODO: Augment data
    print("Creating augmented data")
    # TODO:

    # PyTorch device config
    print("Configuring PyTorch GPU usage")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    # Load data
    print("Loading dataset")
    dataset = np.load(os.path.join(DATA_RAW_DIR, DATASET_FILENAME)) # (8,100,64,64)
    dataset = np.float32(dataset)
    data_to_predict = dataset[1,:,:,:]  # Will predict on first video
    print("data to predict shape: ", data_to_predict.shape)

    ### Data Augmentation and Processing ###
    # Reshape data
    frames_per_video = dataset.shape[1]
    assert(frames_per_video % FRAMES_PER_EXAMPLE ==  0)
    dataset = np.reshape(dataset, (-1, FRAMES_PER_EXAMPLE, 64, 64))

    # Normalization
    # dataset = aug_random_affine_norm(dataset)  # found some strange operation in the data
    dataset = np.float32(((dataset*0.5)+0.5))
    print("dataset shape before DataLoaders:", dataset.shape)
    print("Normalization 0~1 check. Max: ", np.max(dataset), " Min: ", np.min(dataset))
    #########################################

    # Setup 
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
   
    # Setup Optimier and Loss Function 
    optim = Adam(model.parameters(), lr=1e-4)   # Select optimizer
    criterion = nn.MSELoss(reduction='sum')     # Select Loss Function

    # Initialize Best Val Loss
    best_val_loss = 100000000                   # Set a high value
    
    # Use pre-trained Model (Highly suggested using the pre-trained Model to reduce the possibiliy of nan Error during training)
    #pretrained_model = torch.load(os.path.join(DATA_DIR_PATH, 'sin-omni-V2.pth'), map_location=torch.device(device))
    if USE_PRETRAINED_MODEL:
        path_debug = os.path.join(PRE_TRAINED_MODEL_DIR, PRE_TRAINED_MODEL_FILENAME)
        print("Load Pretrained Model from path: ", path_debug)
        checkpoint = torch.load(os.path.join(PRE_TRAINED_MODEL_DIR, PRE_TRAINED_MODEL_FILENAME), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint['val_loss']
        del checkpoint
    
    model.train()                               # Set model to training mode    
    
    # Train the model
    print("Best Loss before Training: ",best_val_loss)
    print("Training the model")
    train_model(model=model, 
                train_loader=train_loader, 
                val_loader=val_loader, 
                optim=optim, 
                criterion=criterion, 
                num_examples=num_examples,
                device=device,
                num_epochs=NUM_EPOCHS,
                save_path = SAVED_MODEL_FILENAME,
                best_loss = best_val_loss
                )

    # Make a prediction
    print("Making prediction")
    predict_model(model, data_to_predict, device)

if __name__ == '__main__':
    main()
