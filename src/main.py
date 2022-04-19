import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import config
from data.normalization import normalize
from data.make_dataset import *
from data.augmentation import aug_random_affine_norm, aug_add_random_pairs, augment
from models.ConvLSTM.Seq2Seq import Seq2Seq
from models.ConvLSTM.prepare_data import prepare_data
from models.ConvLSTM.model_train import train_model
from models.ConvLSTM.model_predict import predict_model


def main():
    # PyTorch device config
    print("Configuring PyTorch GPU usage")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Create datasets
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

    if config.CREATE_DATASET or not os.path.exists(config.DATASET_FILEPATH[2]):
        print("hi")
        make_cfd_wave_dataset(output_filepath=config.DATASET_FILEPATH[2], slice=False)

    if config.CREATE_DATASET or not os.path.exists(config.DATASET_FILEPATH[3]):
        print("hi")
        make_von_karman_dataset(output_filepath=config.DATASET_FILEPATH[3])

    # Load data
    print("Loading dataset")
    dataset_1 = np.float32(np.load(config.DATASET_FILEPATH[0]))
    dataset_2 = np.float32(np.load(config.DATASET_FILEPATH[1]))
    dataset_3 = np.float32(np.load(config.DATASET_FILEPATH[2]))
    dataset_4 = np.float32(np.load(config.DATASET_FILEPATH[3]))

    # Testing Different Dataset
    return 0

    # Reshape data
    frames_per_video = dataset.shape[1]
    frames_to_remove = frames_per_video % config.FRAMES_PER_EXAMPLE
    dataset = dataset[:, 0 : frames_per_video - frames_to_remove, :, :]
    dataset = np.reshape(dataset, (-1, config.FRAMES_PER_EXAMPLE, 64, 64))

    # Normalize dataset pre-augmentation
    dataset = normalize(dataset)

    # Data Augmentation
    print("Augmenting data")
    dataset = augment(dataset, config.NUM_AFFINE_AUG,
                      config.NUM_SUM_AUG, config.RANDOM_SEED)

    # Min and Max Check
    print("dataset shape before DataLoaders:", dataset.shape)
    print("Norm 0~1 check. Max: ", np.max(dataset), " Min: ", np.min(dataset))

    # Setup
    train_loader, val_loader, num_examples = prepare_data(dataset, device)
    print(f"Loaded and prepared dataset {config.DATASET_FILENAME}")

    # Init model (input video frames are grayscale => single channel)
    print("Initializing the model")
    model = Seq2Seq(
        num_channels=1,
        num_kernels=64,
        kernel_size=(3, 3),
        padding=(1, 1),
        activation="relu",
        frame_size=(64, 64),
        num_layers=3,
    ).to(device)

    # Setup Optimier and Loss Function
    optim = Adam(model.parameters(), lr=1e-4)  # Select optimizer
    criterion = nn.MSELoss(reduction="sum")  # Select Loss Function

    # Initialize Best Val Loss
    best_val_loss = 100000000  # Set a high value

    # Use pre-trained Model (Highly suggested using the pre-trained Model to
    # reduce the possibiliy of nan Error during training)
    if config.USE_PRETRAINED_MODEL:
        print(f"Loading pretrained model from {config.TRAINED_MODEL_FILEPATH}")
        state_dict = torch.load(
            config.TRAINED_MODEL_FILEPATH, map_location=torch.device(device)
        )
        model.load_state_dict(state_dict)
        checkpoint = torch.load(config.TRAINED_MODEL_FILEPATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_loss = checkpoint["val_loss"]
        del checkpoint

    model.train()  # Set model to training mode
    optim = Adam(model.parameters(), lr=1e-5)  # Select optimizer
    criterion = nn.MSELoss(reduction="sum")  # Select Loss Function

    # Train the model
    print("Best Loss before Training: ", best_val_loss)
    print("Training the model")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=optim,
        criterion=criterion,
        num_examples=num_examples,
        device=device,
        num_epochs=config.NUM_EPOCHS,
        save_path=config.SAVED_MODEL_FILEPATH,
        best_loss=best_val_loss,
    )



if __name__ == "__main__":
    main()
