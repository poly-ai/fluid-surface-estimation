import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import config
from data.normalization import normalize
from data.make_dataset import make_omni_wave_dataset
from data.augmentation import aug_random_affine_norm, aug_add_random_pairs
from models.ConvLSTM.Seq2Seq import Seq2Seq
from models.ConvLSTM.prepare_data import prepare_data
from models.ConvLSTM.model_train import train_model
from models.ConvLSTM.model_predict import predict_model


def main():
    # Create datasets
    if config.CREATE_DATASET or not os.path.exists(config.DATASET_FILEPATH):
        print(f"Creating raw dataset {config.DATASET_FILENAME}")
        make_omni_wave_dataset(
            output_filepath=config.DATASET_FILEPATH,
            image_dimension=64,
            num_frames=1000,
            wave_freq=1,
        )

    # PyTorch device config
    print("Configuring PyTorch GPU usage")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Load data
    print("Loading dataset")
    dataset = np.float32(np.load(config.DATASET_FILEPATH))  # (8,100,64,64)
    data_to_predict = dataset[1, :, :, :]  # Will predict on first video
    print("data to predict shape: ", data_to_predict.shape)

    # Reshape data
    frames_per_video = dataset.shape[1]
    frames_to_remove = frames_per_video % config.FRAMES_PER_EXAMPLE
    dataset = dataset[:, 0 : frames_per_video - frames_to_remove, :, :]
    dataset = np.reshape(dataset, (-1, config.FRAMES_PER_EXAMPLE, 64, 64))

    # Normalize dataset pre-augmentation
    dataset = normalize(dataset)

    # Data Augmentation
    print("Augmenting data")
    orig_dataset_size = dataset.shape[0]

    # Python list containing all augmentations, to contatenate at end
    aug_list = []
    aug_list.append(dataset.copy())

    # Normalized random-affine augmentations
    for _ in range(0, config.NUM_AFFINE_AUG):
        aug_list.append(aug_random_affine_norm(dataset))  # Random affine
    dataset = np.vstack(aug_list)

    # Random-pair-sum augment. inside normalized random-affine augment.
    aug_data = aug_add_random_pairs(dataset, orig_dataset_size * config.NUM_SUM_AUG)
    aug_data = aug_random_affine_norm(aug_data)
    dataset = np.vstack((dataset, aug_data))

    # Normalization
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

    # Make a prediction
    print("Making prediction")
    predict_model(model, data_to_predict, device)


if __name__ == "__main__":
    main()
