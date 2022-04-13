import numpy as np
import torch
from torch.utils.data import DataLoader


def prepare_data(dataset, device):
    # Prepare train, test, validation data and batches
    # Shuffle Data
    np.random.shuffle(dataset)

    # Train, Test, Validation splits
    num_examples = int(dataset.shape[0])
    train_data = dataset[: int(num_examples * 0.8)]
    val_data = dataset[int(num_examples * 0.8) : int(num_examples * 0.9)]
    test_data = dataset[int(num_examples * 0.9) : int(num_examples)]

    print("Training set shape   : ", train_data.shape)
    print("Validation set shape : ", val_data.shape)
    print("Test set shape       : ", test_data.shape)

    NUM_INPUT_FRAMES = 10

    def collate(batch):
        # Add channel dim, scale pixels between 0 and 1, send to GPU
        # TODO: don't create tensor from list
        batch = torch.tensor(np.array(batch)).unsqueeze(1)
        batch = batch.to(device)
        rand = np.random.randint(NUM_INPUT_FRAMES, 20)
        return (
            batch[:, :, rand - NUM_INPUT_FRAMES : rand],
            batch[:, :, rand],
        )
        # randomly extract 10 sequential frames as input
        # 11-th frame is label

    # Note the values of the input training batch and the output are (0~1)
    # Training Data Loader
    train_loader = DataLoader(
        train_data, shuffle=True, batch_size=16, collate_fn=collate
    )

    # Validation Data Loader
    val_loader = DataLoader(val_data, shuffle=True, batch_size=16, collate_fn=collate)

    return train_loader, val_loader, num_examples
