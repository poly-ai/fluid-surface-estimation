from bdb import set_trace
import os
import pathlib
import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from definitions import DATA_RAW_DIR, TRAINED_MODEL_DIR
from config_loader import load_config
from data.make_dataset import make_cfd_wave_dataset, make_omni_wave_dataset, make_cir_wave_dataset, make_von_karman_dataset
from data.augmentation import aug_random_affine_norm, aug_add_random_pairs, augment
from models.RL_CNN.cnn import *
from models.RL_CNN.model_train import *
from pathlib import Path

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
def main():

    # Config
    config, DATASET_PATH = load_config("config.yaml", "default_config.yaml")
    RE_TRAIN = config['load_model'] != None

    MODEL_TYPE_DIR = Path(os.path.join(TRAINED_MODEL_DIR,f"{config['model_type']}/"))

    if config['load_model'] != None:
        MODEL_LOAD_PATH = os.path.join(MODEL_TYPE_DIR, config['load_model']+".pt")
    else:
        MODEL_LOAD_PATH = ""

    MODEL_BEST_SAVE = Path(os.path.join(MODEL_TYPE_DIR, config['instance_id']+'.pt'))             # best model
    MODEL_LAST_SAVE = Path(os.path.join(MODEL_TYPE_DIR, config['instance_id']+'_last.pt'))   # last model
    TRAIN_HIST = Path(os.path.join(MODEL_TYPE_DIR, config['instance_id']+'.npy'))

    print("Best Model Save path: ", MODEL_BEST_SAVE)
    print("Last Model Save path: ", MODEL_LAST_SAVE)
    print("Train History Save path: ", TRAIN_HIST)

    # Create datasets
    if "omni" in config['datasets']:
        path:Path = DATASET_PATH['omni']

        if not config['force_regenerate_data'] and path.is_file():
            print(f"Omnidirectional dataset found: '{path.name}'")
        else:
            print(f"Generating raw omni dataset '{path.name}'")
            make_omni_wave_dataset(
                output_filepath=path,
                num_directions=config['omni_num_directions'],
                image_dimension=config['image_dimension'],
                num_frames=config['omni_num_frames'],
                wave_freqs=config['omni_wave_freqs'],
            )

    if "circ" in config['datasets']:
        path:Path = DATASET_PATH['circ']
        if not config['force_regenerate_data'] and path.is_file():
            print(f"Circular dataset found: '{path.name}'")
        else:
            print(f"Generating raw circular dataset '{path.name}'")
            make_cir_wave_dataset(
                output_filepath=path,
                image_dimension=config['image_dimension'],
                num_frames=config['circ_num_frames'],
            )

    if "shallow" in config['datasets']:
        path:Path = DATASET_PATH['shallow']
        if not config['force_regenerate_data'] and path.is_file():
            print(f"Shallow dataset found: '{path.name}'")
        else:
            print(f"Generating raw shallow dataset '{path.name}'")
            make_cfd_wave_dataset(output_filepath=path, slice=False)

    if "von-karman" in config['datasets']:
        path:Path = DATASET_PATH['von-karman']
        if not config['force_regenerate_data'] and path.is_file():
            print(f"Von-Karman dataset found: '{path.name}'")
        else:
            print(f"Generating raw Von-Karman dataset '{path.name}'")
            make_von_karman_dataset(output_filepath=path)


    # Load data
    ds_arr = []
    for (name,path) in DATASET_PATH.items():
        print(f"Loading dataset '{name}'")
        ds_arr.append(np.load(path))
    dataset = np.vstack(ds_arr)
    print("Datasets loaded successfully")

    # Data Augmentation
    print(f"Augmenting data with random seed {config['aug_seed']}")
    dataset = augment(dataset,
                      config['num_affine_aug'], 
                      config['num_sum_aug'],
                      config['aug_seed'])

    # Min and Max Check
    print("dataset shape before DataLoaders:", dataset.shape)
    print("Min and Max check. Max: ", np.max(dataset), " Min: ", np.min(dataset))

    # Convert to PyTorch Tensor
    dataset = torch.from_numpy(np.float32(dataset))
    NUM_TOTAL_VIDEOS = dataset.shape[0]
    NUM_VALID_VIDEOS = round(NUM_TOTAL_VIDEOS * config['val_fraction'])
    NUM_TRAIN_VIDEOS = NUM_TOTAL_VIDEOS - NUM_VALID_VIDEOS

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

    if config['model-type'] == "conv-lstm":
        
        
        
        model = Seq2Seq(
            num_channels=1,
            num_kernels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            activation="relu",
            frame_size=(64, 64),
            num_layers=3,
        ).to(device)
        
        model.train()  # Set model to training mode
        optim = Adam(model.parameters(), lr=1e-5)  # Select optimizer
        criterion = nn.MSELoss(reduction="sum")  # Select Loss Function
        
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

    else:

        # Create CNN Policy, Optimizer, Criterion Fcn
        policy = RL_CNN(H_in=dataset.shape[2], W_in=dataset.shape[3]).to(device)
        policy.apply(init_weights)
        optim = Adam(policy.parameters(), lr=1e-4)
        criterion = nn.MSELoss(reduction='mean')

        # Train
        train_RLCNN(policy, criterion, optim, input_data_train, MODEL_LOAD_PATH, MODEL_BEST_SAVE, MODEL_LAST_SAVE, TRAIN_HIST, config['stop_criteria'],
                    RE_TRAIN, NUM_TRAIN_VIDEOS, config['target_frame'], config['render_freq'], config['render_epoch'], config['num_epochs'], device, config['weight_play'],
                    config['save_model_and_history'], input_data_valid, NUM_VALID_VIDEOS)


if __name__ == '__main__':
    main()
