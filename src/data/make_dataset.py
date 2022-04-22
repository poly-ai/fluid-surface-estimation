import math
from pathlib import Path
import numpy as np
import config
from .CFD.wave import generate_cfd_data, translate_cfd_to_grid
from .wave_adv_omi import create_adv_diff_wave
from .wave_cir import create_cir_wave
#from .von_karman import load_von_karman_dataset


def adv_wave_vec_from_radians(radians):
    return [math.sin(radians), math.sin(radians)]


def make_omni_wave_dataset(output_filepath):
    path = Path(config.DATASET_FILEPATH)
    wave_num_directions = config.OMNI_WAVE_NUM_DIRECTIONS
    wave_freqs = config.OMNI_WAVE_SPEEDS

    # Create wave direction vectors
    wave_adv_directions = []
    for radians in range(0, 2*math.pi, 2*math.pi / wave_num_directions):
        wave_adv_directions.append([math.sin(radians), math.cos(radians)])

    # Create wave videos
    for wave_vector in wave_adv_directions:
        for wave_freq in wave_freqs:
            data, theta = create_adv_diff_wave(
                image_dimension=config.IMAGE_DIMENSION,
                num_frames=config.NUM_FRAMES,
                adv_wave_vector=wave_vector,
                adv_wave_freq=wave_freq,
            )
            omni_data.append(data)
            print(f"Created wave data (direction: {theta} degrees)")

    omni_data = np.stack(omni_data)
    print("Dataset shape:", omni_data.shape)

    # Save output
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_filepath, omni_data)
    print(f"Saving dataset to {output_filepath}")


def make_cir_wave_dataset(output_filepath, image_dimension, num_frames):

    wave_freq_list = [1,5]
    wave_number_list = [1,5]
    x_center_list = [1,3,5]
    y_center_list = [1,3,5]

    cir_data = []

    for wave_freq in wave_freq_list:
        for wave_number in wave_number_list:
            for x_center in x_center_list:
                for y_center in y_center_list:
                    print(f"create cir wave wave freq:{wave_freq}, wave number {wave_number}, x center {x_center}, y center {y_center}")
                    data = create_cir_wave(image_dimension=image_dimension,
                                           num_frames=num_frames,
                                           wave_freq=wave_freq,
                                           wave_number=wave_number,
                                           x_center=x_center,
                                           y_center=y_center)
                    cir_data.append(data)

    # Stack results
    cir_data = np.stack(cir_data)

    # Save output
    np.save(output_filepath, cir_data)
    print(f'Saving dataset to {output_filepath}')


def make_cfd_wave_dataset(output_filepath, slice=True):
    path = Path(output_filepath)

    x_center = 0.5
    y_center = 0.5
    x_distri = 100
    y_distri = 100
    height_level = 0.6
    height_delta = 0.6

    if not path.exists():
        x, y, h = generate_cfd_data(x_center, y_center, x_distri, y_distri, height_level, height_delta)
        grid = translate_cfd_to_grid(x, y, h, 0.01)

        if slice:
            data = np.array(
                [
                    grid[:, 0:64, 0:64],
                    grid[:, 0:64, -64:],
                    grid[:, -64:, -64:],
                    grid[:, -64:, 0:64],
                ]
            )
        else:
            data = np.array([grid])

        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_filepath, data)
        print(f"Saving dataset to {output_filepath}")


def make_von_karman_dataset(output_filepath):
    path = Path(output_filepath)

    if not path.exists():
        h = load_von_karman_dataset()
        dataset = np.array([h])

        # Save output
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_filepath, dataset)
        print(f"Saving dataset to {output_filepath}")
