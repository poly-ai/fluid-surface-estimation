from pathlib import Path
import numpy as np
import config
from .CFD.wave import generate_cfd_data, translate_cfd_to_grid
from .wave_adv_omi import create_adv_diff_wave


def make_cfd_wave_dataset(output_filepath):
    path = Path(output_filepath)

    if not path.exists():
        x, y, h = generate_cfd_data()
        grid = translate_cfd_to_grid(x, y, h, 0.01)

        data = np.array(
            [
                grid[:, 0:64, 0:64],
                grid[:, 0:64, -64:],
                grid[:, -64:, -64:],
                grid[:, -64:, 0:64],
            ]
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_filepath, data)
        print(f"Saving dataset to {output_filepath}")


def make_omni_wave_dataset():
    path = Path(config.DATASET_FILEPATH)
    wave_num_directions = config.OMNI_WAVE_NUM_DIRECTIONS
    wave_freqs = config.OMNI_WAVE_SPEEDS

    step = wave_num_directions / 2 * np.pi
    for i in range(wave_num_directions, step=)


    #OLD
    adv_wave_vectors = [
        [1, 0],  # 0°    →
        [1, 1],  # 45°   ↗
        [0, 1],  # 90°   ↑
        [-1, 1],  # 135°  ↖
        [-1, 0],  # 180°  ←
        [-1, -1],  # 225°  ↙
        [0, -1],  # 270°  ↓
        [1, -1],  # 315°  ↘
    ]

    omni_data = []

    for wave_freq in wave_freqs:
    for wave_vector in adv_wave_vectors:
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
