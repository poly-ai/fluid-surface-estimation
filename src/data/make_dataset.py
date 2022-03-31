import numpy as np
from .wave_adv_omi import create_adv_diff_wave

def make_omni_wave_dataset(output_filepath, image_dimension, num_frames):

    adv_wave_vectors = [
        [1, 0],     # 0°    →
        [1, 1],     # 45°   ↗
        [0, 1],     # 90°   ↑
        [-1, 1],    # 135°  ↖
        [-1, 0],    # 180°  ←
        [-1, -1],   # 225°  ↙
        [0, -1],    # 270°  ↓
        [1, -1]     # 315°  ↘
    ]

    omni_data = []

    for wave_vector in adv_wave_vectors:
        data, theta = create_adv_diff_wave(image_dimension, num_frames, adv_wave_vector=wave_vector)
        omni_data.append(data)
        print(f"Created wave data (direction: {theta}°)")

    omni_data = np.stack(omni_data)
    print("Dataset shape:", omni_data.shape)

    # Save output
    np.save(output_filepath, omni_data)
    print(f'Saving dataset to {output_filepath}')

