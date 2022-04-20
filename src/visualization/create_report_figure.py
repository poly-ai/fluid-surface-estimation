import os
import numpy as np
import matplotlib.pyplot as plt
import src.definitions as definitions
from src.data.wave_adv_omi import create_adv_diff_wave
from src.visualization.create_animation import create_3D_animation

DATASET_FILEPATH = os.path.join(definitions.DATA_RAW_DIR, 'wave-sine-omni.npy')
CMAP = "plasma"

def create_image(frame, index, min_z, max_z, width, height):
    
    X = np.arange(width)
    Y = np.arange(height)
    X, Y = np.meshgrid(X, Y)

    # Create figure
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_zlabel("z axis")
    ax.set_zlim(min_z, max_z)
    ax.plot_surface(X, Y, frame, cmap=CMAP, linewidth=0, rstride=1, cstride=1)

    # plt.show()
    plt.savefig(f"frame-{index}.png")


def strange_wave():
    num_frames = 100
    data1, _ = create_adv_diff_wave(64, num_frames, [1, 0], 20)
    data2, _ = create_adv_diff_wave(64, num_frames, [0, -1], 20)
    data3, _ = create_adv_diff_wave(64, num_frames, [1, -1], 20)
    data3 = np.roll(data3, 10, axis=0)
    data = ((data1 + data2 + data3) + 3) / 6

    # create_3D_animation(data, None)
    min_z = np.min(data)
    max_z = np.max(data)
    num_frames = data.shape[0]
    height = data.shape[1]
    width = data.shape[2]

    for i in range(20, 40, 2):
        create_image(data[i, :, :], i, min_z, max_z, width, height)


def visualize_augmentation():
    data1, _ = create_adv_diff_wave(64, 100, [1, 0], 20)
    data2, _ = create_adv_diff_wave(64, 100, [0, 1], 20)
    data1 = (data1 + 1) / 2
    data2 = (data2 + 1) / 2
    data = (data1 + data2) / 2

    create_image(data1[0, :, :], 'data-1', 0, 1, 64, 64)
    create_image(data2[0, :, :], 'data-2', 0, 1, 64, 64)
    create_image(data[0, :, :], 'data-aug', 0, 1, 64, 64)

visualize_augmentation()