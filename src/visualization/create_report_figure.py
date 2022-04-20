import os
import numpy as np
import matplotlib.pyplot as plt
from definitions import DATA_RAW_DIR
from src.data.wave_adv_omi import create_adv_diff_wave
from src.visualization.create_animation import create_3D_animation

DATASET_FILEPATH = os.path.join(DATA_RAW_DIR, 'wave-sine-omni.npy')
CMAP = "plasma"

def create_image(frame, min_z, max_z, width, height):
    
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

    plt.show()


def main():

    data, _ = create_adv_diff_wave(64, 100, [0, 1], 10)
    create_3D_animation(data, None)
    # min_z = np.min(data)
    # max_z = np.max(data)
    # num_frames = data.shape[0]
    # height = data.shape[1]
    # width = data.shape[2]

    # for i in range(0, 100, 20):
    #     create_image(min_z, max_z, width, height)