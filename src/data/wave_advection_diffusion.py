import numpy as np
import math

# ------------------------------------------------------------------------------
# Default configuration
# ------------------------------------------------------------------------------
IMAGE_DIMENSION = 64
NUM_FRAMES = 1000
# Const
dt = 0.01

# Weights for the Linear Combination of
# advecation (travelling) and diffusion (spreading) waves.
# W_ADV = 1, W_DIFF = 0 : pure travelling wave
# W_ADV = 0, W_DIFF = 1 : pure diffusion wave
W_ADV = 1
W_DIFF = 0

# Fixed for the time being...
ADV_WAVE_NUMBER = 1  # number of waves in a frame
ADV_WAVE_FREQ = 1  # speed of waves


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def adv_height(x, t):
    k = ADV_WAVE_NUMBER
    w = ADV_WAVE_FREQ
    return math.sin(k * x - w * t)


def normal(x, xu, std):
    return math.exp(-0.5 * ((x - xu) / std) ** 2) / (std * (2 * np.pi) ** 0.5)


def diff_height(x, t):
    xu = np.pi  # xu1 = 1.2*np.pi; #xu2 = 0.8*np.pi
    std = 1 * t + 0.01
    return normal(x, xu, std)


def adv_dif_combined_height(x, t, w_adv, w_diff):
    return w_adv * adv_height(x, t) + w_diff * diff_height(x, t)


# ------------------------------------------------------------------------------
# Create wave
# ------------------------------------------------------------------------------
def create_wave(
    image_dimension=IMAGE_DIMENSION, num_frames=NUM_FRAMES, w_adv=W_ADV, w_diff=W_DIFF
):

    ticks = np.linspace(0, dt * num_frames, num_frames)
    spatial_coordinates_1d = np.linspace(0, 2 * np.pi, image_dimension)
    data_1d = np.zeros((num_frames, image_dimension))  # (time, position)

    for i in range(num_frames):
        for j in range(image_dimension):
            data_1d[i, j] = adv_dif_combined_height(
                spatial_coordinates_1d[j], ticks[i], w_adv, w_diff
            )

    # Expand to 2D Height Matrix Data
    data = np.zeros((num_frames, image_dimension, image_dimension))
    for i in range(num_frames):
        U2d = np.tile(data_1d[i, :], (image_dimension, 1))
        data[i, :, :] = U2d

    return data
