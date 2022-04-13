import numpy as np
import math

# Fixed for the time being...
ADV_WAVE_FREQ = 1      # This scalar defines the speed of the wave
IMAGE_DIMENSION = 64
NUM_FRAMES = 100
dt = 0.01 # Constant now (Don't change this)

# This vector defines the direction of the travelling wave. 
# The magnitude of this vector defines the "length" of the wave
ADV_WAVE_VECTOR = np.array([1,0])


def create_adv_diff_wave(image_dimension=IMAGE_DIMENSION, 
                         num_frames=NUM_FRAMES, 
                         adv_wave_vector=ADV_WAVE_VECTOR,
                         adv_wave_freq=ADV_WAVE_FREQ):
    
    # Degrees between wave direction and x-axis
    theta = math.atan2(adv_wave_vector[1], adv_wave_vector[0]) * 180 / math.pi

    # Data to generate
    data = np.zeros((num_frames,image_dimension,image_dimension))

    # Deltas, num ticks in example
    dx = 2*np.pi/(image_dimension-1)
    dy = dx
    ticks = np.linspace(0, dt*num_frames, num_frames)

    # Create frames
    for k in range(num_frames):
        t0 = ticks[k]
        for i in range(image_dimension):
            for j in range(image_dimension):
                x = dx*j
                y = (image_dimension-i)*dy
                r = np.array([x,y])
                data[k,i,j] = math.sin(np.dot(adv_wave_vector,r)-adv_wave_freq*t0)

    return data, theta
