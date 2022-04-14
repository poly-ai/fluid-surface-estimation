import numpy as np
import math
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Fixed for the time being...
WAVE_FREQ = 10      # This scalar defines the speed of the wave
WAVE_NUMBER = 1      # This scalar defines the speed of the wave
X_CENTER = np.pi
Y_CENTER = np.pi

IMAGE_DIMENSION = 64
NUM_FRAMES = 400
dt = 0.01 # Constant now (Don't change this)

# This vector defines the direction of the travelling wave. 
# The magnitude of this vector defines the "length" of the wave


def create_cir_wave(image_dimension=IMAGE_DIMENSION, 
                         num_frames=NUM_FRAMES, 
                         wave_freq=WAVE_FREQ,
                         wave_number=WAVE_NUMBER,
                         x_center=X_CENTER,
                         y_center=Y_CENTER):
    
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
                x = dx*j - x_center 
                y = (image_dimension-i)*dy - y_center
                r = (1+(x*x+y*y)**(0.5))
                data[k,i,j] = 1*math.cos(wave_number*r-wave_freq*t0)/(r**0.5)

    # DEBUG
    create_2D_animation(data)

    return data

# ------------------------------------------------------------------------------
# Create 2D video
# ------------------------------------------------------------------------------
def animate_2D(frame_number, image_ref: AxesImage, data):
    image_ref.set_array(data[frame_number,:,:])
    return frame_number

def show_2D_animation(data):
    fig = plt.figure()
    ax = plt.axes()
    im = plt.imshow(data[0,:,:], cmap="gray")
    num_frames = data.shape[0]
    anim = animation.FuncAnimation(fig, animate_2D, interval=2,
                                   fargs=(im, data), frames=num_frames)
    plt.show()


def main():
    wave = create_cir_wave()
    show_2D_animation(wave)

if __name__ == "__main__":
    main()
