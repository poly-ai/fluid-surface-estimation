import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
import time
import sys
import matplotlib.animation as animation

SAVE_PATH = "output/wave-sin-theta_"

IMAGE_DIMENSION = 64
NUM_FRAMES = 100
dt = 0.01 # Constant now (Don't change this)

# Fixed for the time being...
ADV_WAVE_VECTOR = np.array([-1,-1])     # This vector defines the direction of the travelling wave. The magnitude of this vector defines the "length" of the wave
ADV_WAVE_FREQ = 10                      # This scalar defines the speed of the wave

def create_adv_diff_wave(image_dimension=IMAGE_DIMENSION,num_frames=NUM_FRAMES):
    
    data = np.zeros((num_frames,image_dimension,image_dimension))
    
    dx = 2*np.pi/(image_dimension-1)
    dy = dx
    ticks = np.linspace(0, dt*num_frames, num_frames)

    for k in range(num_frames):
        t0 = ticks[k]
        for i in range(image_dimension):
            for j in range(image_dimension):
                x = dx*j
                y = (image_dimension-i)*dy
                r = np.array([x,y])
                data[k,i,j] = math.sin(np.dot(ADV_WAVE_VECTOR,r)-ADV_WAVE_FREQ*t0)

    return data

def animate(i,im,d):
    try:
        print(i)
        d0 = d[i,:,:]
    except:
        print("simultion ends")
        sys.exit()
    im.set_array(d0)
    im.set_cmap("gray")
    time.sleep(0.001)
    return i

def ShowAnim(data):
    fig = plt.figure()
    ax = plt.axes()
    im = plt.imshow(data[0,:,:], cmap = "gray")
    anim = animation.FuncAnimation(fig,animate,interval=2,fargs=(im,data))
    plt.show()

def SaveWave(data):
    a = ADV_WAVE_VECTOR
    theta = math.atan2(a[1],a[0])*180/np.pi # deg between the wave direction with x-axis (the direction of the moving wave)
    np.save(SAVE_PATH+str(int(theta))+".npy",data)
    print("generate wave save successfully: "+ SAVE_PATH+str(int(theta))+".npy")


def main():
    wave = create_adv_diff_wave()

    # Quick Visualization of the Generated Wave
    ShowAnim(wave)

    # Save Wave Data to output folder
    #SaveWave(wave)

if __name__ == "__main__":
    main()
