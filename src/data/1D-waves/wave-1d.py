import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image

DATA1D_OUTPUT = "./output/wave-sin-2.npy"
IMAGE_DIMENSION = 64
NUM_FRAMES = 1000
SIM_TIME = 5

# Weights for the Linear Combination of advecation(travelling) and diffusion(spreading) waves.
# W_ADV = 1, W_DIF = 0. represents pure travelling wave
# W_ADV = 0, W_DIF = 1. represents pure diffusion wave
W_ADV = 1 
W_DIF = 0

ADV_WAVE_NUMBER = 1
ADV_WAVE_FREQ = 1

def save_data(data, filename):
    np.save(filename, data)
    print(f"Saved data to {filename}")

def u_adv(x,t):
    k = ADV_WAVE_NUMBER
    w = ADV_WAVE_FREQ
    return math.sin(k*x-w*t)
    #return math.sin(k*x-w*t)+math.sin(3*k*x-3*w*t)/3+math.sin(5*k*x-5*w*t)/5 # Square Wave

def normal(x,xu,std):
    return math.exp(-0.5*((x-xu)/std)**2)/(std*(2*np.pi)**0.5)

def u_dif(x,t):
    xu = np.pi #xu1 = 1.2*np.pi; #xu2 = 0.8*np.pi
    std =  1*t+0.01
    return normal(x,xu,std)
    #return normal(x,xu1,std)+normal(x,xu2,std)

def u(x,t):
    return W_ADV*u_adv(x,t) + W_DIF*u_dif(x,t)

def main():
    tres = NUM_FRAMES
    xres = IMAGE_DIMENSION
    t = np.linspace(0,SIM_TIME,tres)
    x = np.linspace(0,2*np.pi,xres)
    U = np.zeros((tres,xres))

    for i in range(tres):
        t0 = t[i]
        for j in range(xres):
            x0 = x[j]
            U[i,j] = u(x0,t0)

    # Expand to 2D Height Matrix Data
    data = np.zeros((NUM_FRAMES,xres,xres));
    for i in range(tres):
        U0 = U[i,:]
        U2d = np.tile(U0,(xres,1))
        data[i,:,:] = U2d
    print("Shape of Data:",data.shape)

    save_data(data,DATA1D_OUTPUT); 
    print("Wave Data Ouput Complete\n")


if __name__ == "__main__":
    main()
