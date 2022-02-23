import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image

X_TRAIN_OUTPUT = "./output/x-train-adv.npy"
Y_TRAIN_OUTPUT = "./output/y-train-adv.npy"
IMAGE_DIMENSION = 20
NUM_FRAMES_PER_EXAMPLE = 60

def save_data(data, filename):
    np.save(filename, data)
    print(f"Saved data to {filename}")

def u(x,t):
    k = 1
    w = 1
    return math.sin(k*x-w*t)
    #return math.sin(k*x-w*t)+math.sin(3*k*x-3*w*t)/3+math.sin(5*k*x-5*w*t)/5 # Square Wave

def main():
    tres = 200
    xres = IMAGE_DIMENSION
    t = np.linspace(0,5,tres)
    x = np.linspace(0,2*np.pi,xres)
    U = np.zeros((tres,xres))

    for i in range(tres):
        t0 = t[i]
        for j in range(xres):
            x0 = x[j]
            U[i,j] = u(x0,t0)

    # Display Original 1D Advection Wave Data
    print("Shape of U:",U.shape)

    # Extract (X,Y) Training Data, Nsample = 3
    X_train = np.zeros((tres - NUM_FRAMES_PER_EXAMPLE, xres, xres, NUM_FRAMES_PER_EXAMPLE))
    Y_train = np.zeros((tres - NUM_FRAMES_PER_EXAMPLE, xres, xres))
    for i in range(tres - NUM_FRAMES_PER_EXAMPLE):
        for j in range(NUM_FRAMES_PER_EXAMPLE):
            U0 = U[i+j,:]
            U2d = np.tile(U0,(xres,1))
            X_train[i,:,:,j] = U2d
        Y_train[i,:,:] = np.tile(U[i+NUM_FRAMES_PER_EXAMPLE,:],(xres,1))

    save_data(X_train, X_TRAIN_OUTPUT)
    save_data(Y_train, Y_TRAIN_OUTPUT)
    print("Training Data Processing Completed\n")


if __name__ == "__main__":
    main()
