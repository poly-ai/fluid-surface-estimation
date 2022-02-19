import numpy as np
import math
import matplotlib.pyplot as plt

from PIL import Image

def normal(x,xu,std):
    return math.exp(-0.5*((x-xu)/std)**2)/(std*(2*np.pi)**0.5)

def u(x,t):
    xu = np.pi
    xu1 = 1.2*np.pi
    xu2 = 0.8*np.pi
    std =  1*t+0.01    
    #return normal(x,xu,std)
    return normal(x,xu1,std)+normal(x,xu2,std)
    

def main():
    tres = 200
    xres = 100
    t = np.linspace(0,5,tres)
    x = np.linspace(0,2*np.pi,xres)
    U = np.zeros((tres,xres))

    for i in range(tres):
        t0 = t[i]
        for j in range(xres):
            x0 = x[j]
            U[i,j] = u(x0,t0)

    # Display Original 1D Advection Wave Data
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    print("Original Output U Data:")
    print(U)
    print("Shape of U:",U.shape)
    print("\n")

    # Extract (X,Y) Training Data, Nsample = 3
    Nsample = 3
    X_train = np.zeros((tres-Nsample,xres,xres,Nsample))
    Y_train = np.zeros((tres-Nsample,xres,xres))
    for i in range(tres-Nsample):
        for j in range(Nsample):
            U0 = U[i+j,:]
            U2d = np.tile(U0,(xres,1))
            X_train[i,:,:,j] = U2d
        Y_train[i,:,:] = np.tile(U[i+Nsample,:],(xres,1))

    np.save("output/x-train-dif-2.npy",X_train)
    np.save("output/y-train-dif-2.npy",Y_train)
    print("Training Data Processing Completed\n")

    # Visualize and Plot i-th Training Data (X(i),Y(i))
    i = 0 # time index
    X_train_i = X_train[i,:,:,:]
    Y_train_i = Y_train[i,:,:]
    for j in range(Nsample):
        print(X_train_i[:,:,j])
        #plt.imshow(X_train_i[:,:,j], cmap="gray")
        #plt.show()
    print(Y_train_i)
    plt.imshow(Y_train_i, cmap="gray")
    plt.show()

if __name__ == "__main__":
    main()
