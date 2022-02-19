import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.animation as animation
import time

def animate(i,im,X_train):
    try:
        print(i)
        X0 = X_train[i,:,:,0]
        #print(X0)
    except:
        print("simultion ends")
        sys.exit()
    im.set_array(X0)
    im.set_cmap("gray")
    time.sleep(0.001)
    return i

def ShowAnim(X,Y):    
    X_train = np.load(X)
    Y_train = np.load(X)
    fig = plt.figure()
    ax = plt.axes()
    im = plt.imshow(X_train[0,:,:,0], cmap = "gray")
    anim = animation.FuncAnimation(fig,animate,interval=2,fargs=(im,X_train))
    plt.show()

def main():
    X = "output/x-train-adv.npy"
    Y = "output/y-train-adv.npy"
    ShowAnim(X,Y)

if __name__ == "__main__":
    main()
            
