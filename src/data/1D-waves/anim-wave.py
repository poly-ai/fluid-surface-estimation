import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.animation as animation
import time

DATA1D_OUTPUT = "./output/wave-1.npy"

def animate(i,im,d):
    try:
        #print(i)
        d0 = d[i,:,:]
    except:
        print("simultion ends")
        sys.exit()
    im.set_array(d0)
    im.set_cmap("gray")
    time.sleep(0.001)
    return i

def ShowAnim(X):    
    data = np.load(X)
    fig = plt.figure()
    ax = plt.axes()
    im = plt.imshow(data[0,:,:], cmap = "gray")
    anim = animation.FuncAnimation(fig,animate,interval=2,fargs=(im,data))
    plt.show()

def main():
    ShowAnim(DATA1D_OUTPUT);

if __name__ == "__main__":
    main()
            
