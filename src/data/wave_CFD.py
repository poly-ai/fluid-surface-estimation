from itertools import combinations
from math import floor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
import matplotlib as mpl
import time
import sys
from scipy.interpolate import griddata

def animate(i,field,h,x,y):
    try:
        h0 = h[:,i]
    except:
        print("simultion ends")
        sys.exit()
    field.set_array(h0)
    field._offsets3d = (x,y,h0)
    time.sleep(0.001)
    return i

def ShowAnim(fig,field,h,x,y):
    anim = animation.FuncAnimation(fig,animate,interval=2,fargs=(field,h,x,y))
    plt.show()

def ExtractH(data):
    # Shallow Data Structue is [h0,u0,v0,h1,u1,v1, ..... hT,ut,vt]
    # Since we only use height data we extract height from columns with indexes (0,3,6,9...)
    array = data.to_numpy()
    T = int(array.shape[1]/3)
    h = np.zeros((array.shape[0],T))
    for i in range(T):
        h[:,i] = array[:,i*3]
    return h

def GenerateGrid(X, Y, H, unit):
    # X[i] = x coordinate, i = vertex index
    # Y[i] = y coordinate, i = vertex index
    # H[i, j] = height, i = vertex index, j = time
    # unit = grid size
    def area(x0, y0, x1, y1, x2, y2):
        return 0.5 * abs((x0 - x2) * (y1 - y0) - (x0 - x1) * (y2 - y0))

    def triangle(x, y):
        distances = [(x - X[i]) ** 2 + (y - Y[i]) ** 2 for i in range(X.shape[0])]
        indices = np.argsort(distances)

        for count in range(3, 10):
            for W in combinations(indices[0:count], 3):
                x0 = X[W[0]]
                y0 = Y[W[0]]
                h0 = H[W[0], :]
                x1 = X[W[1]]
                y1 = Y[W[1]]
                h1 = H[W[1], :]
                x2 = X[W[2]]
                y2 = Y[W[2]]
                h2 = H[W[2], :]

                a = area(x0, y0, x1, y1, x2, y2)
                a0 = area(x, y, x1, y1, x2, y2)
                a1 = area(x0, y0, x, y, x2, y2)
                a2 = area(x0, y0, x1, y1, x, y)

                if abs(a0 + a1 + a2 - a) < 1e-6:
                    return (a0 * h0 + a1 * h1 + a2 * h2) / a

        return 0

    result = np.zeros((floor(max(X) / unit), floor(max(Y) / unit), H.shape[1]))

    gX = []
    gY = []
    gH = []

    print(result.shape)
    for i in range(result.shape[0]):
        print(i)
        for j in range(result.shape[1]):
            result[i, j, :] = triangle(i * unit, j * unit)
            gX.append(i * unit)
            gY.append(j * unit)
            gH.append(result[i, j, 0])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.bar3d(gX, gY, np.zeros_like(gH), unit, unit, gH, shade=True)
    plt.savefig('output.png')

def main():
    data = pd.read_csv('output/Shallow_5.csv')
    xydata = pd.read_csv('output/XY.csv')
    Time   = pd.read_csv('output/SimTime_5.csv')

    xy = xydata.to_numpy()
    x = xy[:,0]
    y = xy[:,1]
    t = Time.to_numpy()
    t = t[:,1]
    print("Data Loaded Successfully")

    # Extract height from CFD output data
    h = ExtractH(data)

    # now h matrix contains height data at every space and every time.
    # with rows representing space   --> row i: the height data is stored at (x,y) Coordinates at (x[i], y[i])
    #   , where (x[i],y[i]) is the coordinate of the center of each triangular mesh cell.
    #   , (x,y) comes from tank0.gri mesh (coarse mesh)
    # with columns representing time --> col i: the height data is stored at time step t[i]
    # x,y domain: x (0,1.8), y(0,1.2)
    # time domain: 0.5 sec, dt = 1e-3 sec, t-length = 500;

    GenerateGrid(x, y, h, 0.01)

    # Animation
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    fig.set_figheight(15)
    fig.set_figwidth(30)
    field = ax.scatter(x,y,h[:,0])
    ShowAnim(fig,field,h,x,y)

if __name__ == "__main__":
    main()
