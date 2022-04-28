import numpy as np
import matplotlib.pyplot as plt
from .readgri import readgri
from .flux import FluxFunction
import pandas as pd

# USER SPECIFY
#############################################################
LABEL = "fine"  # Label for the simulation
SIM_TIME = 1  # The total simulation time (> 500 frames)
# NOTE dt = 0.001 sec


# Define the Initial Distribution of the Water Height
def initial_height(x,y,x_center,y_center,x_distri,y_distri,height_level,height_delta):
    h0 = height_level+ height_delta*np.exp((-x_distri*(x-x_center)**2)-(y_distri*(y-y_center)**2));
    #h0 = 1.0+0.3*np.exp((-50*(x-0.5)**2)-(50*(y-0.5)**2));

    # This setup will generate an avg height with 1.0
    # A single peak 1.3 height at (x=1.3,y=0.9)
    # It is similar to the concept of guassian distribution
    # NOTE the value of (0.3) should be positive due to the limit of
    # shallow water equation

    # Adding h1 and return h0 + h1, it will be a two peaks distribution
    # h1 = 1.0+0.3*np.exp((-50*(x-0.25)**2)-(50*(y-0.5)**2));

    return h0


##############################################################


def AreaCell(C, V):
    x1 = V[C[0], 0]
    y1 = V[C[0], 1]
    x2 = V[C[1], 0]
    y2 = V[C[1], 1]
    x3 = V[C[2], 0]
    y3 = V[C[2], 1]
    Ai = 0.5 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    return Ai


def dLength(E, V):
    index1 = E[0]
    index2 = E[1]
    x1 = V[index1, 0]
    y1 = V[index1, 1]
    x2 = V[index2, 0]
    y2 = V[index2, 1]
    length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return length


def WallFlux(R, Lcell, u, n0, dl, E0, Forces):
    g = 9.8
    Lu = u[Lcell]
    h = Lu[0]
    R[Lcell, 0] += dl * 0
    R[Lcell, 1] += dl * n0[0] * 0.5 * g * h**2
    R[Lcell, 2] += dl * n0[1] * 0.5 * g * h**2
    index = E0[3]
    if index > 0:
        Forces[2 * index - 2] += dl * n0[0] * 0.5 * g * h**2
        Forces[2 * index - 1] += dl * n0[1] * 0.5 * g * h**2


def findnorm(V, edge):
    # normal point from left to right
    pA = V[edge[0]]
    pB = V[edge[1]]
    dl = np.sqrt((pA[0] - pB[0]) ** 2 + (pA[1] - pB[1]) ** 2)
    n = np.array([pB[1] - pA[1], pA[0] - pB[0]]) / dl
    return n


def Centeroid(V, C, x_center,y_center,x_distri,y_distri,height_level,height_delta):
    NC = np.shape(C)[0]
    Cent = np.zeros((NC, 2))
    u = np.zeros((NC, 3))
    for i in range(NC):
        C0 = C[i]
        x1 = V[C0[0], 0]
        y1 = V[C0[0], 1]
        x2 = V[C0[1], 0]
        y2 = V[C0[1], 1]
        x3 = V[C0[2], 0]
        y3 = V[C0[2], 1]
        x = (x1 + x2 + x3) / 3
        y = (y1 + y2 + y3) / 3
        Cent[i, :] = np.array([x, y])

        #######################################################################
        # Calls the initial_height function to determine the
        # inital distribution of the water height
        h0 = initial_height(x,y,
                            x_center,y_center,x_distri,y_distri,height_level,height_delta)
        #######################################################################

        vx0 = 0
        vy0 = 0
        u[i, :] = np.array([h0, vx0, vy0])

    return Cent, u


def WriteFile(label, df1, df2):
    print("Writing label: " + label + " .csv files to output/")
    tx_h = "data/raw/CFD/Shallow_" + label + ".csv"
    tx_t = "data/raw/CFD/SimTime_" + label + ".csv"

    print("Height Data\n", df1)
    df1.to_csv(tx_h, sep=",", index=None)

    print("Simulation Time\n", df2)
    df2.to_csv(tx_t, sep=",")

    print("Data : " + label + " .csv files to data/raw/CFD")


def plotline(pt1, pt2):
    x = [pt1[0], pt2[0]]
    y = [pt1[1], pt2[1]]
    plt.plot(x, y)


def run_cfd(grifile,x_center=1.3,y_center=0.9,x_distri=50,y_distri=50,height_level=1,height_delta=0.3):
    # Use Fine grid: tank1.gri
    Mesh = readgri(grifile)

    V = Mesh["V"]
    C = Mesh["E"]
    NC = np.shape(C)[0]

    # Centeroid of Cell and Initial States
    Cent, u = Centeroid(V, C, x_center,y_center,x_distri,y_distri,height_level,height_delta)
    # u = np.ones((NC,3)); #for test
    # u[:,1] = 0;
    # u[:,2] = 0;

    df = pd.DataFrame(u)

    df2 = pd.DataFrame(Cent, columns=["x", "y"])

    df4 = pd.DataFrame()

    Ai = np.zeros(NC)
    for i in range(NC):
        Ai[i] = AreaCell(C[i], V)

    t = np.zeros(1)
    df4 = df4.append(pd.DataFrame(t))

    it = 0
    # Nt = 100
    Tsim = 0.5
    # rescheck = np.zeros(1)
    # df5 = pd.DataFrame();
    # df5 = df5.append(pd.DataFrame(rescheck));

    # Start Propogating with Time
    # while(it<=2):
    while t <= Tsim:
#        print("\niter-----------------------")
#        print(it)

        R = np.zeros((NC, 3))
        Forces = np.zeros((6, 1))
        tisum = np.zeros(NC)

        # Get BE normal
        BE = Mesh["BE"]

        for i in range(np.shape(BE)[0]):
            E0 = BE[i]
            n0 = findnorm(V, E0)
            c0 = 0.5 * (V[E0[0], 0:2] + V[E0[1], 0:2])
            dl = dLength(E0, V)
            LCell = E0[2]
            WallFlux(R, LCell, u, n0, dl, E0, Forces)

            # Wall wave speed
            u0 = u[LCell]
            c0 = np.sqrt(9.8 * u0[0])
            vx = u0[1] / u0[0]
            vy = u0[2] / u0[0]
            wavespeed = np.abs(vx * n0[0] + vy * n0[1]) + c0
            tisum[LCell] += wavespeed * dl
            # plotline(c0,c0+0.05*n0)

        # Get IE normal
        IE = Mesh["IE"]
        for i in range(np.shape(IE)[0]):
            E0 = IE[i]
            n0 = findnorm(V, E0)
            c0 = 0.5 * (V[E0[0], 0:2] + V[E0[1], 0:2])
            dl = dLength(E0, V)
            LCell = E0[2]
            RCell = E0[3]
            uL = u[LCell]
            uR = u[RCell]
            Flux, smag = FluxFunction(uL, uR, n0)
            R[LCell] += dl * Flux
            R[RCell] -= dl * Flux
            tisum[LCell] += smag * dl
            tisum[RCell] += smag * dl

        # Residual Done
        # print("Resid Check");
        # rescheck[0] = LA.norm(R);
        # print(rescheck);
        # print("Height Check")
        # print(np.max(u[:,0]))
        # print("Forces")
        # print(Forces)

        # update states
        # dti = np.divide(2*Ai,tisum);
        # dtg = np.min(dti)*0.9;
        dtg = 0.001
        # print("dt global [sec]")
        # print(dtg)
        for i in range(NC):
            u[i, :] = u[i, :] - dtg * R[i, :] / Ai[i]

        t[0] += dtg
        # Store unknowns
        df = pd.concat([df, pd.DataFrame(u)], axis=1)
        # Store Forces
        # if it == 0:
        #    print("Got you!!!!")
        #    df3 = pd.concat([df3,pd.DataFrame(Forces)], axis = 1);
        #    df3 = pd.concat([df3,pd.DataFrame(Forces)], axis = 1);
        # else:
        #    df3 = pd.concat([df3,pd.DataFrame(Forces)], axis = 1);
        # Store Sim. Time
        df4 = df4.append(pd.DataFrame(t))
        # df5 = df5.append(pd.DataFrame(rescheck));

#        print("current time")
#        print(t[0])
        it += 1

    print("\n##### DONE #####\n")

    return df.to_numpy(), df2.to_numpy(), df4.to_numpy()
