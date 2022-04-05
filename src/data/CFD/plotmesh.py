from concurrent.futures import process
from sys import argv
import numpy as np
import matplotlib.pyplot as plt
from readgri import readgri

#-----------------------------------------------------------
def plotmesh(Mesh, fname):
    V = Mesh['V']; E = Mesh['E']; BE = Mesh['BE']
    f = plt.figure(figsize=(12,12))
    plt.triplot(V[:,0], V[:,1], E, 'k-')
    for i in range(BE.shape[0]):
       plt.plot(V[BE[i,0:2],0],V[BE[i,0:2],1], '-', linewidth=2, color='blue')
    #dosave = not not fname
    plt.axis('equal')
    plt.show()
    plt.savefig('mesh.png')
    plt.close(f)

#-----------------------------------------------------------
def main():
    filename = 'tank1.gri' if len(argv) <= 1 else argv[1]

    print("Fine Mesh Plot")
    # Use readgri fcn to read the mesh data
    Mesh = readgri(filename)

    # Verticies Coordinate [x.y]
    V = Mesh['V'];
    print("Verticies Coordinate [x,y]:\n",V);
    print("\n");

    print(np.max(V[:,1]));

    # Triangular Mesh Verticies Index [v1, v2, v3]
    E = Mesh['E'];
    print("Triangular Mesh Verticies Index:\n",E);
    print("number of mesh cell: ",E.shape[0]);

    # Use plotmesh fcn to visuallize the mesh (with verticies and edges)
    plotmesh(Mesh, [])


if __name__ == "__main__":
    main()
