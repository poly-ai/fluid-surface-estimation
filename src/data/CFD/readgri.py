import numpy as np
from scipy import sparse


# -----------------------------------------------------------
# Identifies interior and boundary edges given element-to-node
# IE contains (n1, n2, elem1, elem2) for each interior edge
# BE contains (n1, n2, elem, bgroup) for each boundary edge
def edgehash(E, B):
    Ne = E.shape[0]
    Nn = np.amax(E) + 1
    H = sparse.lil_matrix((Nn, Nn), dtype=np.int)
    IE = np.zeros([int(np.ceil(Ne * 1.5)), 4], dtype=np.int)
    ni = 0
    for e in range(Ne):
        for i in range(3):
            n1, n2 = E[e, i], E[e, (i + 1) % 3]
            if H[n2, n1] == 0:
                H[n1, n2] = e + 1
            else:
                eR = H[n2, n1] - 1
                IE[ni, :] = n1, n2, e, eR
                H[n2, n1] = 0
                ni += 1
    IE = IE[0:ni, :]
    # boundaries
    nb0 = nb = 0
    for g in range(len(B)):
        nb0 += B[g].shape[0]
    BE = np.zeros([nb0, 4], dtype=np.int)
    for g in range(len(B)):
        Bi = B[g]
        for b in range(Bi.shape[0]):
            n1, n2 = Bi[b, 0], Bi[b, 1]
            if H[n1, n2] == 0:
                n1, n2 = n2, n1
            BE[nb, :] = n1, n2, H[n1, n2] - 1, g
            nb += 1
    return IE, BE


# -----------------------------------------------------------
def readgri(fname):
    f = open(fname, "r")
    Nn, Ne, dim = [int(s) for s in f.readline().split()]
    # read vertices
    V = np.array([[float(s) for s in f.readline().split()] for n in range(Nn)])
    # read boundaries
    NB = int(f.readline())
    B = []
    Bname = []
    for i in range(NB):
        s = f.readline().split()
        Nb = int(s[0])
        Bname.append(s[2])
        Bi = np.array([[int(s) - 1 for s in f.readline().split()] for n in range(Nb)])
        B.append(Bi)
    # read elements
    Ne0 = 0
    E = []
    while Ne0 < Ne:
        s = f.readline().split()
        ne = int(s[0])
        Ei = np.array([[int(s) - 1 for s in f.readline().split()] for n in range(ne)])
        E = Ei if (Ne0 == 0) else np.concatenate((E, Ei), axis=0)
        Ne0 += ne
    f.close()
    # make IE, BE structures
    IE, BE = edgehash(E, B)
    Mesh = {"V": V, "E": E, "IE": IE, "BE": BE, "Bname": Bname}
    return Mesh


# -----------------------------------------------------------
def writegri(Mesh, fname):
    V = Mesh["V"]
    E = Mesh["E"]
    BE = Mesh["BE"]
    Bname = Mesh["Bname"]
    Nv, Ne, Nb = V.shape[0], E.shape[0], BE.shape[0]
    f = open(fname, "w")
    f.write("%d %d 2\n" % (Nv, Ne))
    for i in range(Nv):
        f.write("%.15e %.15e\n" % (V[i, 0], V[i, 1]))
    nbg = 0
    for i in range(Nb):
        nbg = max(nbg, BE[i, 3])
    nbg += 1
    f.write("%d\n" % (nbg))
    for g in range(nbg):
        nb = 0
        for i in range(Nb):
            nb += BE[i, 3] == g
        f.write("%d 2 %s\n" % (nb, Bname[g]))
        for i in range(Nb):
            if BE[i, 3] == g:
                f.write("%d %d\n" % (BE[i, 0] + 1, BE[i, 1] + 1))
    f.write("%d 1 TriLagrange\n" % (Ne))
    for i in range(Ne):
        f.write("%d %d %d\n" % (E[i, 0] + 1, E[i, 1] + 1, E[i, 2] + 1))
    f.close()


# -----------------------------------------------------------
def main():
    _ = readgri("Mesh/tank0.gri")
    # writegri(Mesh, 'Mesh/tank1.gri')


if __name__ == "__main__":
    main()
