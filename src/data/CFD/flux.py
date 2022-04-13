import numpy as np

# PURPOSE: This function calculates the flux for the shallow-water
# equations using the Roe flux function
#
# INPUTS:
#    UL: conservative state vector in the left cell
#    UR: conservative state vector in the right cell
#     n: unit normal pointing from the left cell to the right cell
#
# OUTPUTS:
#  F   : the flux out of the left cell (into the right cell)
#  smag: the maximum propagation speed of disturbances


#-----------------------------------------
def FluxFunction(UL, UR, n):

    # acceleration due to gravity
    g = 9.8

    # process left state
    hL = UL[0]
    if (hL<0): print('Non-physical state!')
    uL = UL[1]/hL
    vL = UL[2]/hL
    unL = uL*n[0] + vL*n[1]
    pL = 0.5*g*hL**2

    # left flux
    FL = np.zeros(4)
    FL[0] = hL*unL
    FL[1] = hL*uL*unL + pL*n[0]; 
    FL[2] = hL*vL*unL + pL*n[1];

    # process right state
    hR = UR[0]
    if (hR<0): print('Non-physical state!')
    uR = UR[1]/hR
    vR = UR[2]/hR
    unR = uR*n[0] + vR*n[1]
    pR = 0.5*g*hR**2

    # right flux
    FR = np.zeros(4)
    FR[0] = hR*unR
    FR[1] = hR*uR*unR + pR*n[0]; 
    FR[2] = hR*vR*unR + pR*n[1];

    # difference in states
    du = UR - UL

    # Roe average (almost)
    h = 0.5*(hL+hR)
    hu = 0.5*(hL*uL + hR*uR)
    hv = 0.5*(hL*vL + hR*vR)
    u = hu/h 
    v = hv/h
    un = u*n[0] + v*n[1]
    c = np.sqrt(g*h)
    
    # eigenvalues
    l = np.zeros(3)
    l[0] = un; l[1] = un-c; l[2] = un+c
    
    # entropy fix
    epsilon = c*.05
    for i in range(3):
        if ((l[i]<epsilon) and (l[i]>-epsilon)):
            l[i] = 0.5*(epsilon + l[i]*l[i]/epsilon)

    # absolute values of eigenvalues
    l = abs(l)

    # combination of eigenvalues
    s2 = 0.5*(l[1]-l[2])
    s3 = 0.5*(l[1]+l[2]-2*l[0])

    # eigenvetor product generator
    G1 = du[0]*un - du[1]*n[0] - du[2]*n[1]

    # functions of G1, s2, s3
    C1 = du[0]*s3 + G1*s2/c
    C2 = G1*s3 + s2*du[0]*c

    # flux assembly
    F = np.zeros(3)
    F[0] = 0.5*(FL[0]+FR[0])-0.5*(l[0]*du[0] + C1   )
    F[1] = 0.5*(FL[1]+FR[1])-0.5*(l[0]*du[1] + C1*u - C2*n[0])
    F[2] = 0.5*(FL[2]+FR[2])-0.5*(l[0]*du[2] + C1*v - C2*n[1])

    # max wave speed
    smag = max(l)

    # max wave speed
    smag = max(l)

    return F, smag

def main():
    UL = np.array([1,0,0]);
    UR = np.array([1,0,0]);
    n = np.array([-1/(np.sqrt(2)),-1/(np.sqrt(2))]);
    Flux, s = FluxFunction(UL,UR,n);
    print(Flux)
    dl = (0.05**2+0.05**2)**0.5;
    print(Flux*dl)

if __name__ == "__main__":
    main()

