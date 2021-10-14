import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la

Lx = 10  # length in x direction
Ly = 5  # length in y direction
Nx = 10  # number of intervals in x-direction
Ny = 10  # number of intervals in y-direction
dx = Lx/Nx  # grid step in x-direction
dy = Ly/Ny  # grid step in y-direction

def createF(x, y,funcName):
    Fvec = np.empty(((Ny-1)*(Nx-1)))
    for j in range(Ny-1):
        for i in range(Nx-1):
            Fvec[j*(Nx-1)+i] = funcName(x[i,j], y[i,j])
    return Fvec