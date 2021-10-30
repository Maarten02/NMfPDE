import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.sparse as sp
import scipy.sparse.linalg as la
import math
import time
Lx = 4          # length in x direction
Ly = 4          # length in y direction
Nx = 100        # number of intervals in x-direction
Ny = 100        # number of intervals in y-direction
dx = Lx/Nx      # grid step in x-direction
dy = Ly/Ny      # grid step in y-direction
Tend = 20       # End time of simulation
Nt = 50040      # Time steps
dt = Tend/Nt    # delta t

Du = 0.05       # Parameter in the du/dt equation
Dv = 1.0        # Parameter in the dv/dt equation
k = 2           # Parameter in both equations
a = 0.1305      # parameter in dv/dt equation
b = 0.7695      # parameter in du/dt equation

def getPerturbator(Nx, Ny, a=0.1305, b=0.7695):
    # Return grid with noise values
    return 0.01*(a+b)*np.random.rand(Ny+1,Nx+1)

# Don't know if this function is needed
def createF(x, y, t, funcName):
    Fvec = np.empty(((Ny+1)*(Nx+1)))
    for j in range(Ny+1):
        for i in range(Nx+1):
            Fvec[j*(Nx+1)+i] = funcName(x[i,j], y[i,j], t)
    return Fvec


def getGridAndInitialConditions(func):
    # Create grid
    x,y = np.mgrid[0:Lx+dx:dx, 0:Ly+dy:dy]

    # Create vector with noise
    rgrid = func(Nx, Ny)
    rvec = rgrid.flatten()

    # Create initial conditions
    u0 = (a + b) * np.ones((Ny + 1) * (Nx + 1)) + rvec
    v0 = b / (a+b) ** 2 * np.ones((Ny+1)*(Nx+1))

    return x, y, u0, v0

def getSystemMatrix():

    # Create first derivative matrices
    Dx = sp.diags([1,-1], offsets=[0,-1], shape=(Nx+2,Nx+1), format='csr')/dx
    Dy = sp.diags([1,-1], offsets=[0,-1], shape=(Ny+2,Ny+1), format='csr')/dy

    pos = [[0,0],[0,1],[-1,-1],[-1,-2]]
    for en in pos:
        Dx[en[0], en[1]] = 0
        Dy[en[0], en[1]] = 0

    # Create the one dimensional second derivative matrices
    Lxx = Dx.transpose().dot(Dx)
    Lyy = Dy.transpose().dot(Dy)

    # Get identity matrices
    Ix = sp.eye(Nx+1)
    Iy = sp.eye(Ny+1)

    # Create the system matrix for the laplacian with Neumann boundary conditions
    A = sp.kron(Iy,Lxx) + sp.kron(Lyy,Ix)
    return A


def simulateReactionFE(u0, v0):

    soln = np.empty(4,(Nx+1)*(Ny+1))

    u = soln[0] = u0
    v = soln[1] = v0
    A = getSystemMatrix()

    for i in range(Nt):
        u += dt * (-Du * A.dot(u) + k * (a - u + u * u * v))
        v += dt * (-Dv * A.dot(v) + k * (b - u * u * v))

    soln[2] = u
    soln[3] = v
    return soln


def NewtonRaphson(wk, k, limit, limited):
    A = getSystemMatrix()
    wkp1 = wk
    itr = 0
    diff = 1
    while diff > 10**-3:

        itr += 1
        if limited:
            if itr == 3:
                print('Iteration limit reached with error norm ', diff)
                limit = True
                break
        u, v = np.split(wkp1, 2)

        UL = -Du*A - k*sp.eye((Ny+1)*(Nx+1)) + 2*k*sp.diags(u*v)
        UR = k*sp.diags(u*u)
        LL = -2*k*sp.diags(u*v)
        LR = -Dv*A - k*sp.diags(u*u)
        Jacobi = sp.bmat([[UL,UR],[LL,LR]])

        mat = sp.eye(2*(Ny+1)*(Nx+1)) - dt*Jacobi

        fu = (-Du * A.dot(u) + k * (a - u + u * u * v))
        fv = (-Dv * A.dot(v) + k * (b - u * u * v))

        vec = wk + dt * np.concatenate((fu,fv)) - wkp1
        pkp1 = la.spsolve(mat, vec)
        wkp1 += pkp1
        diff = np.linalg.norm(pkp1)

        print('Newton Raphson iteration ', itr, ' error norm: ', diff)
    print('converged after ', itr, 'iterations')
    return limit, wkp1

def simulateReactionBE(u0, v0, N, k, limited):
    soln = np.empty((4,(Nx+1)*(Ny+1)))
    soln[0] = u0
    soln[1] = v0
    wkp1 = np.concatenate((u0,v0))
    limit = False

    for i in range(N):
        limit, wkp1 = NewtonRaphson(wkp1, k, limit, limited)
        print('Timestep ', i+1,'of', N, ' completed.')
        print('-----------------')
        if limit:
            print('shits broken')
            break
    soln[2], soln[3] = np.split(wkp1, 2)
    return limit, soln


def deBuggr():
    x, y, u0, v0 = getGridAndInitialConditions(getPerturbator)
    limited = False
    k = 2
    limit, soln = simulateReactionBE(u0, v0, 20, k, limited)
    size = (Ny+1, Nx+1)
    plt.subplot(2, 2, 1)
    plt.imshow(np.reshape(soln[0], size), origin="lower", extent=((x[0, 0], x[-1, -1], y[0, 0], y[-1, -1])))
    plt.title('u start')
    plt.colorbar(orientation='horizontal')
    plt.subplot(2, 2, 2)
    plt.imshow(np.reshape(soln[1], size), origin="lower", extent=((x[0, 0], x[-1, -1], y[0, 0], y[-1, -1])))
    plt.title('v start')
    plt.colorbar(orientation='horizontal')
    plt.subplot(2, 2, 3)
    plt.imshow(np.reshape(soln[2], size), origin="lower", extent=((x[0, 0], x[-1, -1], y[0, 0], y[-1, -1])))
    plt.title('u T = 20')
    plt.colorbar(orientation='horizontal')
    plt.subplot(2, 2, 4)
    plt.imshow(np.reshape(soln[3], size), origin="lower", extent=((x[0, 0], x[-1, -1], y[0, 0], y[-1, -1])))
    plt.title('v T = 20')
    plt.colorbar(orientation='horizontal')
    plt.show()

deBuggr()

def getBENRsteps(k, left=1, right=100):

    x, y, u0, v0 = getGridAndInitialConditions(getPerturbator)
    BEtime_steps = 0
    Running = True
    it = 1

    while Running:

        print("Bisection iteration ", it," started")
        print("left = ", left, "right = ", right)
        middle = int((left + right)*0.5)

        limit, soln = simulateReactionBE(u0, v0, middle, k)

        if right - left == 1 or right == left:
            Running = False
            limit, soln = simulateReactionBE(u0, v0, right, k)
            if not limit:
                BEtime_steps = right
            else:
                BEtime_steps = right + 1
        elif not limit:
            right = middle
        else:
            left = middle
        print("Bisection iteration", it, "completed")
        it += 1
    return BEtime_steps



#=====          ANIMATION CODE            ======

