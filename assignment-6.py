import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.sparse as sp
import scipy.sparse.linalg as la
import math
import time
import os

Lx = 4          # length in x direction
Ly = 4          # length in y direction
Nx = 100        # number of intervals in x-direction
Ny = 100        # number of intervals in y-direction
dx = Lx/Nx      # grid step in x-direction
dy = Ly/Ny      # grid step in y-direction
Tend = 20       # End time of simulation
#NtFE = 50040      # Time steps
#dtFE = Tend/NtFE    # delta t

Du = 0.05       # Parameter in the du/dt equation
Dv = 1.0        # Parameter in the dv/dt equation
#k = 2           # Parameter in both equations
a = 0.1305      # parameter in dv/dt equation
b = 0.7695      # parameter in du/dt equation

def getPerturbator(Nx, Ny, a=0.1305, b=0.7695):
    # Return grid with noise values
    np.random.seed(6942)
    return 0.01*(a+b)*np.random.rand(Ny+1,Nx+1)

def getInitialConditions(func):
    # Create vector with noise
    rgrid = func(Nx, Ny)
    rvec = rgrid.flatten()

    # Create initial conditions
    u0 = (a + b) * np.ones((Ny + 1) * (Nx + 1)) + rvec
    v0 = b / (a+b) ** 2 * np.ones((Ny+1)*(Nx+1))

    return u0, v0

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

def getFEtimesteps(k):
    # Calculate forward Euler timesteps for stability
    NtFE = math.ceil((Tend*(8+2*k*dx**2))/(2*dx**2))
    return NtFE

def simulateReactionFE(u0, v0, NtFE, k):
    dt = Tend / NtFE

    # Initialize initial and final value storage array
    soln = np.empty((4,(Nx+1)*(Ny+1)))

    u = soln[0] = u0
    v = soln[1] = v0
    A = getSystemMatrix()

    # step using forward Euler till t=Tend
    for i in range(NtFE):
        u += dt * (-Du * A.dot(u) + k * (a - u + u * u * v))
        v += dt * (-Dv * A.dot(v) + k * (b - u * u * v))

    soln[2] = u
    soln[3] = v
    return soln


def NewtonRaphson(wk, k, limit, limited, dt, A):
    wkp1 = np.copy(wk)
    itr = 0
    diff = 1
    while diff > 10**-3:

        itr += 1

        # if an iteration limit is set, check if the iteration limit is reached
        if limited:
            if itr == 4:
                limit = True
                break
        u, v = np.split(wkp1, 2)

        # Calculate the Jacobi matrix for latest u and v
        UL = -Du*A - k*sp.eye((Ny+1)*(Nx+1)) + 2*k*sp.diags(u*v)
        UR = k*sp.diags(u*u)
        LL = -2*k*sp.diags(u*v)
        LR = -Dv*A - k*sp.diags(u*u)
        Jacobi = sp.bmat([[UL,UR],[LL,LR]])

        # Evaluate the vector functions in the DE
        fu = (-Du * A.dot(u) + k * (a - u + u * u * v))
        fv = (-Dv * A.dot(v) + k * (b - u * u * v))

        # Calculate the matrix for NR
        mat = sp.eye(2 * (Ny + 1) * (Nx + 1)) - dt * Jacobi

        # Calculate the vector for NR - solve the system - update the solution
        vec = wk + dt * np.concatenate((fu,fv)) - wkp1
        pkp1 = la.spsolve(mat, vec)
        wkp1 += pkp1

        # Calculate the error norm
        diff = np.linalg.norm(vec)

        print('Newton Raphson iteration ', itr, ' error norm: ', diff)
    #print('converged after ', itr, 'iterations')
    return limit, wkp1

def simulateReactionBE(u0, v0, N, k, limited):
    # Get the spatial discrisation - get timestep
    sysm = getSystemMatrix()
    dt = Tend / N

    # Initialize the solution storage array
    soln = np.empty((4,(Nx+1)*(Ny+1)))
    soln[0] = u0
    soln[1] = v0

    # concatenate the u and v vectors
    wkp1 = np.concatenate((u0,v0))

    # Start the NR iterations
    limit = False

    for i in range(N):
        # Time march to T=20
        limit, wkp1 = NewtonRaphson(wkp1, k, limit, limited, dt, sysm)

        print('Timestep ', i+1,'of', N, ' completed.')
        print('-----------------')

        # break the marching when NR iteration limit reached
        if limit:
            print('Not converged for Nt = ', N)
            break

    # Store solution at T=20
    soln[2], soln[3] = np.split(wkp1, 2)

    # Report back if time marching was succesful and the solution
    return limit, soln


def PlotSoln(k):
    # Function for plotting the intial and final timestep solution values
    u0, v0 = getInitialConditions(getPerturbator)

    # Timesteps already determined for this function: don't need to limit NR iterations
    limited = False

    # Get timesteps for BE and FE
    NtBE = 242
    NtFE = getFEtimesteps(k)

    t1 = time.time()
    # Execute either FE or BE
    #limit, soln = simulateReactionBE(u0, v0, NtBE, k, limited)
    soln = simulateReactionFE(u0, v0, NtFE, k)
    t2 = time.time()

    print("FE with Nt = ", NtFE, " solved in ", "{:.2f}".format(t2 - t1), " s")

    # Check if folder exists for storing the images. If not, create one
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Results/')
    file_name = "FE_"+str(k)+"_"+str(NtFE)+"_3"
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # Plot both u and v at t=0 and t=20
    size = (Ny + 1, Nx + 1)
    titleBE = 'k = '+str(k)+' Nt = '+str(NtBE)+' FE it. limit = 3'
    titleFE = 'k = '+str(k)+' Nt = '+str(NtFE)+' FE'
    plt.title(titleFE)
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
    plt.suptitle(titleFE)

    # Store the generated image
    plt.savefig(results_dir + file_name)

# Uncomment to plot the solutions for different k
#l = [2,5,10]
#for i in l:
#    PlotSoln(i)

def getBENRsteps(k, left=20, right=250):
    # Bisection algorithm to determine BE timesteps
    # Get initial conditions - Initialisatise variables
    u0, v0 = getInitialConditions(getPerturbator)
    BEtime_steps = 0
    proceed = False
    Running = True
    it = 1

    # Verify if the timesteps which satisfies max 3 NR iterations within specified interval
    limitLeft, soln = simulateReactionBE(u0, v0, left, k, True)
    limitRight, soln = simulateReactionBE(u0, v0, right, k, True)

    if not limitRight and limitLeft:
        print('Nt within specified interval')
        proceed = True
    else:
        print('No Nt within specified interval ')
        print(limitLeft, limitRight)

    # Execute the bisection algorithm (if Nt within specified interval)
    if proceed:
        while Running:

            print("Bisection iteration ", it," started")
            print("left = ", left, "right = ", right)
            middle = int((left + right)*0.5)

            # See if middle value of interval satisfies the iteration limit - report calculation time
            t1 = time.time()
            limit, soln = simulateReactionBE(u0, v0, middle, k, True)
            t2 = time.time()
            print("BENR with Nt = ", middle, " solved in ", "{:.2f}".format(t2 - t1), " s")

            # Check if interval is 1 or 0 - finish the algorithm and store result
            if right - left == 1 or right == left:
                Running = False
                limit, soln = simulateReactionBE(u0, v0, right, k, True)
                if not limit:
                    BEtime_steps = right
                else:
                    BEtime_steps = right + 1

            # Update interval
            elif not limit:
                right = middle
            else:
                left = middle
            print("-----------")
            it += 1

        # Measure execution time for the found Nt
        t3 = time.time()
        limit, soln = simulateReactionBE(u0, v0, BEtime_steps, k , False)
        t4 = time.time()
        print(">>BENR with Nt = ", BEtime_steps, " solved in ", "{:.2f}".format(t4 - t3), " s")

        return BEtime_steps
    else:
        return -1

#=====       FORWARD EULER: RESULTS       =====
# k=2  --> Nt= 50040 [solved in  10.08  s]
# k=5  --> Nt= 50100 [solved in  9.88  s]
# k=10 --> Nt= 50200 [solved in  10.43  s]

# Determine BE timesteps for the different values of k
kl = [2,5,10]
for i in range(kl):
    getBENRsteps(i)

#=====      BENR: BISECTION RESULTS        =====
#k =   2  Nt =   45  execution time =  24.25  s
#k =   5  Nt =  125  execution time =  67.71  s
#k =  10  Nt =  242  execution time = 124.55  s








