import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la
import math
import time
Lx = 10  # length in x direction
Ly = 5  # length in y direction
Nx = 200  # number of intervals in x-direction
Ny = 100  # number of intervals in y-direction
dx = Lx/Nx  # grid step in x-direction
dy = Ly/Ny  # grid step in y-direction

def coeffK(x,y):
    if (x - 7)**2 + (y-2.5)**2 <= 1.25**2:
        return 0.1
    else:
        return 1.0


def sourceF(x,y,alpha=40):
    f = math.exp(-alpha*(x-3)**2-alpha*(y-2.5)**2) + math.exp(-alpha*(x-7)**2-alpha*(y-2.5)**2)
    return f

def createF(x, y,funcName):
    Fvec = np.empty(((Ny-1)*(Nx-1)))
    for j in range(Ny-1):
        for i in range(Nx-1):
            Fvec[j*(Nx-1)+i] = funcName(x[i,j], y[i,j])
    return Fvec

def notBoundary(i,j):
    # Check if passed gridpoint is a boundary point
    # nBp: notBoundaryPoint

    nBp = True
    if i == -1 or i == Nx-1 or j == -1 or j == Ny-1:
        nBp = False

    return nBp

def entry(coeffun,cnt, x, y):
    if cnt == 0:
        # calculate entry for node (i,j-1)
        entryVal = -coeffun(x,y - dy * 0.5) / dy ** 2
    elif cnt == 1:
        # calculate entry for node (i-1,j)
        entryVal = -coeffun(x - dx * 0.5, y) / dx ** 2
    elif cnt == 2:
        # calculate entry for node (i,j)
        entryVal = (coeffun(x,y - dy * 0.5) + coeffun(x, y + dy * 0.5)) / dy ** 2 + (coeffun(x - dx * 0.5, y) + coeffun(x + dx * 0.5, y)) / dx ** 2
    elif cnt == 3:
        # calculate entry for node (i+1,j)
        entryVal = -coeffun(x + dx * 0.5, y) / dx ** 2
    elif cnt == 4:
        # calculate entry for node (i,j+1)
        entryVal = -coeffun(x, y + dy * 0.5) / dy ** 2

    return entryVal

def create2DLFVM(x,y,coeffFun):
    diags = [[],[],[],[],[]]
    for j in range(0,Ny-1):
        for i in range(0,Nx-1):

            x_coor = x[i,j]
            y_coor = y[i,j]
            diag_points = [[i,j-1],[i-1,j],[i,j],[i+1,j],[i,j+1]]

            for cnt in range(len(diag_points)):
                    nBp = notBoundary(diag_points[cnt][0],diag_points[cnt][1])
                    if nBp:
                        diags[cnt].append(entry(coeffFun, cnt, x_coor, y_coor))
                    else:
                        diags[cnt].append(0)

    A = sp.diags([diags[0][Nx-1:],diags[1][1:],diags[2],diags[3][:-1],diags[4][:-Nx+1]], [-Nx+1, -1, 0 , 1, Nx-1], format='csc')
    return A

def solveFE(uStart, tStart, tEnd, Nt, x, y):
    uloc = []
    dt = (tEnd - tStart) / Nt
    uk = uStart
    A = create2DLFVM(x,y,coeffK)
    f = createF(x,y, sourceF)
    for k in range(Nt):
        ukp1 = uk + dt*(-1*A.dot(uk) + f)
        uk = ukp1
        uloc.append(uk[int((Nx-1)*(Ny-1)*0.5)])
    uEnd = uk
    return uEnd, np.array(uloc)

def solveBE(uStart,tStart,tEnd,Nt, x, y):
    uloc = []
    dt = (tEnd - tStart) / Nt
    uk = uStart
    A = create2DLFVM(x,y,coeffK)
    f = createF(x,y, sourceF)
    for k in range(Nt):
        mat = sp.eye((Nx-1)*(Ny-1)) + dt*A
        vec = uk + dt*f
        ukp1 = la.spsolve(mat,vec)
        uk = ukp1
        uloc.append(uk[int((Nx-1)*(Ny-1)*0.5)])
    uEnd = uk
    return uEnd,np.array(uloc)


x,y = np.mgrid[dx:Lx:dx, dy:Ly:dy]

#---------- Plotting Source and Coeff. function -----------------
# fvec = createF(x,y,sourceF)
# kvec = createF(x,y,coeffK)
#
#
# size = (Ny-1, Nx-1)
# fvec_reshaped = np.reshape(fvec, size)
# kvec_reshaped = np.reshape(kvec,size)
# plt.figure(1)
# plt.subplot(2,1,1)
# plt.imshow(fvec_reshaped, origin="lower", extent=((x[0, 0], x[-1, -1], y[0, 0], y[-1, -1])))
# plt.colorbar(orientation='vertical')
# plt.subplot(2,1,2)
# plt.imshow(kvec_reshaped, origin="lower", extent=((x[0, 0], x[-1, -1], y[0, 0], y[-1, -1])))
# plt.colorbar(orientation='vertical')
# plt.show()
#---------------------------------------------------

u_tilde = la.spsolve(create2DLFVM(x,y,coeffK),createF(x,y,sourceF))
u_tilde_reshaped = np.reshape(u_tilde,(Ny-1,Nx-1))

#--------------------- Plotting the steady solution  -------------------------
# plt.imshow(u_tilde_reshaped, origin="lower", extent=((x[0, 0], x[-1, -1], y[0, 0], y[-1, -1])))
# plt.colorbar(orientation='horizontal')
# plt.show()
#-------------------------------------------------------

ic = np.zeros((Nx-1)*(Ny-1))

t_start = time.time()
epsilon = 1.126771315288 * 10 ** -14
ufe, uloc_array = solveFE(ic,0,100,160000,x,y)
print("commencing backward euler forms")
for i in range(1,11):
    ube, uloc_array = solveBE(ic, 0, 100, 100*i,x,y)
    epsilon_ube = np.linalg.norm(ube - u_tilde)/(math.sqrt(len(ic) - 1))
    print("backward eurler using ", 1000*i, "timesteps")
    print(epsilon_ube)

t_end = time.time()
#ufe_reshaped = np.reshape(ufe, (Ny-1,Nx-1))

# plt.imshow(ufe_reshaped, origin="lower", extent=((x[0, 0], x[-1, -1], y[0, 0], y[-1, -1])))
# plt.colorbar(orientation='horizontal')
# plt.show()

#epsilon = np.linalg.norm(ufe - u_tilde)/(math.sqrt(len(ic) - 1))

print("solved in ", "{:.2f}".format(t_end - t_start), " s")
#print("epsilon = ", str(epsilon))