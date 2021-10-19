import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.sparse as sp
import scipy.sparse.linalg as la
import math
import time
Lx = 10  # length in x direction
Ly = 10  # length in y direction
Nx = 200  # number of intervals in x-direction
Ny = 200  # number of intervals in y-direction
dx = Lx/Nx  # grid step in x-direction
dy = Ly/Ny  # grid step in y-direction
Nt = 425
def coeffK(x,y, t=None):
    if (x - 7)**2 + (y-5)**2 <= 1.25**2:
        return 0.5
    else:
        return 1.0

def sourceF(x,y,t,alpha=40, omega=4*math.pi):
    if t >= 0 and t <= 0.5:
        f = math.sin(omega * t)*math.exp(-alpha*(x-3)**2-alpha*(y-5)**2)
    else:
        f = 0
    return f

def createF(x, y, t, funcName):
    Fvec = np.empty(((Ny-1)*(Nx-1)))
    for j in range(Ny-1):
        for i in range(Nx-1):
            Fvec[j*(Nx-1)+i] = funcName(x[i,j], y[i,j], t)
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


def stepWave(ukm1, uk, tStart, dt, sysm):
    uEnd = 2*uk - ukm1 + dt**2*(-sysm.dot(uk)+createF(x,y,tStart-dt,sourceF))
    return uEnd

def simulateWave(N=425, T=15):
    dt = T/N
    ic = np.zeros((Nx-1)*(Ny-1))
    u1 = dt**2*0.5*(createF(x,y,0,sourceF))
    allframes = np.empty((N,(Nx-1)*(Ny-1)))
    allframes[0] = ic
    allframes[1] = u1
    sysm = create2DLFVM(x, y, coeffK)
    for i in range(2,N-1):
        #print(i)
        uk = allframes[i-1]
        ukm1 = allframes[i-2]
        allframes[i] = stepWave(ukm1, uk, i*dt, dt, sysm)
    return allframes

x,y = np.mgrid[dx:Lx:dx, dy:Ly:dy]

simulate = False
if simulate:
    print('check')
    np.save('frames',simulateWave())
    simulate = False

allframes = np.load('frames.npy')
print(allframes.shape)
cmin = allframes.min()
cmax = allframes.max()
print(cmin, cmax)
def animate(frame):
    dt = 15/425
    t = frame*dt
    img.set_array(np.reshape(allframes[frame], [Ny-1, Nx-1]))
    tlt.set_text(r"$u(x,y,t), t=$ " + str(np.round(t+2*dt,3)))
    return img

t=0
fig = plt.figure()
init = np.reshape(allframes[0], (Ny-1, Nx-1))
img = plt.imshow(init, extent=[dx/2, Lx-dx/2, Ly-dy/2, dy/2], interpolation='none')
plt.gca().invert_yaxis()
plt.colorbar(img, orientation='horizontal')
tlt = plt.title(r"$u(x,y,t), t=$ " + str(np.round(t,3)))
img.set_clim(cmin, cmax)
anim = mpl.animation.FuncAnimation(fig, animate, Nt-1, interval=50, repeat=False)
#anim.save('assignment5.gif', writer=mpl.animation.PillowWriter(fps=30))
plt.show()
# Currently not used
#---------- Plotting Source and Coeff. function -----------------
# fvec = createF(x, y, 0.125, sourceF)
# kvec = createF(x, y, 0.125, coeffK)
#
#
# size = (Ny-1, Nx-1)
# fvec_reshaped = np.reshape(fvec, size)
# kvec_reshaped = np.reshape(kvec,size)
# plt.figure(1)
# plt.subplot(1,2,1)
# plt.imshow(fvec_reshaped, origin="lower", extent=((x[0, 0], x[-1, -1], y[0, 0], y[-1, -1])))
# plt.colorbar(orientation='horizontal')
# plt.subplot(1,2,2)
# plt.imshow(kvec_reshaped, origin="lower", extent=((x[0, 0], x[-1, -1], y[0, 0], y[-1, -1])))
# plt.colorbar(orientation='horizontal')
# plt.show()
#---------------------------------------------------