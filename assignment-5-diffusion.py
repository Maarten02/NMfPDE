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
        return 0.2
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
    uloc = [0.0]
    dt = (tEnd - tStart) / Nt
    uk = uStart
    A = create2DLFVM(x,y,coeffK)
    f = createF(x,y, sourceF)
    Adt = sp.eye((Nx-1)*(Ny-1)) - A*dt
    fdt = f*dt
    for k in range(Nt):
        uk = Adt.dot(uk) + fdt
        uloc.append(uk[int((Nx-1)*(Ny-1)*0.5)])
    uEnd = uk
    return uEnd, np.array(uloc)

def solveBE(uStart,tStart,tEnd,Nt, x, y):
    uloc = [0.0]
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

def getBEsteps(ic, epsilon, left=38, right=42):

    size = (Nx-1)*(Ny-1)
    BEtime_steps = 0
    Running = True
    it = 1

    while Running:

        print("iteration ", it," started")
        print("left = ", left, "right = ", right)
        middle = int((left + right)*0.5)

        ubem, uloc_array3 = solveBE(ic, 0, 100, middle, x, y)
        epsilon_ubem = np.linalg.norm(ubem - u_tilde) / (math.sqrt(size - 1))

        if right - left == 1 or right == left:
            Running = False
            ubes, uloc_array3 = solveBE(ic, 0, 100, right, x, y)
            epsilon_ubes = np.linalg.norm(ubes - u_tilde) / (math.sqrt(size - 1))
            if epsilon_ubes < epsilon:
                BEtime_steps = right
            else:
                BEtime_steps = right + 1
        elif epsilon_ubem < epsilon:
            right = middle
        else:
            left = middle
        print("iteration", it, "completed")
        it += 1
    return BEtime_steps

x,y = np.mgrid[dx:Lx:dx, dy:Ly:dy]
u_tilde = la.spsolve(create2DLFVM(x,y,coeffK),createF(x,y,sourceF))
u_tilde_reshaped = np.reshape(u_tilde,(Ny-1,Nx-1))

# plt.imshow(u_tilde_reshaped, origin="lower", extent=((x[0, 0], x[-1, -1], y[0, 0], y[-1, -1])))
# plt.colorbar(orientation='horizontal')
# plt.show()

ic = np.zeros((Nx-1)*(Ny-1))
t1 = time.time()
ufe, uloc_array = solveFE(ic,0,100,160000,x,y)
t2 = time.time()
ube, uloc_array2 = solveBE(ic, 0, 100, 41, x, y)
t3 = time.time()

print("Forward Euler solved in ", "{:.2f}".format(t2 - t1), " s")
print("Backward Euler solved in ", "{:.2f}".format(t3 - t2), " s")
epsilonfe = np.linalg.norm(ufe - u_tilde)/(math.sqrt(len(ic)))
epsilonbe = np.linalg.norm(ube - u_tilde)/math.sqrt(len(ic))
print(" Forward euler epsilon: " ,epsilonfe," Backward euler epsilon: ",  epsilonbe)
# steps = getBEsteps(ic, epsilonfe)
# print("steps required for BE: ", steps)
print(uloc_array2)
plt.semilogx(uloc_array, label='Forward Euler')
plt.semilogx(uloc_array2, label='Backward Euler')
plt.legend()
plt.xlabel("Time iteration steps")
plt.ylabel("Solution value")
plt.show()

# Currently not used
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
#--------------------- Plotting the steady solution  -------------------------
# plt.imshow(u_tilde_reshaped, origin="lower", extent=((x[0, 0], x[-1, -1], y[0, 0], y[-1, -1])))
# plt.colorbar(orientation='horizontal')
# plt.show()
#-------------------------------------------------------
#ufe_reshaped = np.reshape(ufe, (Ny-1,Nx-1))

# plt.imshow(ufe_reshaped, origin="lower", extent=((x[0, 0], x[-1, -1], y[0, 0], y[-1, -1])))
# plt.colorbar(orientation='horizontal')
# # plt.show()
# epsilon = np.linalg.norm(ufe - u_tilde)/(math.sqrt(len(ic) - 1))
# print("epsilon = ", str(epsilon))

# BEtime_steps = getBEsteps(ic)
# print("At least ", BEtime_steps, " timesteps for backward Euler")
#
# print("epsilon = ", str(epsilon))
# # print("Forward Euler solved in  ", "{:.2f}".format(t_end1 - t_start), " s")
# # t_start2 = time.time()
# # ube, uloc_array2 = solveBE(ic, 0, 100, 46, x, y)
# # t_end2 = time.time()
# # epsilon = np.linalg.norm(ube - u_tilde)/(math.sqrt(len(ic)))
# # print("epsilon = ", str(epsilon))
# # print("Backward Euler solved in ", "{:.2f}".format(t_end2 - t_start2), " s")
#
# # ufe_reshaped = np.reshape(ufe, (Ny-1, Nx-1))
# # plt.imshow(ufe_reshaped, origin="lower", extent=((x[0, 0], x[-1, -1], y[0, 0], y[-1, -1])))
# # plt.colorbar(orientation='horizontal')
# # plt.show()