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

# def createGrid(Lx, Ly, Nx, Ny):
#     dx = Lx / Nx  # grid step in x-direction
#     dy = Ly / Ny  # grid step in y-direction
#

def sourceF(x,y,alpha=40):
    print("x,y ", x, y)
    f = 0
    for j in range(1,5):
        for i in range(1,10):

            f += np.exp(-alpha*(x-i)**2-alpha*(y-j)**2)
    return f

def coeffK1(x,y):
    return 1.0

def coeffK(x,y):
    return 1+0.1*(x+y+x*y)

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

# create grid, source fcn values
x,y = np.mgrid[dx:Lx:dx, dy:Ly:dy]
fvec = createF(x,y,sourceF)
print(fvec)
#--------------------- Plot source fcn and coefficient fcn ------------------------------
kvec = createF(x,y,coeffK)
print(kvec)
#reshape fvec and kvec for plotting
size = (Ny-1, Nx-1)
fvec_reshaped = np.reshape(fvec, size)
kvec_reshaped = np.reshape(kvec,size)
plt.figure(1)
plt.subplot(2,1,1)
plt.imshow(fvec_reshaped, origin="lower", extent=((x[0, 0], x[-1, -1], y[0, 0], y[-1, -1])))
plt.colorbar(orientation='vertical')
plt.subplot(2,1,2)
plt.imshow(kvec_reshaped, origin="lower", extent=((x[0, 0], x[-1, -1], y[0, 0], y[-1, -1])))
plt.colorbar(orientation='vertical')
plt.show()

#---------------------------------------------------------------------------------------

A = create2DLFVM(x, y, coeffK)
print(A.toarray())
print()
print(A.sum())
#----------- Verifying system matrix --------------------------
# print(A.toarray())
# plt.spy(A, marker='o', markersize=10, color='g')
# plt.show()
#--------------------------------------------------------------

uvec = la.spsolve(A, fvec)
u = np.reshape(uvec,(Ny-1,-1))
f = np.reshape(fvec, (Ny-1, Nx-1))
print(u.shape)
print(u)
plt.figure(2)
plt.imshow(u, origin="lower", extent=((x[0, 0], x[-1, -1], y[0, 0], y[-1, -1])))
#plt.imshow(f, origin="lower", extent=((x[0, 0], x[-1, -1], y[0, 0], y[-1, -1])))
plt.colorbar(orientation='horizontal')

plt.show()