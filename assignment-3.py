import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la

LeftX = -1.5
RightX = 1.5
LeftY = -1.5
RightY = 1.5

Nx = 100
Ny = 100
stepX = (RightX - LeftX) / Nx
stepY = (RightY - LeftY) / Ny
def FDLaplacian2D(Nx,Ny):

    dx = (RightX - LeftX) / Nx
    dy = (RightY - LeftY) / Ny

    diagonal = np.ones((Nx+1))
    Dx = sp.diags([1,-1], offsets=[0,-1], shape=(Nx+2,Nx+1))
    Dy = sp.diags([1,-1], offsets=[0,-1], shape=(Ny+2,Ny+1))

    Lxx = Dx.transpose().dot(Dx)
    Lyy = Dy.transpose().dot(Dy)

    Ix = sp.eye(Nx+1)
    Iy = sp.eye(Ny+1)

    A = sp.kron(Iy,Lxx) + sp.kron(Lyy,Ix)
    return A

def sourcefunc(x,y):
    return 1 + x + y - x*y

def domainfunc(x,y):
    return (x**2 + y**2 - 1)**3 - x**2*y**3


y,x = np.mgrid[LeftY+stepY:RightY:stepY, LeftX+stepX:RightX:stepX]
f = sourcefunc(x,y)
domain = domainfunc(x,y)
rows, cols, vals = sp.find(domain < 0)

minc = np.min(f) # background color
ffill = minc*np.ones([Ny-1,Nx-1]) # empty rectangular image
ffill[rows,cols] = f[rows,cols] # filling part of the image ffill with the values of f over deformed domain

#---------------------- Plotting the source functions ----------------------
# #plt.ion()
# plt.figure(1)
# #plt.clf()
# plt.subplot(1,2,1)
# plt.imshow(f, origin="lower", extent=((x[0, 0], x[-1, -1], y[0, 0], y[-1, -1])))
# plt.colorbar(orientation='horizontal')
#
# plt.subplot(1,2,2)
# plt.imshow(ffill, origin="lower", extent=((x[0, 0], x[-1, -1], y[0, 0], y[-1, -1])))
# plt.colorbar(orientation='horizontal')
# plt.show()
# # sm = FDLaplacian2D(200,200)
#
# # plt.spy(A)
# plt.show()
#-----------------------------------------------------------------------------
# lexicographic domain function
domainLX = np.reshape(x,-1) # reshape 2D domain array into 1D domainLX array
domainLY = np.reshape(y,-1) # reshape 2D domain array into 1D domainLX array
# find lexicographic indices of inner points
rowsLX, colsLX, valsLX = sp.find(domainfunc(domainLX, domainLY) < 0)
#lexicographic source vector on rectangular domain
fLX = np.reshape(f, -1) # reshape 2D f array into 1D fLX array
# lexicographic source vector on deformed domain
fLXd = fLX[colsLX]

