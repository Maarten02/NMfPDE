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
    Dx = sp.diags([1,-1], offsets=[0,-1], shape=(Nx,Nx-1))/dx
    Dy = sp.diags([1,-1], offsets=[0,-1], shape=(Ny,Ny-1))/dy

    Lxx = Dx.transpose().dot(Dx)
    Lyy = Dy.transpose().dot(Dy)

    Ix = sp.eye(Nx-1)
    Iy = sp.eye(Ny-1)

    A = sp.kron(Iy,Lxx) + sp.kron(Lyy,Ix)

    return A

def sourcefunc(x,y):
    return 1 + x + y - x*y

def domainfunc(x,y):
    return (x**2 + y**2 - 1)**3 - x**2*y**3


y,x = np.mgrid[LeftY+stepY:RightY:stepY, LeftX+stepX:RightX:stepX]

f = sourcefunc(x,y)
#print(f)
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
# #plt.title('The source function on a rectangular domain with $N_x*N_y = 100*100$')
# plt.subplot(1,2,2)
# plt.imshow(ffill, origin="lower", extent=((x[0, 0], x[-1, -1], y[0, 0], y[-1, -1])))
# plt.colorbar(orientation='horizontal')
# #plt.title('The source function on $\Omega$,$N_x*N_y = 100*100$')
# #plt.show()
# # sm = FDLaplacian2D(200,200)
#
#
# #plt.show()
#-----------------------------------------------------------------------------
# lexicographic domain function
domainLX = np.reshape(domain,-1) # reshape 2D(x) domain array into 1D domainLX array
# find lexicographic indices of inner points
rowsLX, colsLX, valsLX = sp.find(domainLX < 0)
#print(colsLX)
#lexicographic source vector on rectangular domain
fLX = np.reshape(f, -1) # reshape 2D f array into 1D fLX array
# lexicographic source vector on deformed domain
fLXd = fLX[colsLX]
#print(fLXd)
# 2D FD Laplacian on rectangular domain
A = FDLaplacian2D(Nx,Ny)
# plt.spy(A, marker='o', markersize=10, color='g')
# plt.grid()
# plt.show()
#print(A.toarray())
# 2D FD Laplacian on deformed domain
Ad = A.tocsr()[colsLX,:].tocsc()[:,colsLX]
#tol = 1e-8
#print(np.all(np.abs(Ad-Ad.T) < tol))
#print(Ad.toarray())
u = la.spsolve(Ad, fLXd)
#print(u)
# preparing to display the solution
# minc = np.min(u) # background color
# ufill = minc*np.ones([Ny-1,Nx-1]) # emptyrectangular image
# ufill[rows,cols] = u # filling part of the imagewith the solution
#
# plt.figure(2)
# plt.imshow(ufill, origin="lower", extent=((x[0, 0], x[-1, -1], y[0, 0], y[-1, -1])))
# plt.colorbar(orientation='horizontal')
# plt.plot('The solution u on $\Omega$, $N_x*N_y=100*100')
# plt.show()

eigenvals, eigenvecs = sp.linalg.eigsh(Ad, k=20, which='SM')

for i in range(np.size(eigenvecs, axis=1)):
    x = eigenvecs[:,i]
    mag = x.dot(u)
    print('eigenvector', i+1, 'has dotproduct:', mag)
#print(len(eigenvals), eigenvals, eigenvecs)