# Solution of 1D Poisson's equation with FDM
# Maarten Krikke (c) 2021
import numpy as np
import matplotlib.pyplot as plt

#---------- Construct grid ---------
n = 5
xgrid = np.linspace(-1,1,n+1)

#---------- Source functions -------
def func1(x):
    return -6*x+2

def func2(x):
    return -12*x**2-6*x+2

f1 = func1(xgrid)
f2 = func2(xgrid)
#--------------- Exact solutions -------
def u1ex(x):
    return x**3-x**2+x-3

def u2ex(x):
    return x**4+x**3-x**2+x-4

u1 = u1ex(xgrid)
u2 = u2ex(xgrid)

#------- Plot the source functions ------
plt.figure()
plt.plot(xgrid, f1, 'b', label="f1")
plt.plot(xgrid, f2, 'r', label="f2")
plt.legend()
plt.title(r"$f_1$ and $f_2$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.show()

#----------- Plot the exact solutions ----
plt.figure()
plt.plot(xgrid, u1, 'b', label="u1")
plt.plot(xgrid, u2, 'r', label="u2")
plt.legend()
plt.title(r"$u_1^{ex}$ and $u_2^{ex}$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.show()