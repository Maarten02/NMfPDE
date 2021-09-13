# Solution of 1D Poisson's equation with FDM
# Maarten Krikke (c) 2021
import numpy as np
import matplotlib.pyplot as plt

#---------- Construct grid ---------
n = 5
xgrid = np.linspace(-1,1,n+1)

#---------- Source functions -------
def func1(x):
    return x**3-x**2+x-3

def func2(x):
    return x**4+x**3-x**2-5

f1 = func1(xgrid)
f2 = func2(xgrid)

#------- Plot the source functions ------
plt.figure()
plt.plot(xgrid, f1, 'b')
plt.plot(xgrid, f2, 'r')
plt.legend()
plt.title(r"$f_1$ and $f_2$")
plt.xlabel(r"$y$")
plt.ylabel(r"$x$")