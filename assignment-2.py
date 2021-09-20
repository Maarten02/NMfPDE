# Solution of 1D Poisson's equation with FDM
# Maarten Krikke (c) 2021
import numpy as np
import matplotlib.pyplot as plt
import math

def func1(x):
    return -6*x+2

def func2(x):
    return -12*x**2-6*x+2

def u1ex(x):
    return x**3-x**2+x-3

def u2ex(x):
    return x**4+x**3-x**2+x-4

def get_system_matrix(n,h):
    off_diag = np.ones((n-2))*-1
    diag = np.ones((n-1))*2
    A_1 = np.diag(off_diag, k=-1)
    A_2 = np.diag(diag)
    A_3 = np.diag(off_diag, k=1)
    return (A_1 + A_2 + A_3)/h**2

def rmse(u_num, u_ex):
    n = len(u_ex)
    end = n-1
    return np.sqrt(np.sum(np.square((u_num - u_ex[1:end])))/(end-1))


def solution_rmse(n):
    h=2/n
    xgrid = np.linspace(-1,1,n+1)

    f1 = func1(xgrid)
    f2 = func2(xgrid)

    u1ex_xg = u1ex(xgrid)
    u2ex_xg = u2ex(xgrid)

    A = get_system_matrix(n,h)

    ylb = -6
    yrb = -2

    f1rhs = f1[1:n]
    f2rhs = f2[1:n]
    f1rhs[0] += ylb/h**2
    f1rhs[n-2] += yrb/h**2
    f2rhs[0]  += ylb/h**2
    f2rhs[n-2] += yrb/h**2

    u1_inner = np.linalg.solve(A, f1rhs)
    u2_inner = np.linalg.solve(A, f2rhs)
    u1 = np.concatenate([[ylb], u1_inner , [yrb]])
    u2 = np.concatenate([[ylb], u2_inner , [yrb]])
    # print(rmse(u1_inner, u1ex_xg), rmse(u2_inner, u2ex_xg))
    error1 = rmse(u1_inner, u1ex_xg)
    error2 = rmse(u2_inner, u2ex_xg)
    # print(error1, error2)
    print(n)
    return error1,error2


n_range = np.arange(3,1501)

s1_rmse = []
s2_rmse = []
for n in n_range:
    rmse1, rmse2 = solution_rmse(n)
    s1_rmse.append(rmse1)
    s2_rmse.append(rmse2)

s1_rmse = np.array(s1_rmse)
s2_rmse = np.array(s2_rmse)

plt.figure()
plt.loglog(n_range, s1_rmse, 'b', label="$RMSE u_1$")
plt.loglog(n_range, s2_rmse, 'r', label="$RMSE u_2$")
plt.legend()
plt.show()
