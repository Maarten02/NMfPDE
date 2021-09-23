import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la

LeftX = -1.5
RightX = 1.5
LeftY = -1.5
RightY = 1.5

Nx = 4
Ny = 4
dx = (RightX - LeftX) / Nx
dy = (RightY - LeftY) / Ny

D = sp.diags()

