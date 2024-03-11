import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags
from scipy.integrate import odeint
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
import torch
import matplotlib.pyplot as plt
from utils_solver import *


#These are all rewritten with 'sed' command
time_dim = 1001
space_dim = 1001
x_min, x_max = -3.0, 3.0
t_max = 1.0

Dx = (x_max - x_min) / (space_dim - 1)
Dt = t_max / (time_dim - 1)
x_grid = np.linspace(x_min, x_max, space_dim)
t_grid = np.linspace(0, t_max, time_dim)


u0 = initial_condition(0.8677419354838709, 1.0225806451612904, x_grid)
maxk = 10
convergence_threshold = 1e-8
nt = len(t_grid)
nx = len(x_grid)

u = solver(u0, maxk, convergence_threshold, nt - 1, nx, Dt, Dx)
u = u.reshape(1, nt, nx)

    
np.save('1DBurgers_a' + str(0.8677419354838709) + '_w' + str(1.0225806451612904) +'.npy',u)