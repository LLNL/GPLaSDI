import numpy as np
from utils import *
import matplotlib.pyplot as plt

def initial_condition(a, w, x_grid):

    return a * np.exp(-x_grid ** 2 / 2 / w / w)


time_dim = 1001
space_dim = 1001

x_min, x_max = -3, 3
t_max = 1
Dx = (x_max - x_min) / (space_dim - 1)
Dt = t_max / (time_dim - 1)
x_grid = np.linspace(x_min, x_max, space_dim)
t_grid = np.linspace(0, t_max, time_dim)
t_mesh, x_mesh = np.meshgrid(t_grid, x_grid)

a_min, a_max = 0.7, 0.9
w_min, w_max = 0.9, 1.1

a_train = np.array([a_min, a_max])
w_train = np.array([w_min, w_max])

a_train, w_train = np.meshgrid(a_train, w_train)
param_train = np.hstack((a_train.reshape(-1, 1), w_train.reshape(-1, 1)))
n_train = param_train.shape[0]

n_a_grid = 21
n_w_grid = 21
a_grid = np.linspace(a_min, a_max, n_a_grid)
w_grid = np.linspace(w_min, w_max, n_w_grid)
a_grid, w_grid = np.meshgrid(a_grid, w_grid)
param_grid = np.hstack((a_grid.reshape(-1, 1), w_grid.reshape(-1, 1)))
n_test = param_grid.shape[0]


U0 = [initial_condition(param_train[i, 0], param_train[i, 1], x_grid) for i in range(n_train)]
X_train = generate_initial_data(U0, time_dim, space_dim, Dt, Dx)

data_train = {'param_train' : param_train, 'X_train' : X_train, 'n_train' : n_train, 'U0' : U0}
np.save('data/data_train.npy', data_train)

U0 = [initial_condition(param_grid[i, 0], param_grid[i, 1], x_grid) for i in range(n_test)]
X_test = generate_initial_data(U0, time_dim, space_dim, Dt, Dx)

data_test = {'param_grid' : param_grid, 'X_test' : X_test, 'n_test' : n_test, 'U0' : U0}
np.save('data/data_test.npy', data_test)
