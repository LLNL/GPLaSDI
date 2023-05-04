import numpy as np
from solver_burgers import solver

Re = 10000
nx = 60
ny = nx
ic = 2  # initial condition, 1: Sine, 2: Gaussian
nt = 200
tstop = 1.0
dt = tstop / nt
t = np.linspace(0, tstop, nt + 1)
nxy = (nx - 2) * (ny - 2)
dx = 1 / (nx - 1)
dy = 1 / (ny - 1)
maxitr = 10
tol = 1e-8

time_dim = nt + 1
space_dim = nx * ny * 2

a_min, a_max = 0.7, 0.9
w_min, w_max = 0.9, 1.1

a_train = np.array([a_min, a_max])
w_train = np.array([w_min, w_max])

a_train, w_train = np.meshgrid(a_train, w_train)
param_train = np.hstack((a_train.reshape(-1, 1), w_train.reshape(-1, 1)))
n_train = param_train.shape[0]

'''
for i in range(n_train):
    print(i)
    a, w = param_train[i, 0], param_train[i, 1]
    u, v, _ = solver(ic, a, w, Re, nx, ny, nt, dt, nxy, dx, dy, maxitr, tol)
    X_i = np.hstack((u, v))
    X_i = X_i.reshape(1, time_dim, space_dim)

    if i == 0:
        X_train = X_i
    else:
        X_train = np.concatenate((X_train, X_i), axis = 0)


data_train = {'param_train' : param_train, 'X_train' : X_train, 'n_train' : n_train}
np.save('data/data_train.npy', data_train)
'''

n_a_grid = 21
n_w_grid = 21
a_grid = np.linspace(a_min, a_max, n_a_grid)
w_grid = np.linspace(w_min, w_max, n_w_grid)
a_grid, w_grid = np.meshgrid(a_grid, w_grid)
param_grid = np.hstack((a_grid.reshape(-1, 1), w_grid.reshape(-1, 1)))
n_test = param_grid.shape[0]
batch_size = 2
batch_index = list(np.arange(0, n_test, batch_size))
if batch_index[-1] != n_test:
    batch_index.append(n_test)

n_batch = len(batch_index) - 1

for b in range(200, n_batch):
    for i in range(batch_index[b], batch_index[b + 1]):
        print(i)

        a, w = param_grid[i, 0], param_grid[i, 1]
        u, v, _ = solver(ic, a, w, Re, nx, ny, nt, dt, nxy, dx, dy, maxitr, tol)
        X_i = np.hstack((u, v))
        X_i = X_i.reshape(1, time_dim, space_dim)

        if i == batch_index[b]:
            X_batch = X_i
        else:
            X_batch = np.concatenate((X_batch, X_i), axis = 0)

    param_batch = param_grid[batch_index[b]:batch_index[b + 1], :]
    data_batch = {'param_batch' : param_batch, 'X_batch' : X_batch}
    np.save('data/data_test_batch_' + str(b + 1) + '.npy', data_batch)
