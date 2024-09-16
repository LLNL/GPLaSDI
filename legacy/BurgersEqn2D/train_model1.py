import numpy as np
import torch
import os
from train_framework import BayesianGLaSDI


class Autoencoder(torch.nn.Module):
    def __init__(self, space_dim, hidden_units, n_z):
        super(Autoencoder, self).__init__()

        n_layers = len(hidden_units)
        self.n_layers = n_layers

        fc1_e = torch.nn.Linear(space_dim, hidden_units[0])
        self.fc1_e = fc1_e

        if n_layers > 1:
            for i in range(n_layers - 1):
                fc_e = torch.nn.Linear(hidden_units[i], hidden_units[i + 1])
                setattr(self, 'fc' + str(i + 2) + '_e', fc_e)

        fc_e = torch.nn.Linear(hidden_units[-1], n_z)
        setattr(self, 'fc' + str(n_layers + 1) + '_e', fc_e)

        g_e = torch.nn.Softplus()
        # g_e = torch.nn.Sigmoid()
        self.g_e = g_e

        fc1_d = torch.nn.Linear(n_z, hidden_units[-1])
        self.fc1_d = fc1_d

        if n_layers > 1:
            for i in range(n_layers - 1, 0, -1):
                fc_d = torch.nn.Linear(hidden_units[i], hidden_units[i - 1])
                setattr(self, 'fc' + str(n_layers - i + 1) + '_d', fc_d)

        fc_d = torch.nn.Linear(hidden_units[0], space_dim)
        setattr(self, 'fc' + str(n_layers + 1) + '_d', fc_d)



    def encoder(self, x):

        for i in range(1, self.n_layers + 1):
            fc = getattr(self, 'fc' + str(i) + '_e')
            x = self.g_e(fc(x))

        fc = getattr(self, 'fc' + str(self.n_layers + 1) + '_e')
        x = fc(x)

        return x


    def decoder(self, x):

        for i in range(1, self.n_layers + 1):
            fc = getattr(self, 'fc' + str(i) + '_d')
            x = self.g_e(fc(x))

        fc = getattr(self, 'fc' + str(self.n_layers + 1) + '_d')
        x = fc(x)

        return x


    def forward(self, x):

        x = Autoencoder.encoder(self, x)
        x = Autoencoder.decoder(self, x)

        return x



def initial_condition_func(ic, a, w, nx, ny):

    if ic == 1:
        xmin = 0
        xmax = 1
        ymin = 0
        ymax = 1
    elif ic == 2:
        xmin = -3
        xmax = 3
        ymin = -3
        ymax = 3
        x0 = 0
        y0 = 0

    [xv, yv] = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny), indexing = 'xy')

    zv = a * np.exp(-((xv - x0) ** 2 + (yv - y0) ** 2) / w)
    z = zv.flatten()

    u0 = np.vstack((z.reshape(1, -1), z.reshape(1, -1)))
    
    return u0 

# Space Time Definition
time_dim = 201
space_dim = 7200
t_max = 1

# Parameter Space Definition
a_min, a_max = 0.7, 0.9
w_min, w_max = 0.9, 1.1

n_a_grid = 21
n_w_grid = 21

# Autoencoder Definition
hidden_units = [100, 20, 20, 20]
n_z = 5
autoencoder = Autoencoder(space_dim, hidden_units, n_z)

# Training Parameters
n_samples = 20
lr = 0.001
n_iter = 500000
n_greedy = 50000
sindy_weight = 0.1
coef_weight = 1e-6

# FOM Parameters
ic = 2
Re = 10000
nx = 60
ny = 60
nt = 200
dt = 1 / nt
nxy = (nx - 2) * (ny - 2)
dx = 1 / (nx - 1)
dy = 1 / (ny - 1)
maxitr = 10
tol = 1e-8

# Path
path_data = '//p/gpfs1/cbonnevi/BurgersEqn2D/data/'
path_checkpoint = 'checkpoint/'
path_results = '//p/gpfs1/cbonnevi/BurgersEqn2D/results/'

# CUDA
cuda = torch.cuda.is_available()

# Best Loss
keep_best_loss = True

# --------------- Loading Parameters --------------- #

t_grid = np.linspace(0, t_max, time_dim)

data_train = np.load(path_data + 'data_train.npy', allow_pickle = True).item()

model_parameters = {}

model_parameters['time_dim'] = time_dim
model_parameters['space_dim'] = space_dim
model_parameters['n_z'] = n_z
model_parameters['t_grid'] = t_grid

model_parameters['Dt'] = dt
model_parameters['Dx'] = dx
model_parameters['Dy'] = dy
model_parameters['ic'] = ic
model_parameters['Re'] = Re
model_parameters['nx'] = nx
model_parameters['ny'] = ny
model_parameters['nxy'] = nxy
model_parameters['nt'] = nt
model_parameters['maxitr'] = maxitr
model_parameters['tol'] = tol

model_parameters['initial_condition'] = initial_condition_func

model_parameters['a_min'] = a_min
model_parameters['a_max'] = a_max
model_parameters['w_min'] = w_min
model_parameters['w_max'] = w_max
model_parameters['n_a_grid'] = n_a_grid
model_parameters['n_w_grid'] = n_w_grid

model_parameters['data_train'] = data_train

a_grid = np.linspace(a_min, a_max, n_a_grid)
w_grid = np.linspace(w_min, w_max, n_w_grid)
a_grid, w_grid = np.meshgrid(a_grid, w_grid)
param_grid = np.hstack((a_grid.reshape(-1, 1), w_grid.reshape(-1, 1)))
n_test = param_grid.shape[0]

model_parameters['param_grid'] = param_grid
model_parameters['n_test'] = n_test

model_parameters['n_samples'] = n_samples
model_parameters['lr'] = lr
model_parameters['n_iter'] = n_iter
model_parameters['n_greedy'] = n_greedy
model_parameters['sindy_weight'] = sindy_weight
model_parameters['coef_weight'] = coef_weight

model_parameters['path_checkpoint'] = path_checkpoint
model_parameters['path_results'] = path_results
model_parameters['keep_best_loss'] = keep_best_loss

model_parameters['cuda'] = cuda

bglasdi = BayesianGLaSDI(autoencoder, model_parameters)
bglasdi.train()
