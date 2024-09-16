import numpy as np
import torch
import os
from train_framework import BayesianGLaSDI

import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(1)

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

        # x = torch.tanh(x)
        # x = torch.softmax(x)
 
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


def initial_condition_func(T, k):

    nx = 128
    nv = 128

    x_min = 0
    x_max = 6.2340979219672459E+00

    v_min = -7
    v_max = 6.8906250000000000E+00

    x_grid = np.linspace(x_min, x_max, nx)
    v_grid = np.linspace(v_min, v_max, nv)
    x_grid, v_grid = np.meshgrid(x_grid, v_grid)
    x_grid = x_grid.reshape(-1, 1)
    v_grid = v_grid.reshape(-1, 1)

    u0 = 8 / np.sqrt(2 * np.pi * T) * (1 + 0.1 * np.cos(k * x_grid)) * (np.exp(- (v_grid - 2) ** 2 / 2 / T) + np.exp(- (v_grid + 2) ** 2 / 2 / T))
    u0 = u0.T

    return u0


# Space Time Definition
time_dim = 251
space_dim = 16384
# space_dim = 40000
t_max = 5

Dt = 0.02
# Dt = 0.001
# Parameter Space Definition
T_min, T_max = 0.9, 1.1
k_min, k_max = 1.0, 1.2

n_T_grid = 21
n_k_grid = 21

# Autoencoder Definition
# hidden_units = [1000, 500, 100]
hidden_units = [1000, 200, 50, 50, 50] 
n_z = 5
autoencoder = Autoencoder(space_dim, hidden_units, n_z)

# Training Parameters
n_samples = 20
lr = 0.00001
n_iter = 300000
n_greedy = 50000
sindy_weight = 0.1
coef_weight = 1e-5

# sine_term = True
sine_term = False
poly_order = 1
n_proc = 4

# Path
path_data = '//p/gpfs1/cbonnevi/Vlasov1D1V/data/'
path_checkpoint = 'checkpoint/'
path_results = '//p/gpfs1/cbonnevi/Vlasov1D1V/results/'

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

model_parameters['Dt'] = Dt

model_parameters['initial_condition'] = initial_condition_func

model_parameters['a_min'] = T_min
model_parameters['a_max'] = T_max
model_parameters['w_min'] = k_min
model_parameters['w_max'] = k_max
model_parameters['n_a_grid'] = n_T_grid
model_parameters['n_w_grid'] = n_k_grid

model_parameters['poly_order'] = poly_order
model_parameters['sine_term'] = sine_term

model_parameters['n_proc'] = n_proc

model_parameters['data_train'] = data_train

T_grid = np.linspace(T_min, T_max, n_T_grid)
k_grid = np.linspace(k_min, k_max, n_k_grid)
T_grid, k_grid = np.meshgrid(T_grid, k_grid)
param_grid = np.hstack((T_grid.reshape(-1, 1), k_grid.reshape(-1, 1)))
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
