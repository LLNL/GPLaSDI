import torch
import numpy as np
import os
from utils import *
from solver_burgers import *
import time


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

    if ic == 1:  # sine
        xmin = 0
        xmax = 1
        ymin = 0
        ymax = 1
    elif ic == 2:  # Gaussian
        xmin = -3
        xmax = 3
        ymin = -3
        ymax = 3
        x0 = 0  # Gaussian center
        y0 = 0  # Gaussian center

    [xv, yv] = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny), indexing = 'xy')

    if ic == 1:  # IC: sine
        zv = a * np.sin(2 * np.pi * xv) * np.sin(2 * np.pi * yv)
        zv[np.nonzero(xv > 0.5)] = 0.0
        zv[np.nonzero(yv > 0.5)] = 0.0
    elif ic == 2:  # IC: Gaussian
        zv = a * np.exp(-((xv - x0) ** 2 + (yv - y0) ** 2) / w)
        z = zv.flatten()

    u0 = np.hstack((z.reshape(1, -1), z.reshape(1, -1)))

    return u0


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


date = '01_29_2023_23_40'
bglasdi_results = np.load('//p/gpfs1/cbonnevi/BurgersEqn2D/results/bglasdi_' + date + '.npy', allow_pickle = True).item()

autoencoder_param = bglasdi_results['autoencoder_param']
t_grid = bglasdi_results['t_grid']
time_dim = 201
space_dim = 7200

n_hidden = len(autoencoder_param.keys()) // 4 - 1
hidden_units = [autoencoder_param['fc' + str(i + 1) + '_e.weight'].shape[0] for i in range(n_hidden)]
n_z = autoencoder_param['fc' + str(n_hidden + 1) + '_e.weight'].shape[0]

autoencoder = Autoencoder(space_dim, hidden_units, n_z)
autoencoder.load_state_dict(autoencoder_param)
autoencoder = autoencoder.to(device)

a_min, a_max = 0.7, 0.9
w_min, w_max = 0.9, 1.1

maxitr = 10
tol = 1e-8
ic = 2

Dt = bglasdi_results['Dt']
Dx = bglasdi_results['Dx']
Dy = bglasdi_results['Dy']
nt = t_grid.shape[0] - 1
nx, ny = 60, 60
nxy = (nx - 2) * (ny - 2)
Re = 10000

sindy_coef = bglasdi_results['sindy_coef']
gp_dictionnary = bglasdi_results['gp_dictionnary']

maxk = 10
convergence_threshold = 1e-8

n_time_test = 22
burn = 2
time_fom = np.zeros(n_time_test)
time_rom = np.zeros(n_time_test)

param = np.random.rand(n_time_test, 2)
param[:, 0] = a_min + param[:, 0] * (a_max - a_min)
param[:, 1] = w_min + param[:, 1] * (w_max - w_min)

for i in range(n_time_test):

    a, w = param[i, 0], param[i, 1]
    u0 = initial_condition_func(ic, a, w, nx, ny)

    tic = time.time()
    _ = solver(ic, a, w, Re, nx, ny, nt, Dt, nxy, Dx, Dy, maxitr, tol)
    toc = time.time() - tic

    time_fom[i] = toc

    tic = time.time()
    _ = direct_mean_prediction(a, w, u0, autoencoder, t_grid, gp_dictionnary, sindy_coef, device)
    toc = time.time() - tic

    time_rom[i] = toc

    print(np.round(time_fom, 5))
    print(np.round(time_rom, 5))

time_fom = time_fom[burn:]
time_rom = time_rom[burn:]

average_time_fom = time_fom.mean()
average_time_rom = time_rom.mean()

for i in range(n_time_test - burn):
    speed_up = time_fom[i] / time_rom[i]
    print('Test #%02d, FOM: %2.5f, ROM: %2.5f, SpeedUp: %4.2fx' % (i + 1, time_fom[i], time_rom[i], speed_up))

print('\nAverage:  FOM: %2.5f, ROM: %2.5f, SpeedUp: %4.2fx' % (average_time_fom, average_time_rom, average_time_fom / average_time_rom))
