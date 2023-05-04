import torch
import numpy as np
import time
from utils import *


def initial_condition(a, w, x_grid):

    return a * np.exp(-x_grid ** 2 / 2 / w / w)

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

        # g_e = torch.nn.Softplus()
        g_e = torch.nn.Sigmoid()

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


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


date = '01_24_2023_12_49'
bglasdi_results = np.load('//p/gpfs1/cbonnevi/BurgersEqn1D_2/results/bglasdi_' + date + '.npy', allow_pickle = True).item()

autoencoder_param = bglasdi_results['autoencoder_param']
t_grid = bglasdi_results['t_grid']
x_grid = bglasdi_results['x_grid']
time_dim, space_dim = t_grid.shape[0], x_grid.shape[0]

n_hidden = len(autoencoder_param.keys()) // 4 - 1
hidden_units = [autoencoder_param['fc' + str(i + 1) + '_e.weight'].shape[0] for i in range(n_hidden)]
n_z = autoencoder_param['fc' + str(n_hidden + 1) + '_e.weight'].shape[0]

autoencoder = Autoencoder(space_dim, hidden_units, n_z)
autoencoder.load_state_dict(autoencoder_param)
autoencoder = autoencoder.to(device)

a_min, a_max = 0.7, 0.9
w_min, w_max = 0.9, 1.1

Dt = bglasdi_results['Dt']
Dx = bglasdi_results['Dx']

sindy_coef = bglasdi_results['sindy_coef']
gp_dictionnary = bglasdi_results['gp_dictionnary']

maxk = 10
convergence_threshold = 1e-8

n_time_test = 55
burn = 5
time_fom = np.zeros(n_time_test)
time_rom = np.zeros(n_time_test)

param = np.random.rand(n_time_test, 2)
param[:, 0] = a_min + param[:, 0] * (a_max - a_min)
param[:, 1] = w_min + param[:, 1] * (w_max - w_min)

for i in range(n_time_test):

    a, w = param[i, 0], param[i, 1]
    u0 = initial_condition(a, w, x_grid)

    tic = time.time()
    _ = solver(u0, maxk, convergence_threshold, time_dim - 1, space_dim, Dt, Dx)
    toc = time.time() - tic

    time_fom[i] = toc

    tic = time.time()
    _ = direct_mean_prediction(a, w, u0, autoencoder, t_grid, gp_dictionnary, sindy_coef, device)
    toc = time.time() - tic

    time_rom[i] = toc

    print(np.round(time_fom[i], 5))
    print(np.round(time_rom[i], 5))

time_rom = time_rom[burn:]
time_fom = time_fom[burn:]

average_time_fom = time_fom.mean()
average_time_rom = time_rom.mean()

for i in range(n_time_test - burn):
    speed_up = time_fom[i] / time_rom[i]
    print('Test #%02d, FOM: %2.5f, ROM: %2.5f, SpeedUp: %4.2fx' % (i + 1, time_fom[i], time_rom[i], speed_up))

print('\nAverage:  FOM: %2.5f, ROM: %2.5f, SpeedUp: %4.2fx' % (average_time_fom, average_time_rom, average_time_fom / average_time_rom))
