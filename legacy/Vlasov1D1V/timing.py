import numpy as np
import torch
from train_framework import *

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


def direct_mean_prediction(a, b, u0, autoencoder, t_grid, gp_dictionnary, sindy_coef, device):

    coef_matrix = np.zeros_like(sindy_coef[0])
    coef_y, coef_x = coef_matrix.shape

    param = np.array([[a, b]])

    c = 1
    for i in range(coef_y):
        for j in range(coef_x):
            coef_matrix[i, j] = gp_dictionnary['coef_' + str(c)].predict(param).item()
            c += 1

    coef_matrix = coef_matrix.T
    dzdt = lambda z, t: coef_matrix[:, 1:] @ z + coef_matrix[:, 0]

    u0 = torch.Tensor(u0.reshape(1, 1, -1)).to(device)
    Z0 = autoencoder.encoder(u0)
    Z0 = Z0[0, 0, :].cpu().detach().numpy()

    Z = odeint(dzdt, Z0, t_grid)
    Z = Z.reshape(1, Z.shape[0], Z.shape[1])
    Z = torch.Tensor(Z).to(device)

    Pred = autoencoder.decoder(Z).cpu()
    Pred = Pred[0, :, :].T.detach().numpy()

    return Pred


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

#device = 'cpu'

date = '07_29_2022_23_28'
bglasdi_results = np.load('//p/gpfs1/cbonnevi/Vlasov1D1V/results/bglasdi_' + date + '.npy', allow_pickle = True).item()

autoencoder_param = bglasdi_results['autoencoder_param']
t_grid = bglasdi_results['t_grid']

sindy_coef = bglasdi_results['sindy_coef']
gp_dictionnary = bglasdi_results['gp_dictionnary']

space_dim = 16384
time_dim = 251

n_hidden = len(autoencoder_param.keys()) // 4 - 1
hidden_units = [autoencoder_param['fc' + str(i + 1) + '_e.weight'].shape[0] for i in range(n_hidden)]
n_z = autoencoder_param['fc' + str(n_hidden + 1) + '_e.weight'].shape[0]

autoencoder = Autoencoder(space_dim, hidden_units, n_z)
autoencoder.load_state_dict(autoencoder_param)
autoencoder = autoencoder.to(device)

T_min, T_max = 0.9, 1.1
k_min, k_max = 1.0, 1.2

n_time_test = 22
burn = 2
time_fom = np.zeros(n_time_test)
time_rom = np.zeros(n_time_test)

param = np.random.rand(n_time_test, 2)
param[:, 0] = T_min + param[:, 0] * (T_max - T_min)
param[:, 1] = k_min + param[:, 1] * (k_max - k_min)

for i in range(n_time_test):

    T, k = param[i, 0], param[i, 1]

    tic = time.time()
    write_files(T, k, iproc = [2, 2])
    run_hypar(2)
    toc = time.time() - tic
    
    time_fom[i] = toc

    tic = time.time()
    u0 = initial_condition_func(T, k)
    _ = direct_mean_prediction(T, k, u0, autoencoder, t_grid, gp_dictionnary, sindy_coef, device)
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
