import numpy as np
import torch
import os
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

    [xv, yv] = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny), indexing='xy')

    zv = a * np.exp(-((xv - x0) ** 2 + (yv - y0) ** 2) / w)
    z = zv.flatten()

    u0 = np.vstack((z.reshape(1, -1), z.reshape(1, -1)))

    return u0


date = '07_27_2022_20_25'
bglasdi_results = np.load('//p/gpfs1/cbonnevi/BurgersEqn2D_1/results/bglasdi_' + date + '.npy', allow_pickle = True).item()

autoencoder_param = bglasdi_results['autoencoder_param']

X_train = bglasdi_results['final_X_train']
param_train = bglasdi_results['final_param_train']
param_grid = bglasdi_results['param_grid']
t_grid = bglasdi_results['t_grid']
Dt = bglasdi_results['Dt']

sindy_coef = bglasdi_results['sindy_coef']
gp_dictionnary = bglasdi_results['gp_dictionnary']

total_time = bglasdi_results['total_time']
start_train_phase = bglasdi_results['start_train_phase']
start_fom_phase = bglasdi_results['start_fom_phase']
end_train_phase = bglasdi_results['end_train_phase']
end_fom_phase = bglasdi_results['end_fom_phase']

data_test_1 = np.load('//p/gpfs1/cbonnevi/BurgersEqn2D_1/data/data_batch_compiled_1.npy', allow_pickle = True).item()
data_test_2 = np.load('//p/gpfs1/cbonnevi/BurgersEqn2D_1/data/data_batch_compiled_2.npy', allow_pickle = True).item()
data_test_3 = np.load('//p/gpfs1/cbonnevi/BurgersEqn2D_1/data/data_batch_compiled_3.npy', allow_pickle = True).item()
data_test_4 = np.load('//p/gpfs1/cbonnevi/BurgersEqn2D_1/data/data_batch_compiled_4.npy', allow_pickle = True).item()

X_test_1 = data_test_1['X_batch']
X_test_2 = data_test_2['X_batch']
X_test_3 = data_test_3['X_batch']
X_test_4 = data_test_4['X_batch']

X_test = np.concatenate((X_test_1, X_test_2, X_test_3, X_test_4), axis = 0)
n_init = bglasdi_results['n_init']
n_samples = bglasdi_results['n_samples']
n_a_grid = bglasdi_results['n_a_grid']
n_w_grid = bglasdi_results['n_w_grid']
a_grid = bglasdi_results['a_grid']
w_grid = bglasdi_results['w_grid']

ic = 2
nx, ny = 60, 60

space_dim = 7200
time_dim = 201

n_hidden = len(autoencoder_param.keys()) // 4 - 1
hidden_units = [autoencoder_param['fc' + str(i + 1) + '_e.weight'].shape[0] for i in range(n_hidden)]
n_z = autoencoder_param['fc' + str(n_hidden + 1) + '_e.weight'].shape[0]

print('Hidden Units:')
for i in range(n_hidden): print(hidden_units[i])
print('n_z: ' + str(n_z))
  
autoencoder = Autoencoder(space_dim, hidden_units, n_z)
autoencoder.load_state_dict(autoencoder_param)

Z = autoencoder.encoder(X_train).detach().numpy()
Z0 = initial_condition_latent(param_grid, initial_condition_func, ic, nx, ny, autoencoder)
Zis, gp_dictionnary, interpolation_data, sindy_coef, n_coef, coef_samples = simulate_interpolated_sindy(param_grid, Z0, t_grid, n_samples, Dt, Z, param_train)
'''
n_coef = n_z * (n_z + 1)
coef_samples = [interpolate_coef_matrix(gp_dictionnary, param_grid[i, :], n_samples, n_coef, sindy_coef) for i in range(param_grid.shape[0])]
Zis = [simulate_uncertain_sindy(gp_dictionnary, param_grid[i, 0], n_samples, Z0[i], t_grid, sindy_coef, n_coef, coef_samples[i]) for i in range(param_grid.shape[0])]
'''
max_e_relative, max_e_relative_mean, max_std = compute_errors(n_a_grid, n_w_grid, Zis, autoencoder, X_test)

errors = {'max_e_relative' : max_e_relative, 'max_e_relative_mean' : max_e_relative_mean, 'max_std' : max_std, 'a_grid' : a_grid, 'w_grid' : w_grid, 'param_train' : param_train, 'n_init' : n_init}
np.save('//p/gpfs1/cbonnevi/BurgersEqn2D_1/errors/errors_' + date + '.npy', errors)