import numpy as np
import torch
import os

from train_framework import *
from utils.utils_sindy import *
from utils.utils_errors import *


torch.manual_seed(1)

def interpolate_coef_matrix_mean(gp_dictionnary, param, n_coef, sindy_coef):

    coef_samples = []
    coef_x, coef_y = sindy_coef[0].shape
    if param.ndim == 1:
        param = param.reshape(1, -1)

    gp_pred = eval_gp(gp_dictionnary, param, n_coef)

    coeff_matrix = np.zeros([coef_x, coef_y])
    k = 1
    for i in range(coef_x):
        for j in range(coef_y):
            mean = gp_pred['coef_' + str(k)]['mean']

            coeff_matrix[i, j] = mean
            k += 1

    coef_samples.append(coeff_matrix)

    return coef_samples



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

date = '03_27_2023_05_43'
bglasdi_results = np.load('//p/gpfs1/cbonnevi/Vlasov1D1V/results/bglasdi_' + date + '.npy', allow_pickle = True).item()

nx = 128
nv = 128

x_min = 0
x_max = 6.2340979219672459E+00

v_min = -7
v_max = 6.8906250000000000E+00

x_mesh = np.linspace(x_min, x_max, nx)
v_mesh = np.linspace(v_min, v_max, nv)

x_mesh, v_mesh = np.meshgrid(x_mesh, v_mesh)

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

data_test_1 = np.load('//p/gpfs1/cbonnevi/Vlasov1D1V/data/data_batch_compiled_1.npy', allow_pickle = True).item()
data_test_2 = np.load('//p/gpfs1/cbonnevi/Vlasov1D1V/data/data_batch_compiled_2.npy', allow_pickle = True).item()
data_test_3 = np.load('//p/gpfs1/cbonnevi/Vlasov1D1V/data/data_batch_compiled_3.npy', allow_pickle = True).item()
data_test_4 = np.load('//p/gpfs1/cbonnevi/Vlasov1D1V/data/data_batch_compiled_4.npy', allow_pickle = True).item()
data_test_5 = np.load('//p/gpfs1/cbonnevi/Vlasov1D1V/data/data_batch_compiled_5.npy', allow_pickle = True).item()

X_test_1 = data_test_1['X_batch']
X_test_2 = data_test_2['X_batch']
X_test_3 = data_test_3['X_batch']
X_test_4 = data_test_4['X_batch']
X_test_5 = data_test_5['X_batch']

X_test = np.concatenate((X_test_1, X_test_2, X_test_3, X_test_4, X_test_5), axis = 0)
n_init = bglasdi_results['n_init']
n_samples = bglasdi_results['n_samples']
n_T_grid = bglasdi_results['n_a_grid']
n_k_grid = bglasdi_results['n_w_grid']
T_grid = bglasdi_results['a_grid']
k_grid = bglasdi_results['w_grid']

n_test = X_test.shape[0]
space_dim = 16384
time_dim = 251

poly_order = 1
sine_term = False

n_hidden = len(autoencoder_param.keys()) // 4 - 1
hidden_units = [autoencoder_param['fc' + str(i + 1) + '_e.weight'].shape[0] for i in range(n_hidden)]
n_z = autoencoder_param['fc' + str(n_hidden + 1) + '_e.weight'].shape[0]
library_size = lib_size(n_z, poly_order)

print('Hidden Units:')
for i in range(n_hidden): print(hidden_units[i])
print('n_z: ' + str(n_z))
  
autoencoder = Autoencoder(space_dim, hidden_units, n_z)
autoencoder.load_state_dict(autoencoder_param)

Z = autoencoder.encoder(X_train).detach().numpy()
Z0 = initial_condition_latent(param_grid, initial_condition_func, autoencoder)

interpolation_data = build_interpolation_data(sindy_coef, param_train)
gp_dictionnary = fit_gps(interpolation_data)
n_coef = interpolation_data['n_coef']

n_samples = 50
coef_samples = [interpolate_coef_matrix(gp_dictionnary, param_grid[i, :], n_samples, n_coef, sindy_coef) for i in range(n_test)]
Zis = [simulate_uncertain_sindy(gp_dictionnary, param_grid[i, :], n_samples, Z0[i], t_grid, sindy_coef, n_coef, n_z, poly_order, library_size, sine_term, coef_samples[i]) for i in range(n_test)]

#coef_samples = [interpolate_coef_matrix_mean(gp_dictionnary, param_grid[i, :], n_coef, sindy_coef) for i in range(n_test)]
#n_samples = 1
#Zis = [simulate_uncertain_sindy(gp_dictionnary, param_grid[i, :], n_samples, Z0[i], t_grid, sindy_coef, n_coef, n_z, poly_order, library_size, sine_term, coef_samples[i]) for i in range(n_test)]

max_e_relative, max_e_relative_mean, max_std = compute_errors(n_T_grid, n_k_grid, Zis, autoencoder, X_test)

errors = {'max_e_relative' : max_e_relative, 'max_e_relative_mean' : max_e_relative_mean, 'max_std' : max_std, 'T_grid' : T_grid, 'k_grid' : k_grid, 'param_train' : param_train, 'n_init' : n_init}
np.save('//p/gpfs1/cbonnevi/Vlasov1D1V/errors/errors_' + date + '_use_samples.npy', errors)
