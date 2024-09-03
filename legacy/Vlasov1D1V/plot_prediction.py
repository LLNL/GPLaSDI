import numpy as np
import torch
from train_framework import *
from utils.utils_sindy import *
from utils.utils_plot import *

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

X_test = np.concatenate((X_test_1, X_test_2, X_test_3, X_test_4), axis = 0)
n_init = bglasdi_results['n_init']
n_samples = bglasdi_results['n_samples']
n_T_grid = bglasdi_results['n_a_grid']
n_k_grid = bglasdi_results['n_w_grid']
T_grid = bglasdi_results['a_grid']
k_grid = bglasdi_results['w_grid']

space_dim = 16384
time_dim = 251

n_hidden = len(autoencoder_param.keys()) // 4 - 1
hidden_units = [autoencoder_param['fc' + str(i + 1) + '_e.weight'].shape[0] for i in range(n_hidden)]
n_z = autoencoder_param['fc' + str(n_hidden + 1) + '_e.weight'].shape[0]

autoencoder = Autoencoder(space_dim, hidden_units, n_z)
autoencoder.load_state_dict(autoencoder_param)

Z = autoencoder.encoder(X_train)
Z0 = initial_condition_latent(param_grid, initial_condition_func, autoencoder)
Zis, gp_dictionnary, interpolation_data, sindy_coef, n_coef, coef_samples = simulate_interpolated_sindy(param_grid, Z0, t_grid, n_samples, Dt, Z, param_train, n_z)

T, k =  0.9, 1.04
scale = 1

param = np.array([[T, k]])

u0 = initial_condition_func(T, k)
z0 = autoencoder.encoder(torch.Tensor(u0.reshape(1, 1, -1)))
z0 = z0[0, 0, :].detach().numpy()
Z = simulate_uncertain_sindy(gp_dictionnary, param, n_samples, z0, t_grid, sindy_coef, n_coef, n_z, poly_order = 1)

time_step = 50
fig, pred, true, Z = plot_prediction(T, k, time_step, scale, initial_condition_func, autoencoder, gp_dictionnary, n_samples, t_grid, sindy_coef, n_coef, nx, nv, x_mesh, v_mesh, X_test, param_grid, Z)
prediction_data = {'pred' : pred, 'true' : true, 'Z' : Z, 't_grid' : t_grid, 'x_mesh' : x_mesh, 'v_mesh' : v_mesh, 'param' : param}

np.save('prediction/pred_T_' + str(T) + '_k_' + str(k) + '.npy', prediction_data)

'''
for time_step in range(250):
    fig, _, _, _ = plot_prediction(T, k, time_step, scale, initial_condition_func, autoencoder, gp_dictionnary, n_samples, t_grid, sindy_coef, n_coef, nx, nv, x_mesh, v_mesh, X_test, param_grid, Z)
    fig.savefig('plot/case2/Step_{time_step:03d}_T{T:1.2f}_k{k:1.2f}.jpeg'.format(time_step = time_step, T = T, k = k), bbox = 'tight')
'''


