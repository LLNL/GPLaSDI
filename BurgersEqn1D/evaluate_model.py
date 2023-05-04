import torch
import numpy as np
from utils import *

def initial_condition(a, w, x_grid):

    return a * np.exp(-x_grid ** 2 / 2 / w / w)

class Autoencoder(torch.nn.Module):
    def __init__(self, space_dim, hidden_units, n_z):
        super(Autoencoder, self).__init__()

        fc1_e = torch.nn.Linear(space_dim, hidden_units[0])
        fc2_e = torch.nn.Linear(hidden_units[0], hidden_units[1])
        fc3_e = torch.nn.Linear(hidden_units[1], n_z)

        g_e = torch.nn.Sigmoid()

        self.fc1_e = fc1_e
        self.fc2_e = fc2_e
        self.fc3_e = fc3_e
        self.g_e = g_e

        fc1_d = torch.nn.Linear(n_z, hidden_units[1])
        fc2_d = torch.nn.Linear(hidden_units[1], hidden_units[0])
        fc3_d = torch.nn.Linear(hidden_units[0], space_dim)

        self.fc1_d = fc1_d
        self.fc2_d = fc2_d
        self.fc3_d = fc3_d


    def encoder(self, x):

        x = self.g_e(self.fc1_e(x))
        x = self.g_e(self.fc2_e(x))
        x = self.fc3_e(x)

        return x


    def decoder(self, x):

        x = self.g_e(self.fc1_d(x))
        x = self.g_e(self.fc2_d(x))
        x = self.fc3_d(x)

        return x


    def forward(self, x):

        x = Autoencoder.encoder(self, x)
        x = Autoencoder.decoder(self, x)

        return x


date = '6_22_2022_913'
bglasdi_results = np.load('results/bglasdi_' + date + '.npy', allow_pickle = True).item()

autoencoder_param = bglasdi_results['autencoder_param']

X_train = bglasdi_results['final_X_train']
param_train = bglasdi_results['final_param_train']
param_grid = bglasdi_results['param_grid']

sindy_coef = bglasdi_results['sindy_coef']
gp_dictionnary = bglasdi_results['gp_dictionnary']

n_init = bglasdi_results['n_init']
n_samples = 20
n_a_grid = 21
n_w_grid = 21
a_min, a_max = 0.7, 0.9
w_min, w_max = 0.9, 1.1
a_grid = np.linspace(a_min, a_max, n_a_grid)
w_grid = np.linspace(w_min, w_max, n_w_grid)
a_grid, w_grid = np.meshgrid(a_grid, w_grid)

t_grid = bglasdi_results['t_grid']
x_grid = bglasdi_results['x_grid']
t_mesh, x_mesh = np.meshgrid(t_grid, x_grid)
Dt = bglasdi_results['Dt']
Dx = bglasdi_results['Dx']

total_time = bglasdi_results['total_time']
start_train_phase = bglasdi_results['start_train_phase']
start_fom_phase = bglasdi_results['start_fom_phase']
end_train_phase = bglasdi_results['end_train_phase']
end_fom_phase = bglasdi_results['end_fom_phase']

data_test = np.load('data/data_test.npy', allow_pickle = True).item()
X_test = data_test['X_test']

time_dim, space_dim = t_grid.shape[0], x_grid.shape[0]

n_hidden = len(autoencoder_param.keys()) // 4 - 1
hidden_units = [autoencoder_param['fc' + str(i + 1) + '_e.weight'].shape[0] for i in range(n_hidden)]
n_z = autoencoder_param['fc' + str(n_hidden + 1) + '_e.weight'].shape[0]

autoencoder = Autoencoder(space_dim, hidden_units, n_z)
autoencoder.load_state_dict(autoencoder_param)

Z = autoencoder.encoder(X_train).detach().numpy()
Z0 = initial_condition_latent(param_grid, initial_condition, x_grid, autoencoder)
Zis, gp_dictionnary, interpolation_data, sindy_coef, n_coef, coef_samples = simulate_interpolated_sindy(param_grid, Z0, t_grid, n_samples, Dt, Z, param_train)

max_e_residual, max_e_relative, max_e_relative_mean, max_std = compute_errors(n_a_grid, n_w_grid, Zis, autoencoder, X_test, Dt, Dx)
plot_errors(max_e_relative, n_a_grid, n_w_grid, a_grid, w_grid, param_train, n_init)

a, w = 0.8, 1.0
maxk = 10
convergence_threshold = 1e-8
param = np.array([[a, w]])
u0 = initial_condition(a, w, x_grid)
true = solver(u0, maxk, convergence_threshold, time_dim - 1, space_dim, Dt, Dx)
true = true.T
scale = 1

z0 = autoencoder.encoder(torch.Tensor(u0.reshape(1, 1, -1)))
z0 = z0[0, 0, :].detach().numpy()

plot_prediction(param, autoencoder, gp_dictionnary, n_samples, z0, t_grid, sindy_coef, n_coef, t_mesh, x_mesh, scale, true, Dt, Dx)