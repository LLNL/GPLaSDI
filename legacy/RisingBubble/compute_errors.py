import numpy as np
import torch
import os

from train_framework import *
from utils.utils_errors import *

torch.manual_seed(1)

def interpolate_coef_matrix_mean(gp_dictionnary, param, n_coef, sindy_coef, normalize):

    coef_samples = []
    coef_x, coef_y = sindy_coef[0].shape
    if param.ndim == 1:
        param = param.reshape(1, -1)

    gp_pred = eval_gp(gp_dictionnary, param, n_coef, normalize)
    
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


date = '03_29_2023_05_29'
bglasdi_results = np.load('//p/gpfs1/cbonnevi/RisingBubble/results/noise_000/bglasdi_' + date + '.npy', allow_pickle = True).item()

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

data_test = np.load('//p/gpfs1/cbonnevi/RisingBubble/data/data_test_batch_1.npy', allow_pickle = True).item()
X_test = data_test['X_batch']

for i in range(2, 19):
    data_test = np.load('//p/gpfs1/cbonnevi/RisingBubble/data/data_test_batch_' + str(i) + '.npy', allow_pickle = True).item()
    X_test = np.concatenate((X_test, data_test['X_batch']), axis = 0)

X_test -= 300

def initial_condition_func(tc, rc):
     
    param = np.array([[tc, rc]])
    m = np.where((1 * (np.round(param_grid, 4) == np.round(param[0, :], 4))).sum(1) == 2)[0][0]
    theta = X_test[m, 0:1, :]

    return theta			

n_init = bglasdi_results['n_init']
n_samples = 50
n_a_grid = bglasdi_results['n_a_grid']
n_w_grid = bglasdi_results['n_w_grid']
a_grid = bglasdi_results['a_grid']
w_grid = bglasdi_results['w_grid']

space_dim = 10100
time_dim = 301

n_hidden = len(autoencoder_param.keys()) // 4 - 1
hidden_units = [autoencoder_param['fc' + str(i + 1) + '_e.weight'].shape[0] for i in range(n_hidden)]
n_z = autoencoder_param['fc' + str(n_hidden + 1) + '_e.weight'].shape[0]

print('Hidden Units:')
for i in range(n_hidden): print(hidden_units[i])
print('n_z: ' + str(n_z))
  
autoencoder = Autoencoder(space_dim, hidden_units, n_z)
autoencoder.load_state_dict(autoencoder_param)

interpolation_data = build_interpolation_data(sindy_coef, param_train)
gp_dictionnary = fit_gps(interpolation_data, normalize = True)
n_coef = interpolation_data['n_coef']

Z0 = initial_condition_latent(param_grid, initial_condition_func, autoencoder)
n_coef = n_z * (n_z + 1)
coef_samples = [interpolate_coef_matrix(gp_dictionnary, param_grid[i, :], n_samples, n_coef, sindy_coef, normalize = True) for i in range(param_grid.shape[0])]
Zis = [simulate_uncertain_sindy(gp_dictionnary, param_grid[i, 0], n_samples, Z0[i], t_grid, sindy_coef, n_coef, coef_samples[i], normalize = True) for i in range(param_grid.shape[0])]

#n_samples = 1
#coef_samples = [interpolate_coef_matrix_mean(gp_dictionnary, param_grid[i, :], n_coef, sindy_coef, normalize = True) for i in range(param_grid.shape[0])]
#Zis = [simulate_uncertain_sindy(gp_dictionnary, param_grid[i, 0], n_samples, Z0[i], t_grid, sindy_coef, n_coef, coef_samples[i], normalize = True) for i in range(param_grid.shape[0])]

m = 200
Z_m = torch.Tensor(Zis[m])
X_pred_m = autoencoder.decoder(Z_m).detach().numpy()
e_relative_m_mean = np.linalg.norm((X_test[m, :, :] - X_pred_m.mean(0)), axis = 1) / np.linalg.norm(X_test[m, :, :], axis = 1)
max_e_relative_m_mean = e_relative_m_mean.max()

print(max_e_relative_m_mean)

max_e_relative, max_e_relative_mean, max_std = compute_errors(n_a_grid, n_w_grid, Zis, autoencoder, X_test)

errors = {'max_e_relative' : max_e_relative, 'max_e_relative_mean' : max_e_relative_mean, 'max_std' : max_std, 'a_grid' : a_grid, 'w_grid' : w_grid, 'param_train' : param_train, 'n_init' : n_init}
np.save('//p/gpfs1/cbonnevi/RisingBubble/errors/errors_' + date + '_use_samples.npy', errors)
