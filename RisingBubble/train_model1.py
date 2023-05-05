import numpy as np
import torch
import os
from train_framework import BayesianGLaSDI

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

# Space Time Definition
time_dim = 301
space_dim = 10100
t_max = 3 

Dt = 0.01

# Parameter Space Definition
tc_min, tc_max = 0.5, 0.6
rc_min, rc_max = 150, 160

n_tc_grid = 21
n_rc_grid = 21

noise = 0.0

# Autoencoder Definition
hidden_units = [1000, 200, 50, 20]
n_z = 5
autoencoder = Autoencoder(space_dim, hidden_units, n_z)

# Training Parameters
n_samples = 20
lr = 0.0001
n_iter = 320000
n_greedy = 40000
sindy_weight = 0.25
coef_weight = 1e-6
normalize = True

n_proc = 4

# Path
path_data = '//p/gpfs1/cbonnevi/RisingBubble/data/'
path_checkpoint = 'checkpoint/'
path_results = '//p/gpfs1/cbonnevi/RisingBubble/results/noise_000/'

# CUDA
cuda = torch.cuda.is_available()

# Best Loss
keep_best_loss = True

# --------------- Loading Parameters --------------- #

t_grid = np.linspace(0, t_max, time_dim)

data_train = np.load(path_data + 'data_train_noise_000.npy', allow_pickle = True).item()

model_parameters = {}

model_parameters['time_dim'] = time_dim
model_parameters['space_dim'] = space_dim
model_parameters['n_z'] = n_z
model_parameters['t_grid'] = t_grid

model_parameters['Dt'] = Dt


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

model_parameters['initial_condition'] = initial_condition_func

model_parameters['n_proc'] = n_proc

model_parameters['a_min'] = tc_min
model_parameters['a_max'] = tc_max
model_parameters['w_min'] = rc_min
model_parameters['w_max'] = rc_max
model_parameters['n_a_grid'] = n_tc_grid
model_parameters['n_w_grid'] = n_rc_grid

model_parameters['noise'] = noise

model_parameters['data_train'] = data_train

tc_grid = np.linspace(tc_min, tc_max, n_tc_grid)
rc_grid = np.linspace(rc_min, rc_max, n_rc_grid)
tc_grid, rc_grid = np.meshgrid(tc_grid, rc_grid)
param_grid = np.hstack((tc_grid.reshape(-1, 1), rc_grid.reshape(-1, 1)))
n_test = param_grid.shape[0]

model_parameters['param_grid'] = param_grid
model_parameters['n_test'] = n_test

model_parameters['n_samples'] = n_samples
model_parameters['lr'] = lr
model_parameters['n_iter'] = n_iter
model_parameters['n_greedy'] = n_greedy
model_parameters['sindy_weight'] = sindy_weight
model_parameters['coef_weight'] = coef_weight
model_parameters['normalize'] = normalize

model_parameters['path_checkpoint'] = path_checkpoint
model_parameters['path_results'] = path_results
model_parameters['keep_best_loss'] = keep_best_loss

model_parameters['cuda'] = cuda

bglasdi = BayesianGLaSDI(autoencoder, model_parameters)
bglasdi.train()
