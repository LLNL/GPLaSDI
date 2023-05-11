import numpy as np
import torch
from train_framework import BayesianGLaSDI

torch.manual_seed(0)
np.random.rand(0)

def initial_condition(a, w, x_grid):

    return a * np.exp(-x_grid ** 2 / 2 / w / w)

class Autoencoder(torch.nn.Module):
    def __init__(self, space_dim, hidden_units, n_z):
        super(Autoencoder, self).__init__()

        fc1_e = torch.nn.Linear(space_dim, hidden_units[0])
        fc2_e = torch.nn.Linear(hidden_units[0], n_z)

        g_e = torch.nn.Sigmoid()

        self.fc1_e = fc1_e
        self.fc2_e = fc2_e
        self.g_e = g_e

        fc1_d = torch.nn.Linear(n_z, hidden_units[0])
        fc2_d = torch.nn.Linear(hidden_units[0], space_dim)

        self.fc1_d = fc1_d
        self.fc2_d = fc2_d


    def encoder(self, x):

        x = self.g_e(self.fc1_e(x))
        x = self.fc2_e(x)

        return x


    def decoder(self, x):

        x = self.g_e(self.fc1_d(x))
        x = self.fc2_d(x)

        return x


    def forward(self, x):

        x = Autoencoder.encoder(self, x)
        x = Autoencoder.decoder(self, x)

        return x


# Space Time Definition
time_dim = 1001
space_dim = 1001
x_min, x_max = -3, 3
t_max = 1

# Parameter Space Definition
a_min, a_max = 0.7, 0.9
w_min, w_max = 0.9, 1.1

n_a_grid = 21
n_w_grid = 21

# Autoencoder Definition
hidden_units = [100]
n_z = 5
autoencoder = Autoencoder(space_dim, hidden_units, n_z)

# Training Parameters
n_samples = 20
lr = 0.001
n_iter = 28000
n_greedy = 2000
max_greedy_iter = 28000
sindy_weight = 0.1
coef_weight = 1e-6

# Path
# path_data = '//p/gpfs1/cbonnevi/BurgersEqn1D/data/'
path_data = 'data/'
path_checkpoint = 'checkpoint/'
# path_results = '//p/gpfs1/cbonnevi/BurgersEqn1D/results/'
path_results = 'results/'

# CUDA
cuda = torch.cuda.is_available()

# --------------- Loading Parameters --------------- #

Dx = (x_max - x_min) / (space_dim - 1)
Dt = t_max / (time_dim - 1)
x_grid = np.linspace(x_min, x_max, space_dim)
t_grid = np.linspace(0, t_max, time_dim)
t_mesh, x_mesh = np.meshgrid(t_grid, x_grid)

data_train = np.load(path_data + 'data_train.npy', allow_pickle = True).item()
data_test = np.load(path_data + 'data_test.npy', allow_pickle = True).item()

model_parameters = {}

model_parameters['time_dim'] = time_dim
model_parameters['space_dim'] = space_dim
model_parameters['n_z'] = n_z

model_parameters['Dt'] = Dt
model_parameters['Dx'] = Dx
model_parameters['t_grid'] = t_grid
model_parameters['x_grid'] = x_grid

model_parameters['initial_condition'] = initial_condition

model_parameters['a_min'] = a_min
model_parameters['a_max'] = a_max
model_parameters['w_min'] = w_min
model_parameters['w_max'] = w_max
model_parameters['n_a_grid'] = n_a_grid
model_parameters['n_w_grid'] = n_w_grid

model_parameters['data_train'] = data_train
model_parameters['data_test'] = data_test

model_parameters['n_samples'] = n_samples
model_parameters['lr'] = lr
model_parameters['n_iter'] = n_iter
model_parameters['max_greedy_iter'] = max_greedy_iter
model_parameters['n_greedy'] = n_greedy
model_parameters['sindy_weight'] = sindy_weight
model_parameters['coef_weight'] = coef_weight

model_parameters['path_checkpoint'] = path_checkpoint
model_parameters['path_results'] = path_results

model_parameters['cuda'] = cuda

bglasdi = BayesianGLaSDI(autoencoder, model_parameters)
bglasdi.train()
