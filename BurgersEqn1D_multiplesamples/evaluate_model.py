import torch
import numpy as np
from utils import *
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

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


def simulate_uncertain_sindy_mean(gp_dictionnary, param, z0, t_grid, sindy_coef, n_coef, coef_samples = None):

    if coef_samples is None:
        coef_samples = interpolate_coef_matrix_mean(gp_dictionnary, param, n_coef, sindy_coef)

    Z0 = [z0]
    Z = simulate_sindy(coef_samples, Z0, t_grid)

    return Z


def simulate_interpolated_sindy_mean(param_test, Z0, t_grid, Dt, Z, param_train):

    dZdt, Z = compute_sindy_data(Z, Dt)
    sindy_coef = solve_sindy(dZdt, Z)
    interpolation_data = build_interpolation_data(sindy_coef, param_train)
    gp_dictionnary = fit_gps(interpolation_data)
    n_coef = interpolation_data['n_coef']

    coef_samples = [interpolate_coef_matrix_mean(gp_dictionnary, param_test[i, :], n_coef, sindy_coef) for i in range(param_test.shape[0])]

    Z_simulated = [simulate_uncertain_sindy_mean(gp_dictionnary, param_test[i, 0], Z0[i], t_grid, sindy_coef, n_coef, coef_samples[i]) for i in range(param_test.shape[0])]

    return Z_simulated, gp_dictionnary, interpolation_data, sindy_coef, n_coef, coef_samples



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


date = '06_20_2024_15_28'
bglasdi_results = np.load('results/bglasdi_' + date + '.npy', allow_pickle = True).item()

autoencoder_param = bglasdi_results['autoencoder_param']

X_train = bglasdi_results['final_X_train']
param_train = bglasdi_results['final_param_train']
param_test = bglasdi_results['param_test']

sindy_coef = bglasdi_results['sindy_coef']
gp_dictionnary = bglasdi_results['gp_dictionnary']

n_init = bglasdi_results['n_init']
n_samples = 20
# n_a_grid = 21
# n_w_grid = 21
# a_min, a_max = 0.7, 0.9
# w_min, w_max = 0.9, 1.1
# a_grid = np.linspace(a_min, a_max, n_a_grid)
# w_grid = np.linspace(w_min, w_max, n_w_grid)
# a_grid, w_grid = np.meshgrid(a_grid, w_grid)

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
# X_test = data_test['X_test']

data_sim = np.load('data/data_sim2.npy', allow_pickle = True).item()
X_sim = data_sim['X_sim']
# X_sim = X_sim[5:-1,:,:]

time_dim, space_dim = t_grid.shape[0], x_grid.shape[0]

n_hidden = len(autoencoder_param.keys()) // 4 - 1
hidden_units = [autoencoder_param['fc' + str(i + 1) + '_e.weight'].shape[0] for i in range(n_hidden)]
n_z = autoencoder_param['fc' + str(n_hidden + 1) + '_e.weight'].shape[0]

autoencoder = Autoencoder(space_dim, hidden_units, n_z)
autoencoder.load_state_dict(autoencoder_param)

Z = autoencoder.encoder(X_train).detach().numpy()
Z0 = initial_condition_latent(param_test, initial_condition, x_grid, autoencoder)
Zis_samples, gp_dictionnary, interpolation_data, sindy_coef, n_coef, coef_samples = simulate_interpolated_sindy(param_test, Z0, t_grid, n_samples, Dt, Z, param_train)
Zis_mean, _, _, _, _, _ = simulate_interpolated_sindy_mean(param_test, Z0, t_grid, Dt, Z, param_train)

_, _, max_e_relative_mean, _ = compute_errors_vec(param_test, Zis_mean, autoencoder, X_sim, Dt, Dx)
_, _, _, max_std = compute_errors_vec(param_test, Zis_samples, autoencoder, X_sim, Dt, Dx)

gp_pred = eval_gp(gp_dictionnary, param_test, n_coef)

#plot mean and standard deviation of each sindy coefficient
coef_x, coef_y = sindy_coef[0].shape
fig1, axs1 = plt.subplots(coef_y, coef_x, figsize = (15, 13)) #will be means
fig2, axs2 = plt.subplots(coef_y, coef_x, figsize = (15, 13)) #will be std

refine = 10
cm = plt.cm.jet

k = 1

for i in range(coef_y):
    for j in range(coef_x):

        mean = gp_pred['coef_' + str(k)]['mean']
        std = gp_pred['coef_' + str(k)]['std']
        
        im1 = axs1[i,j].scatter(param_test[:, 0], param_test[:, 1], c=mean)
        im2 = axs2[i,j].scatter(param_test[:, 0], param_test[:, 1], c=std)

        fig1.colorbar(im1, ax = axs1[i,j],ticks = np.array([mean.min(), mean.max()]), format = '%2.1f')
        fig2.colorbar(im2, ax = axs2[i,j],ticks = np.array([std.min(), std.max()]), format = '%2.1f')
        axs1[i,j].set_title(r'$\mu^*_{' + str(i + 1) + str(j + 1) + '}$')
        axs1[i, j].axis('equal')
        axs1[i, j].set_xlim([np.min(param_test[:, 0]) - 0.02, np.max(param_test[:, 0]) + 0.02])
        axs1[i, j].set_ylim([np.min(param_test[:, 1]) - 0.02, np.max(param_test[:, 1]) + 0.02])
        #axs1[i, j].invert_yaxis()
        axs1[i, j].get_xaxis().set_visible(False)
        axs1[i, j].get_yaxis().set_visible(False)
        
        axs2[i,j].set_title(r'$\sqrt{\Sigma^*_{' + str(i + 1) + str(j + 1) + '}}$')
        axs2[i, j].axis('equal')
        axs2[i, j].set_xlim([np.min(param_test[:, 0]) - 0.02, np.max(param_test[:, 0]) + 0.02])
        axs2[i, j].set_ylim([np.min(param_test[:, 1]) - 0.02, np.max(param_test[:, 1]) + 0.02])
        #axs2[i, j].invert_yaxis()
        axs2[i, j].get_xaxis().set_visible(False)
        axs2[i, j].get_yaxis().set_visible(False)
        # if i != coef_x - 1:
        #     plt.gca().get_xaxis().set_visible(False)

        # if j != 0:
        #     plt.gca().get_yaxis().set_visible(False)

        if i == coef_y - 1:
            axs1[i, j].set_xlabel('a')
            axs1[i, j].get_xaxis().set_visible(True)
            axs2[i, j].set_xlabel('a')
            axs2[i, j].get_xaxis().set_visible(True)

        if j == 0:
            axs1[i, j].set_ylabel('w')
            axs1[i, j].get_yaxis().set_visible(True)
            axs2[i, j].set_ylabel('w')
            axs2[i, j].get_yaxis().set_visible(True)

        k += 1

        

fig, ax = plt.subplots(1, 1, figsize = (10, 10))

cmap = LinearSegmentedColormap.from_list('rg', ['C0', 'w', 'C3'], N = 256)
im = ax.scatter(param_test[:,0],param_test[:,1],c = max_e_relative_mean * 100, cmap = cmap,s=100)
fig.colorbar(im, ax = ax, fraction = 0.04)
ax.set_xlabel('a', fontsize=15)
ax.set_ylabel('w', fontsize=15)
ax.set_title('Maximum Relative Error', fontsize=25)
ax.scatter(param_train[:, 0], param_train[:, 1],marker='X',color='k',s=200)


fig, ax = plt.subplots(1, 1, figsize = (10, 10))

cmap = LinearSegmentedColormap.from_list('rg', ['C0', 'w', 'C3'], N = 256)
im = ax.scatter(param_test[:,0],param_test[:,1],c = max_std * 100, cmap = cmap,s=100)
fig.colorbar(im, ax = ax, fraction = 0.04)
ax.set_xlabel('a', fontsize=15)
ax.set_ylabel('w', fontsize=15)
ax.set_title(r'max$_{(t,x)}\sqrt{V[\tilde{u}_{\xi^*}]}$', fontsize=25)
ax.scatter(param_train[:, 0], param_train[:, 1],marker='X',color='k',s=200)


a, w = 0.8, .95
maxk = 10
convergence_threshold = 1e-8
param = np.array([[a, w]])
u0 = initial_condition(a, w, x_grid)
true = solver(u0, maxk, convergence_threshold, time_dim - 1, space_dim, Dt, Dx)
true = true.T
scale = 1

z0 = autoencoder.encoder(torch.Tensor(u0.reshape(1, 1, -1)))
z0 = z0[0, 0, :].detach().numpy()

Z = simulate_uncertain_sindy(gp_dictionnary, param, n_samples, z0, t_grid, sindy_coef, n_coef)
Z_mean = Z.mean(0)
Z_std = Z.std(0)


pred = autoencoder.decoder(torch.Tensor(Z)).detach().numpy()
pred_std = pred.std(0)

plot_prediction(param, autoencoder, gp_dictionnary, n_samples, z0, t_grid, sindy_coef, n_coef, t_mesh, x_mesh, scale, true, Dt, Dx)

DEBUG_FLAG = 1
