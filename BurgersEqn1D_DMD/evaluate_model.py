import torch
import numpy as np
from utils2 import *
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy.linalg as LA


def interpolate_coef_matrix_mean(gp_dictionnary, param, n_coef, sindy_coef):

    coef_samples = []
    coef_x, coef_y = sindy_coef[0].shape
    if param.ndim == 1:
        param = param.reshape(1, -1)

    gp_pred = eval_gp(gp_dictionnary, param, n_coef)
    
    U, _, _ = LA.svd(sindy_coef[0])


    coeff_matrix = np.zeros([coef_x, coef_y])
    k = 1
    for i in range(coef_x):
        for j in range(coef_y):
            mean = gp_pred['coef_' + str(k)]['mean']
            # if k == 1:
            #     print(mean)
            # coeff_matrix[i, j] = mean
            coeff_matrix[i, j] = mean[0]
            k += 1

    # coef_samples.append(U.T@coeff_matrix@U)
    coef_samples.append(coeff_matrix)
    # coef_samples.append( coeff_matrix*0.001 + np.eye(coeff_matrix.shape[0]) )

    return coef_samples


def simulate_uncertain_sindy_mean(gp_dictionnary, param, z0, t_grid, sindy_coef, n_coef, coef_samples = None):

    if coef_samples is None:
        coef_samples = interpolate_coef_matrix_mean(gp_dictionnary, param, n_coef, sindy_coef)

    Z0 = [z0]
    Z = simulate_sindy(coef_samples, Z0, t_grid)

    return Z


def simulate_interpolated_sindy_mean(param_grid, Z0, t_grid, Dt, Z, param_train):

    # dZdt, Z = compute_sindy_data(Z, Dt)
    # sindy_coef = solve_sindy(dZdt, Z)
    sindy_coef = solve_sindy(Z)
    interpolation_data = build_interpolation_data(sindy_coef, param_train)
    gp_dictionnary = fit_gps(interpolation_data)
    n_coef = interpolation_data['n_coef']

    coef_samples = [interpolate_coef_matrix_mean(gp_dictionnary, param_grid[i, :], n_coef, sindy_coef) for i in range(param_grid.shape[0])]

    Z_simulated = [simulate_uncertain_sindy_mean(gp_dictionnary, param_grid[i, 0], Z0[i], t_grid, sindy_coef, n_coef, coef_samples[i]) for i in range(param_grid.shape[0])]

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
        # g_e = torch.nn.ReLU()

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


#date = '05_11_2023_12_50'

#sigmoid
bglasdi_results = np.load('results/bglasdi_08_09_2024_16_38.npy', allow_pickle = True).item()

#ReLU
bglasdi_results = np.load('results/bglasdi_08_09_2024_17_07.npy', allow_pickle = True).item()

#Softplus
bglasdi_results = np.load('results/bglasdi_08_09_2024_17_26.npy', allow_pickle = True).item()

# bglasdi_results = np.load('results/bglasdi_08_09_2024_18_50.npy', allow_pickle = True).item()

#this run ran and sampled a lot of pts
# bglasdi_results = np.load('results/bglasdi_08_09_2024_19_26.npy', allow_pickle = True).item()

#training on final run
#i forgot to label this one, but for some reaosn it's bad
# bglasdi_results = np.load('results/bglasdi_08_09_2024_25_47.npy', allow_pickle = True).item()
#best one, only trained for 2k iterations on final data
# bglasdi_results = np.load('results/bglasdi_08_09_2024_26_22.npy', allow_pickle = True).item()


# bglasdi_results = np.load('results/bglasdi_08_15_2024_18_31.npy', allow_pickle = True).item()

# bglasdi_results = np.load('results/bglasdi_08_16_2024_18_39.npy', allow_pickle = True).item()

#softplus
#1E-5
bglasdi_results = np.load('results/bglasdi_09_13_2024_11_35.npy', allow_pickle = True).item()
#1E-8
bglasdi_results = np.load('results/bglasdi_09_13_2024_11_41.npy', allow_pickle = True).item()

#sigmoid
bglasdi_results = np.load('results/bglasdi_02_23_2024_21_11.npy', allow_pickle = True).item()


# bglasdi_results = bglasdi
autoencoder_param = bglasdi_results['autoencoder_param']

X_train = bglasdi_results['final_X_train']
param_train = bglasdi_results['final_param_train']
param_grid = bglasdi_results['param_grid']

sindy_coef = bglasdi_results['sindy_coef']
# gp_dictionnary = Nonegp_dictionnary = bglasdi_results['gp_dictionnary']
gp_dictionnary = None


n_init = bglasdi_results['n_init']
n_samples = 20
n_a_grid = 21
n_w_grid = 21
# n_a_grid = 5
# n_w_grid = 5
a_min, a_max = 0.7, 0.9
w_min, w_max = 0.9, 1.1
a_grid = np.linspace(a_min, a_max, n_a_grid)
w_grid = np.linspace(w_min, w_max, n_w_grid)
a_grid, w_grid = np.meshgrid(a_grid, w_grid)
param_grid = np.hstack((a_grid.reshape(-1, 1), w_grid.reshape(-1, 1)))
n_test = param_grid.shape[0]

param_grid_orig = bglasdi_results['param_grid']

k = []
for i in range(len(param_grid)):
    for j in range(len(param_grid_orig)):
        if all(param_grid[i,:] == param_grid_orig[j,:]):
            k.append(j)
            break

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
X_test_orig = data_test['X_test']
X_test = X_test_orig[k,:,:]


time_dim, space_dim = t_grid.shape[0], x_grid.shape[0]

n_hidden = len(autoencoder_param.keys()) // 4 - 1
hidden_units = [autoencoder_param['fc' + str(i + 1) + '_e.weight'].shape[0] for i in range(n_hidden)]
n_z = autoencoder_param['fc' + str(n_hidden + 1) + '_e.weight'].shape[0]

autoencoder = Autoencoder(space_dim, hidden_units, n_z)
autoencoder.load_state_dict(autoencoder_param)

Z = autoencoder.encoder(X_train).detach().numpy()
Z0 = initial_condition_latent(param_grid, initial_condition, x_grid, autoencoder)
Zis_samples, gp_dictionnary, interpolation_data, sindy_coef, n_coef, coef_samples = simulate_interpolated_sindy(param_grid, Z0, t_grid, n_samples, Dt, Z, param_train)
Zis_mean, _, _, _, _, _ = simulate_interpolated_sindy_mean(param_grid, Z0, t_grid, Dt, Z, param_train)

_, _, max_e_relative_mean, _ = compute_errors(n_a_grid, n_w_grid, Zis_mean, autoencoder, X_test, Dt, Dx)
_, _, _, max_std = compute_errors(n_a_grid, n_w_grid, Zis_samples, autoencoder, X_test, Dt, Dx)
max_e_relative_mean_AE = compute_errors_AE(n_a_grid, n_w_grid, Zis_mean, autoencoder, X_test, Dt, Dx)

gp_pred = eval_gp(gp_dictionnary, param_grid, n_coef)

fig1, axs1 = plt.subplots(5, 5, figsize = (15, 13))
fig2, axs2 = plt.subplots(5, 5, figsize = (15, 13))
refine = 10
cm = plt.cm.jet

for i in range(5):
    for j in range(5):
        # if j != 5:
        #     coef_nbr = 6 + j * 5 + i
        # else:
        #     coef_nbr = i + 1
        coef_nbr = j + i*5 + 1
        coef_nbr = 'coef_' + str(coef_nbr)

        std = gp_pred[coef_nbr]['std'].reshape(n_a_grid, n_w_grid)
        p = axs1[i, j].contourf(a_grid, w_grid, std, refine, cmap = cm)
        fig1.colorbar(p, ticks = np.array([std.min(), std.max()]), format='%2.2f', ax = axs1[i, j])
        axs1[i, j].scatter(param_train[:, 0], param_train[:, 1], c='k', marker='+')
        axs1[i, j].set_title(r'$\sqrt{\Sigma^*_{' + str(i + 1) + str(j + 1) + '}}$')
        axs1[i, j].axis('equal')
        axs1[i, j].set_xlim([0.68, 0.92])
        axs1[i, j].set_ylim([0.88, 1.12])
        axs1[i, j].invert_yaxis()
        axs1[i, j].get_xaxis().set_visible(False)
        axs1[i, j].get_yaxis().set_visible(False)

        mean = gp_pred[coef_nbr]['mean'].reshape(n_a_grid, n_w_grid)
        p = axs2[i, j].contourf(a_grid, w_grid, mean, refine, cmap = cm)
        fig2.colorbar(p, ticks = np.array([mean.min(), mean.max()]), format='%2.2f', ax = axs2[i, j])
        axs2[i, j].scatter(param_train[:, 0], param_train[:, 1], c='k', marker='+')
        axs2[i, j].set_title(r'$\mu^*_{' + str(i + 1) + str(j + 1) + '}$')
        axs2[i, j].axis('equal')
        axs2[i, j].set_xlim([0.68, 0.92])
        axs2[i, j].set_ylim([0.88, 1.12])
        axs2[i, j].invert_yaxis()
        axs2[i, j].get_xaxis().set_visible(False)
        axs2[i, j].get_yaxis().set_visible(False)

        if j == 0:
            axs1[i, j].set_ylabel('w')
            axs1[i, j].get_yaxis().set_visible(True)
            axs2[i, j].set_ylabel('w')
            axs2[i, j].get_yaxis().set_visible(True)
        if i == 4:
            axs1[i, j].set_xlabel('a')
            axs1[i, j].get_xaxis().set_visible(True)
            axs2[i, j].set_xlabel('a')
            axs2[i, j].get_xaxis().set_visible(True)


fig, ax = plt.subplots(1, 1, figsize = (10, 10))

cmap = LinearSegmentedColormap.from_list('rg', ['C0', 'w', 'C3'], N = 256)

im = ax.imshow(max_e_relative_mean * 100, cmap = cmap)
fig.colorbar(im, ax = ax, fraction = 0.04)

ax.set_xticks(np.arange(0, n_a_grid, 2), labels=np.round(a_grid[0, ::2], 2))

# ax.set_xticks(np.arange(0, n_a_grid, 2) )
# ax.set_xticklabels(np.round(a_grid[0, ::2], 2))

ax.set_yticks(np.arange(0, n_w_grid, 2), labels=np.round(w_grid[::2, 0], 2))

# ax.set_yticks(np.arange(0, n_w_grid, 2) )
# ax.set_yticklabels(np.round(w_grid[0, ::2], 2))

for i in range(n_a_grid):
    for j in range(n_w_grid):
            ax.text(j, i, round(max_e_relative_mean[i, j] * 100, 1), ha='center', va='center', color='k')

grid_square_x = np.arange(-0.5, n_a_grid, 1)
grid_square_y = np.arange(-0.5, n_w_grid, 1)

n_train = param_train.shape[0]
for i in range(n_train):
    a_index = np.sum((a_grid[0, :] < param_train[i, 0]) * 1)
    w_index = np.sum((w_grid[:, 0] < param_train[i, 1]) * 1)

    if i < n_init:
        color = 'r'
    else:
        color = 'k'

    ax.plot([grid_square_x[a_index], grid_square_x[a_index]], [grid_square_y[w_index], grid_square_y[w_index] + 1],
            c=color, linewidth=2)
    ax.plot([grid_square_x[a_index] + 1, grid_square_x[a_index] + 1],
            [grid_square_y[w_index], grid_square_y[w_index] + 1], c=color, linewidth=2)
    ax.plot([grid_square_x[a_index], grid_square_x[a_index] + 1], [grid_square_y[w_index], grid_square_y[w_index]],
            c=color, linewidth=2)
    ax.plot([grid_square_x[a_index], grid_square_x[a_index] + 1],
            [grid_square_y[w_index] + 1, grid_square_y[w_index] + 1], c=color, linewidth=2)

ax.set_xlabel('$a$', fontsize=20)
ax.set_ylabel('$w$', fontsize=20)
ax.set_title('GPLaSDI', fontsize=25)


fig, ax = plt.subplots(figsize = (10, 10))

cmap = LinearSegmentedColormap.from_list('rg', ['C0', 'w', 'C3'], N = 256)
#cmap = plt.cm.magma

im = ax.imshow(max_std / 1, cmap = cmap)
fig.colorbar(im, ax = ax, fraction = 0.04)

ax.set_xticks(np.arange(0, n_a_grid, 2), labels=np.round(a_grid[0, ::2], 2))

# ax.set_xticks(np.arange(0, n_a_grid, 2) )
# ax.set_xticklabels(np.round(a_grid[0, ::2], 2))

ax.set_yticks(np.arange(0, n_w_grid, 2), labels=np.round(w_grid[::2, 0], 2))

# ax.set_yticks(np.arange(0, n_w_grid, 2) )
# ax.set_yticklabels(np.round(w_grid[0, ::2], 2))

for i in range(n_a_grid):
    for j in range(n_w_grid):
        # ax.text(j, i, round(max_std[i, j], 1), ha='center', va='center', color='k')
        str_std = list('{:05f}'.format(max_std[i, j]))
        ax.text(j, i, str_std[2] + '.' + str_std[3], ha='center', va='center', color='k')



grid_square_x = np.arange(-0.5, n_a_grid, 1)
grid_square_y = np.arange(-0.5, n_w_grid, 1)

n_train = param_train.shape[0]
for i in range(n_train):
    a_index = np.sum((a_grid[0, :] < param_train[i, 0]) * 1)
    w_index = np.sum((w_grid[:, 0] < param_train[i, 1]) * 1)

    if i < n_init:
        color = 'r'
    else:
        color = 'k'

    ax.plot([grid_square_x[a_index], grid_square_x[a_index]], [grid_square_y[w_index], grid_square_y[w_index] + 1],
            c=color, linewidth=2)
    ax.plot([grid_square_x[a_index] + 1, grid_square_x[a_index] + 1],
            [grid_square_y[w_index], grid_square_y[w_index] + 1], c=color, linewidth=2)
    ax.plot([grid_square_x[a_index], grid_square_x[a_index] + 1], [grid_square_y[w_index], grid_square_y[w_index]],
            c=color, linewidth=2)
    ax.plot([grid_square_x[a_index], grid_square_x[a_index] + 1],
            [grid_square_y[w_index] + 1, grid_square_y[w_index] + 1], c=color, linewidth=2)

ax.set_xlabel('$a$', fontsize=20)
ax.set_ylabel('$w$', fontsize=20)
ax.set_title(r'max$_{(t,x)}\sqrt{V[\tilde{u}_{\xi^*}]}$   ($10^{-1}$)', fontsize=25)




# a, w = 0.9, 1.07
# maxk = 10
# convergence_threshold = 1e-8
# param = np.array([[a, w]])
# u0 = initial_condition(a, w, x_grid)
# true = solver(u0, maxk, convergence_threshold, time_dim - 1, space_dim, Dt, Dx)
# true = true.T
# scale = 1

# z0 = autoencoder.encoder(torch.Tensor(u0.reshape(1, 1, -1)))
# z0 = z0[0, 0, :].detach().numpy()

# # knn = 4
# # interpolation_data = build_interpolation_data(sindy_coef, param_train)
# # interpolation_data = build_interpolation_data_knn(sindy_coef, param_train,param,knn)
# # gp_dictionnary = fit_gps_knn(interpolation_data)
# # n_coef = interpolation_data['n_coef']
# Z = simulate_uncertain_sindy(gp_dictionnary, param, n_samples, z0, t_grid, sindy_coef, n_coef)
# Z_mean = Z.mean(0)
# Z_std = Z.std(0)


# pred = autoencoder.decoder(torch.Tensor(Z)).detach().numpy()
# pred_mean = pred.mean(0)
# pred_std = pred.std(0)

# test = autoencoder.encoder(torch.Tensor(true.T))
# test = test.detach().numpy()
# fig = plt.figure()
# ax = plt.axes()
# ax.plot(test)
# ax.plot(Z_mean,'--')

# plot_prediction(param, autoencoder, gp_dictionnary, n_samples, z0, t_grid, sindy_coef, n_coef, t_mesh, x_mesh, scale, true, Dt, Dx)


param_ind = 9
a, w = param_grid[param_ind,0], param_grid[param_ind,1]

maxk = 10
convergence_threshold = 1e-8
param = np.array([[a, w]])
u0 = initial_condition(a, w, x_grid)


true = X_test[param_ind,:,:]
true = true.T
scale = 1

z0 = autoencoder.encoder(torch.Tensor(u0.reshape(1, 1, -1)))
z0 = z0[0, 0, :].detach().numpy()

# knn = 4
# interpolation_data = build_interpolation_data(Amats, param_train)
# interpolation_data = build_interpolation_data_knn(Amats, param_train,param,knn)
# gp_dictionnary = fit_gps_knn(interpolation_data)
# n_coef = interpolation_data['n_coef']
Z = Zis_mean[param_ind]
Z_mean = Z.mean(0)
Z_std = Z.std(0)

#this is the rel error
# np.linalg.norm(autoencoder(torch.Tensor(true.T)).detach().numpy().T - true,axis=0)/np.linalg.norm(true,axis=0)

pred = autoencoder.decoder(torch.Tensor(Z)).detach().numpy()
pred_mean = pred.mean(0)
pred_std = pred.std(0)

test = autoencoder.encoder(torch.Tensor(true.T))
test = test.detach().numpy()
fig = plt.figure()
ax = plt.axes()
ax.plot(test)
ax.plot(Z_mean,'--')

plot_prediction(param, autoencoder, gp_dictionnary, n_samples, z0, t_grid, sindy_coef, n_coef, t_mesh, x_mesh, scale, true, Dt, Dx)

# %%
n_coef = n_z*(n_z)
gp_pred = eval_gp(gp_dictionnary, param_grid, n_coef)
std = np.zeros(len(param_grid))


coef_x, coef_y = sindy_coef[0].shape

k = 1

for i in range(coef_x):
    for j in range(coef_y):
        std = std + gp_pred['coef_' + str(k)]['std']

        k += 1
        
fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
fig.set_size_inches(16,10)
axes.set_title('Std')
z1 = axes.scatter(param_grid[:, 0], param_grid[:, 1], c=std,s = 1000,marker='s')
fig.colorbar(z1, ax = axes,ticks = np.array([std.min(), std.max()]))
# fig.colorbar(z1, ax = axes,ticks = np.array([std.min(), std.max()]), format = '%2.1f')


DEBUG_FLAG = 1


