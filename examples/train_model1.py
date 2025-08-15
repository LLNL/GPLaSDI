#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

sys.path.append('/p/gpfs1/khurana1/test/GPLaSDI')
import numpy as np
import torch
import time
from src.lasdi.physics import Physics
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import pickle
from src.lasdi.FNO.fno import FNO
from src.lasdi.gp import eval_gp

torch.manual_seed(0)
np.random.rand(0)

#For movies

# import matplotlib.animation as manimation
# FFMpegWriter = manimation.writers['ffmpeg']
# metadata = dict(title='Movie Test', artist='Matplotlib',
#                 comment='Blast')
# writer = FFMpegWriter(fps=15)

# name = 'steelruns'

date = time.localtime()
date_str = "{month:02d}_{day:02d}_{year:04d}_{hour:02d}_{minute:02d}"
date_str = date_str.format(month = date.tm_mon, day = date.tm_mday, year = date.tm_year, hour = date.tm_hour + 3, minute = date.tm_min)

name = date_str 

#Initialize training params
dt = 0.001
ae_weight = 1e2
sindy_weight = 1.e-1
coef_weight= 1.e-6
lr = 1e-1

path_data = 'data/'


path_data = 'data/'
data_train = np.load(path_data + 'data_train.npy', allow_pickle = True).item()
data_test = np.load(path_data + 'data_test.npy', allow_pickle = True).item()

# These can be any data, as long as they are in the form [num_simulations, num_timesteps, num_spatial_nodes]
X_train = torch.Tensor(data_train['X_train'])
X_test = (data_test['X_test'])
param_train = data_train['param_train']
param_test = data_test['param_grid']

sol_dim = 1
grid_dim = [data_test['param_grid'].shape[0]]
time_dim = X_train.shape[1]
space_dim = X_train.shape[-1]


t_grid = np.linspace(0, (time_dim - 1)*dt, time_dim)

#This physics class will also be able to train with any data.
#However, we will not be able to adaptively sample/generate new data
class CustomPhysicsModel(Physics):
    def __init__(self):
        self.dim = 1
        self.nt = time_dim
        self.dt = dt
        self.qdim = sol_dim
        self.qgrid_size = [space_dim]
        self.t_grid = t_grid
        return
    
    ''' See lasdi.physics.Physics class for necessary subroutines '''
    
physics = CustomPhysicsModel()

# Autoencoder Definition
hidden_units = [100]

#latent dim (number of modes to keep)
n_z = 16
n_layers = 3 
n_iter = 1000 

# Tranposing the train data to match the shape for 1d fno!
model = FNO(n_modes=(1001,n_z),
             in_channels=1, 
             out_channels=1,
             hidden_channels=1, 
             projection_channel_ratio=2, use_channel_mlp=False,
             positional_embedding=None, physics = physics, n_layers = n_layers, test_flag=False)

best_loss = np.inf

optimizer = torch.optim.Adam(model.parameters(), lr = lr)
MSE = torch.nn.MSELoss()

cuda = torch.cuda.is_available()
if cuda:
    device = 'cuda'
else:
    device = 'cpu'
    
# device = 'mps'
    
print(device)
        
model = model.to(device)

# Adding extra dimension to the X_train
X_train = X_train.unsqueeze(1)
d_Xtrain = X_train.to(device)

n_train = X_train.shape[0]
save_interval = 1
hist_file = '%s.loss_history.txt' % name

loss_hist = np.zeros([n_iter, 4])
grad_hist = np.zeros([n_iter, 4])
 
tic_start = time.time()
max_coef = np.zeros((n_iter, 2))
total_loss = []
loss_ae_list = []
loss_sindy_list = []
loss_sindy_coefs_list = []

for iter in range(n_iter):
    optimizer.zero_grad()
    # model = ae.to(device)
    d_Xpred = model(d_Xtrain)

    loss_ae = MSE(d_Xtrain, d_Xpred) #Reconstruction loss

    max_coef[iter][0] = torch.max(torch.abs(model.coefs[0]))
    
    loss = ae_weight * loss_ae + sindy_weight * sum(model.sindy_loss) / n_train + coef_weight * sum(model.loss_coefs) / n_train

    # Append to the lists for plotting purposes!
    # Append to list for plotting!
    total_loss.append(loss.detach().cpu().numpy())
    loss_sindy_list.append(sindy_weight * sum(model.sindy_loss).detach().cpu().numpy() / n_train)
    loss_ae_list.append(ae_weight * loss_ae.detach().cpu().numpy())
    loss_sindy_coefs_list.append(coef_weight * sum(model.loss_coefs).detach().cpu().numpy() / n_train)

    loss_hist[iter] = [loss.item(), loss_ae.item(), sum(model.sindy_loss).item(), sum(model.loss_coefs).item()]   

    loss.backward()

    optimizer.step()

    if ((loss.item() < best_loss) and (iter % save_interval == 0)):
        os.makedirs(os.path.dirname('checkpoint/' + './%s_checkpoint.pt' % name), exist_ok=True)
        torch.save(model.cpu().state_dict(), 'checkpoint/%s_checkpoint.pt' % name)
        model = model.to(device)
        best_loss = loss.item()
        best_coefs = model.coefs

    print("Iter: %05d/%d, Loss: %.5e, Loss AE: %.5e, Loss SI: %.5e, Loss COEF: %.5e"
            % (iter + 1, n_iter, loss.item(), loss_ae.item(), sum(model.sindy_loss).item(), sum(model.loss_coefs).item()))

tic_end = time.time()
total_time = tic_end - tic_start

## save results

os.makedirs(os.path.dirname('losses/' + './%s_checkpoint.pt' % name), exist_ok=True)
np.savetxt('losses/%s.loss_history.txt' % name, loss_hist)

if (loss.item() < best_loss):
    torch.save(model.cpu().state_dict(), 'checkpoint/%s_checkpoint.pt' % name)
    best_loss = loss.item()
else:
    model.cpu().load_state_dict(torch.load('checkpoint/%s_checkpoint.pt' % name, weights_only=False))

bglasdi_results = {'autoencoder_param': model.cpu().state_dict(), 'final_param_train': param_train,
                            'lr': lr, 'n_iter': n_iter,
                            'sindy_weight': sindy_weight, 'coef_weight': coef_weight,
                            't_grid' : t_grid, 'dt' : dt,'total_time' : total_time} 
    
os.makedirs(os.path.dirname('results/'), exist_ok=True)
np.save('results/bglasdi_' + date_str + '_lr' + str(lr) + '_sw' + str(sindy_weight) + '_cw'+str(coef_weight)+'_nt' + str(time_dim) + '_niter' + str(n_iter) + '_nz' + str(n_z) +'.npy', bglasdi_results)


## Plotting the loss and stuff!

plt.figure()
plt.plot(max_coef[:,0])#, label = 'Real weight')
# plt.plot(max_coef[:,1], label = 'Imaginary weight')
plt.title('Max SINDY coefficient')
plt.ylabel('Magnitude')
plt.xlabel('iterations')
plt.grid()
plt.savefig('plots/max_weight.png')

plt.figure(figsize=(12, 8))
plt.suptitle('Training Loss v/s Iterations (LOG SCALE)')

plt.subplot(2, 2, 1)
plt.plot(total_loss)
plt.xlabel('Iterations')
plt.ylabel('Total Loss')
plt.grid()
plt.yscale('log')

plt.subplot(2, 2, 2)
plt.plot(loss_ae_list)
plt.xlabel('Iterations')
plt.ylabel('Loss AE')
plt.grid()
plt.yscale('log')

plt.subplot(2, 2, 3)
plt.plot(loss_sindy_list)
plt.xlabel('Iterations')
plt.ylabel('Sindy loss')
plt.grid()
plt.yscale('log')

plt.subplot(2, 2, 4)
plt.plot(loss_sindy_coefs_list)
plt.xlabel('Iterations')
plt.ylabel('Sindy coefficient loss')
plt.grid()
plt.yscale('log')

plt.savefig('plots/loss.png')
# %% Plotting

# autoencoder_param = model.cpu().state_dict()
# AEparams = [value for value in autoencoder_param.items()]
# AEparamnum = 0
# for i in range(len(AEparams)):
#     AEparamnum = AEparamnum + (AEparams[i][1].detach().numpy()).size

from lasdi.gp import fit_gps
from lasdi.gplasdi import sample_roms, average_rom
import numpy.linalg as LA

def IC(param):
    return param[0] * np.exp(- np.linspace(-3, 3, 1001) ** 2 / 2 / param[1] / param[1])

#Need to know IC explicitly. If you only have IC which corresponds to certain param value,
#then you can do something like

# def IC(param):
#     for i in range(X_test.shape[0]):
#         if np.abs(param[0] - all_param[i]) <1E-8:
#             initcond = X_test[i,0,:]
#             break
#     return initcond

physics.initial_condition = IC
model = model.cpu()

# Get the model predictions!
"""
Deviating from the gausian process here! GP Stuff needs more thought!
"""

# Get the initial conditions for the entire param grid!
n_param = param_test.shape[0]
z0 = np.zeros((n_param,1,1,1001), dtype=np.float32)

sol_shape = [1, 1] + physics.qgrid_size

for i in range(n_param):
    u0 = physics.initial_condition(param_test[i])
    u0 = u0.reshape(sol_shape)
    z0[i] = u0

# Interpolate the sindy coefficients!
gp_dictionnary_real = fit_gps(param_train, best_coefs[0])
# gp_dictionnary_imag = fit_gps(param_train, best_coefs[1])
pred_mean_real, _ = eval_gp(gp_dictionnary_real, param_test)
# pred_mean_imag,_ = eval_gp(gp_dictionnary_imag, param_test)

### Plotting the latent space trajectories!
test_param = X_train[0]
test_coefs = [pred_mean_real[0], 0]

# Plotting model!
plot_model = FNO(n_modes=(1001,n_z),
                in_channels=1, 
                out_channels=1,
                hidden_channels=1, 
                projection_channel_ratio=2, use_channel_mlp=False,
                positional_embedding=None, physics = physics, n_layers = n_layers,
                test_flag=False, best_coefs = test_coefs, plot_latent=True)
# Load the weights into the model!
plot_model.cpu().load_state_dict(torch.load('checkpoint/%s_checkpoint.pt' % name, weights_only=False))

# Get the plots!
plot_model(torch.unsqueeze(test_param,0))

#### Getting the entire testing!
# Get the pred_stuff!
X_pred_mean = np.zeros(X_test.shape)

for i in range(param_test.shape[0]):
 
    # Get the interpolated coeff:
    coefs = [pred_mean_real[i],0]# pred_mean_imag[i]]
 
    # Defining the test model!
    test_model = FNO(n_modes=(1001,n_z),
                in_channels=1, 
                out_channels=1,
                hidden_channels=1, 
                projection_channel_ratio=2, use_channel_mlp=False,
                positional_embedding=None, physics = physics, n_layers = n_layers,
                test_flag=True, best_coefs=coefs, plot_latent=False)

    # Load the weights into the model!
    test_model.cpu().load_state_dict(torch.load('checkpoint/%s_checkpoint.pt' % name, weights_only=False))

    # Get the prediction!
    X_pred_mean[i] = test_model(torch.unsqueeze(torch.from_numpy(z0[i]),0)).detach().numpy()


# %% 

#Plot a single value

param_ind = 2
param = np.array([param_test[param_ind]])


true = X_test[param_ind,:,:]

## Heatmap of errors

param_grid = param_test
avg_rel_err = LA.norm(X_pred_mean - X_test,axis=2)/LA.norm(X_test,axis=2)
max_rel_err = np.max(avg_rel_err, axis = 1)

figsize=(12, 12)

n_a_grid = 21 
n_w_grid = 21

a_grid = param_grid[:21,0]
w_grid = param_grid[::21,1]


n_p1 = n_a_grid
n_p2 = n_w_grid
p1_grid = param_grid[:21,0]
p2_grid = param_grid[::21,1]

fig, ax = plt.subplots(1, 1, figsize = figsize)
values = max_rel_err.T.reshape(21,21)*100

n_init = len(param_train)


from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('rg', ['C0', 'w', 'C3'], N = 256)

im = ax.imshow(values, cmap = cmap)
fig.colorbar(im, ax = ax, fraction = 0.04)

ax.set_xticks(np.arange(0, n_a_grid, 2), labels=np.round(a_grid[::2], 2))
ax.set_yticks(np.arange(0, n_w_grid, 2), labels=np.round(w_grid[::2], 2))

for i in range(n_p1):
    for j in range(n_p2):
        ax.text(j, i, round(values[i, j], 1), ha='center', va='center', color='k')

grid_square_x = np.arange(-0.5, n_p1, 1)
grid_square_y = np.arange(-0.5, n_p2, 1)

n_train = param_train.shape[0]
for i in range(n_train):
    p1_index = np.sum((p1_grid < param_train[i, 0]) * 1)
    p2_index = np.sum((p2_grid < param_train[i, 1]) * 1)

    if i < n_init:
        color = 'r'
    else:
        color = 'k'

    ax.plot([grid_square_x[p1_index], grid_square_x[p1_index]], [grid_square_y[p2_index], grid_square_y[p2_index] + 1],
            c=color, linewidth=2)
    ax.plot([grid_square_x[p1_index] + 1, grid_square_x[p1_index] + 1],
            [grid_square_y[p2_index], grid_square_y[p2_index] + 1], c=color, linewidth=2)
    ax.plot([grid_square_x[p1_index], grid_square_x[p1_index] + 1], [grid_square_y[p2_index], grid_square_y[p2_index]],
            c=color, linewidth=2)
    ax.plot([grid_square_x[p1_index], grid_square_x[p1_index] + 1],
            [grid_square_y[p2_index] + 1, grid_square_y[p2_index] + 1], c=color, linewidth=2)

ax.set_xlabel('$a$', fontsize=25)
ax.set_ylabel('$w$', fontsize=25)
ax.set_title('Relative Error (%)', fontsize=30)
plt.savefig('plots/relative_error.png')
plt.show()

# Print the total time!
print(f'Total Training Time: {total_time}')
