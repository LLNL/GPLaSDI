import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.utils_sindy import *

def plot_prediction(T, k, time_step, scale, initial_condition_func, autoencoder, gp_dictionnary, n_samples, t_grid, sindy_coef, n_coef, nx, nv, x_mesh, v_mesh, X_test, param_grid, Z = None):
           
    param = np.array([[T, k]])
        
    if Z is None:

        u0 = initial_condition_func(T, k)

        z0 = autoencoder.encoder(torch.Tensor(u0.reshape(1, 1, -1)))
        z0 = z0[0, 0, :].detach().numpy()

        Z = simulate_uncertain_sindy(gp_dictionnary, param, n_samples, z0, t_grid, sindy_coef, n_coef)

    n_z = Z.shape[2]

    pred = autoencoder.decoder(torch.Tensor(Z)).detach().numpy()
    pred_mean = pred.mean(0)
    pred_std = pred.std(0)

    mean = pred_mean[time_step, :]
    std = pred_std[time_step, :]

    mean = mean.reshape(nx, nv)
    std = std.reshape(nx, nv)

    fig = plt.figure(figsize = (15, 8))
    plt.subplot(231)
    for s in range(n_samples):
        for i in range(n_z):
            plt.plot(t_grid, Z[s, :, i], 'C' + str(i), alpha=0.3)
            plt.scatter(t_grid[time_step], Z[s, time_step, i], color = 'k', marker = '+')
    plt.title('Latent Space')

    plt.subplot(232)
    plt.contourf(x_mesh, v_mesh, mean[::scale, ::scale], 100, cmap=plt.cm.jet)
    plt.colorbar()
    plt.title('Mean Prediction')

    plt.subplot(233)
    plt.contourf(x_mesh, v_mesh, std[::scale, ::scale], 100, cmap=plt.cm.jet)
    plt.colorbar()
    plt.clim(0, 0.3)
    plt.title('Standard Deviation')

    m = np.where((1 * (np.round(param_grid, 4) == np.round(param[0, :], 4))).sum(1) == 2)[0][0]

    plt.subplot(235)
    ground_true = X_test[m, time_step, :]
    ground_true = ground_true.reshape(nx, nv)
    plt.contourf(x_mesh, v_mesh, ground_true[::scale, ::scale], 100, cmap=plt.cm.jet)
    plt.colorbar()
    plt.title('True')

    plt.subplot(236)
    error = np.abs(ground_true - mean)
    plt.contourf(x_mesh, v_mesh, error[::scale, ::scale], 100, cmap=plt.cm.jet)
    plt.colorbar()
    plt.clim(0, 0.3)
    plt.title('Absolute Error')

    return fig, pred, X_test[m, :, :], Z
