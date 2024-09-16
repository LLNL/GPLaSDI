import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.utils_sindy import *

def extrap(tc, rc, initial_condition_func, autoencoder, gp_dictionnary, n_samples, t_grid, sindy_coef, n_coef, normalize = False):

    '''

    DEPRECATED

    '''

    param = np.array([[tc, rc]])

    u0 = initial_condition_func(tc, rc)

    z0 = autoencoder.encoder(torch.Tensor(u0.reshape(1, 1, -1)))
    z0 = z0[0, 0, :].detach().numpy()
    
    print(sindy_coef) 

    coef_samples = interpolate_coef_matrix(gp_dictionnary, param, n_samples, n_coef, sindy_coef, normalize)
    Z = simulate_uncertain_sindy(gp_dictionnary, param, n_samples, z0, t_grid, sindy_coef, n_coef, coef_samples, normalize)

    n_z = Z.shape[2]

    pred = autoencoder.decoder(torch.Tensor(Z)).detach().numpy()
    pred_mean = pred.mean(0)
    pred_std = pred.std(0)

    return pred, Z



def plot_prediction(tc, rc, time_step, scale, initial_condition_func, autoencoder, gp_dictionnary, n_samples, t_grid, sindy_coef, n_coef, nx, ny, x_mesh, y_mesh, X_test, param_grid, m, Z = None, normalize = False):
         
    param = np.array([[tc, rc]])
        
    if Z is None:

        u0 = initial_condition_func(tc, rc)

        z0 = autoencoder.encoder(torch.Tensor(u0.reshape(1, 1, -1)))
        z0 = z0[0, 0, :].detach().numpy()

        Z = simulate_uncertain_sindy(gp_dictionnary, param, n_samples, z0, t_grid, sindy_coef, n_coef, normalize)

    n_z = Z.shape[2]

    pred = autoencoder.decoder(torch.Tensor(Z)).detach().numpy()
    pred_mean = pred.mean(0)
    pred_std = pred.std(0)

    mean = pred_mean[time_step, :]
    std = pred_std[time_step, :]

    mean = mean.reshape(ny, nx)
    std = std.reshape(ny, nx)

    fig = plt.figure(figsize = (15, 8))
    plt.subplot(231)
    for s in range(n_samples):
        for i in range(n_z):
            plt.plot(t_grid * 100, Z[s, :, i], 'C' + str(i), alpha=0.3)
            plt.scatter(t_grid[time_step] * 100, Z[s, time_step, i], color = 'k', marker = '+')
    plt.title('Latent Space')

    plt.subplot(232)
    plt.contourf(x_mesh, y_mesh, mean[::scale, ::scale], 100, cmap=plt.cm.jet)
    plt.colorbar()
    plt.title('Mean Prediction')

    plt.subplot(233)
    plt.contourf(x_mesh, y_mesh, std[::scale, ::scale], 100, cmap=plt.cm.jet)
    plt.colorbar()
    plt.clim(0, 0.08)
    plt.title('Standard Deviation')
    
    plt.subplot(235)
    ground_true = X_test[m, time_step, :]
    ground_true = ground_true.reshape(ny, nx)
    plt.contourf(x_mesh, y_mesh, ground_true[::scale, ::scale], 100, cmap=plt.cm.jet)
    plt.colorbar()
    plt.title('True')

    plt.subplot(236)
    error = np.abs(ground_true - mean)
    plt.contourf(x_mesh, y_mesh, error[::scale, ::scale], 100, cmap=plt.cm.jet)
    plt.colorbar()
    plt.clim(0, 0.08)
    plt.title('Absolute Error')

    return fig, pred, X_test[m, :, :], Z



def plot_gps(gp_dictionnary, param_train, sindy_coef, a_grid, b_grid, n_a_grid, n_b_grid, param_grid, n_coef, normalize = False):

    gp_pred = eval_gp(gp_dictionnary, param_grid, n_coef, normalize)

    plt.figure()
    coef_x, coef_y = sindy_coef[0].shape

    k = 1

    for i in range(coef_x):
        for j in range(coef_y):
            plt.subplot(coef_x, coef_y, k)

            mean = gp_pred['coef_' + str(k)]['mean'].reshape(n_a_grid, n_b_grid)

            plt.contourf(a_grid, b_grid, mean, 100)
            plt.colorbar(ticks = np.array([mean.min(), mean.max()]), format = '%2.1f')
            plt.scatter(param_train[:, 0], param_train[:, 1], c='k', marker='+')
            plt.title('C' + str(k) + ' (Mean)')

            if i != coef_x - 1:
                plt.gca().get_xaxis().set_visible(False)

            if j != 0:
                plt.gca().get_yaxis().set_visible(False)

            if i == coef_x - 1:
                plt.xlabel('a')

            if j == 0:
                plt.ylabel('w')

            k += 1

    plt.figure()

    k = 1

    for i in range(coef_x):
        for j in range(coef_y):
            plt.subplot(coef_x, coef_y, k)

            std = gp_pred['coef_' + str(k)]['std'].reshape(n_a_grid, n_b_grid)

            plt.contourf(a_grid, b_grid, std, 100)
            plt.colorbar(ticks = np.array([std.min(), std.max()]), format = '%2.1f')
            plt.scatter(param_train[:, 0], param_train[:, 1], c='k', marker='+')
            plt.title('C' + str(k) + ' (Std)')

            if i != coef_x - 1:
                plt.gca().get_xaxis().set_visible(False)

            if j != 0:
                plt.gca().get_yaxis().set_visible(False)

            if i == coef_x - 1:
                plt.xlabel('a')

            if j == 0:
                plt.ylabel('w')

            k += 1
