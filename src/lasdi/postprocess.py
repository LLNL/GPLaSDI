import numpy as np
import torch
import matplotlib.pyplot as plt

def compute_errors(X_pred, physics, X_test):

    '''

    Compute the maximum relative errors on a parameter

    '''

    assert(X_pred.shape == X_test.shape)
    residual = physics.residual(X_pred)
    
    X_pred = X_pred.reshape(X_pred.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    rel_error = np.linalg.norm(X_pred - X_test, axis=1) / np.linalg.norm(X_test, axis=1)

    return rel_error.max(), residual

def plot_prediction(param, autoencoder, physics, sindy, gp_dictionnary, n_samples, true, scale=1):

    from .gplasdi import sample_roms
    import matplotlib.pyplot as plt

    Z = sample_roms(autoencoder, physics, sindy, gp_dictionnary, param, n_samples)
    Z = Z[0]

    n_z = autoencoder.n_z

    pred = autoencoder.decoder(torch.Tensor(Z)).detach().numpy()
    pred_mean = pred.mean(0)
    pred_std = pred.std(0)

    r, e = physics.residual(pred_mean)

    t_mesh, x_mesh = physics.t_grid, physics.x_grid
    if (physics.x_grid.ndim > 1):
        raise RuntimeError('plot_prediction supports only 1D physics!')

    plt.figure()

    plt.subplot(231)
    for s in range(n_samples):
        for i in range(n_z):
            plt.plot(t_mesh, Z[s, :, i], 'C' + str(i), alpha = 0.3)
    plt.title('Latent Space')

    plt.subplot(232)
    plt.contourf(t_mesh, x_mesh, pred_mean[::scale, ::scale].T, 100, cmap = plt.cm.jet)
    plt.colorbar()
    plt.title('Decoder Mean Prediction')

    plt.subplot(233)
    plt.contourf(t_mesh, x_mesh, pred_std[::scale, ::scale].T, 100, cmap = plt.cm.jet)
    plt.colorbar()
    plt.title('Decoder Standard Deviation')

    plt.subplot(234)
    plt.contourf(t_mesh, x_mesh, true[::scale, ::scale], 100, cmap = plt.cm.jet)
    plt.colorbar()
    plt.title('Ground True')

    plt.subplot(235)
    error = np.abs(true - pred_mean.T)
    plt.contourf(t_mesh, x_mesh, error, 100, cmap = plt.cm.jet)
    plt.colorbar()
    plt.title('Absolute Error')

    plt.subplot(236)
    plt.contourf(t_mesh[:-1], x_mesh[:-1], r, 100, cmap = plt.cm.jet)
    plt.colorbar()
    plt.title('Residual')

    plt.tight_layout()

def plot_gp2d(p1_mesh, p2_mesh, gp_mean, gp_std, param_train, param_labels=['p1', 'p2'], plot_shape=[6, 5], figsize=(15, 13), refine=10, cm=plt.cm.jet, margin=0.05):
    assert(p1_mesh.ndim == 2)
    assert(p2_mesh.ndim == 2)
    assert(gp_mean.ndim == 3)
    assert(gp_std.ndim == 3)
    assert(param_train.ndim == 2)
    assert(gp_mean.shape == gp_std.shape)

    plot_shape_ = [gp_mean.shape[-1] // plot_shape[-1], plot_shape[-1]]
    if (gp_mean.shape[-1] % plot_shape[-1] > 0):
        plot_shape_[0] += 1

    p1_range = [p1_mesh.min() * (1. - margin), p1_mesh.max() * (1. + margin)]
    p2_range = [p2_mesh.min() * (1. - margin), p2_mesh.max() * (1. + margin)]

    fig1, axs1 = plt.subplots(plot_shape_[0], plot_shape_[1], figsize = figsize)
    fig2, axs2 = plt.subplots(plot_shape_[0], plot_shape_[1], figsize = figsize)

    for i in range(plot_shape_[0]):
        for j in range(plot_shape_[1]):
            k = j + i * plot_shape_[1]

            if (k >= gp_mean.shape[-1]):
                axs1[i, j].set_xlim(p1_range)
                axs1[i, j].set_ylim(p2_range)
                axs2[i, j].set_xlim(p1_range)
                axs2[i, j].set_ylim(p2_range)
                if (j == 0):
                    axs1[i, j].set_ylabel(param_labels[1])
                    axs1[i, j].get_yaxis().set_visible(True)
                    axs2[i, j].set_ylabel(param_labels[1])
                    axs2[i, j].get_yaxis().set_visible(True)
                if (i == plot_shape_[0] - 1):
                    axs1[i, j].set_xlabel(param_labels[0])
                    axs1[i, j].get_xaxis().set_visible(True)
                    axs2[i, j].set_xlabel(param_labels[0])
                    axs2[i, j].get_xaxis().set_visible(True)

                continue

            std = gp_std[:, :, k]
            p = axs1[i, j].contourf(p1_mesh, p2_mesh, std, refine, cmap = cm)
            fig1.colorbar(p, ticks = np.array([std.min(), std.max()]), format='%2.2f', ax = axs1[i, j])
            axs1[i, j].scatter(param_train[:, 0], param_train[:, 1], c='k', marker='+')
            axs1[i, j].set_title(r'$\sqrt{\Sigma^*_{' + str(i + 1) + str(j + 1) + '}}$')
            axs1[i, j].set_xlim(p1_range)
            axs1[i, j].set_ylim(p2_range)
            axs1[i, j].invert_yaxis()
            axs1[i, j].get_xaxis().set_visible(False)
            axs1[i, j].get_yaxis().set_visible(False)

            mean = gp_mean[:, :, k]
            p = axs2[i, j].contourf(p1_mesh, p2_mesh, mean, refine, cmap = cm)
            fig2.colorbar(p, ticks = np.array([mean.min(), mean.max()]), format='%2.2f', ax = axs2[i, j])
            axs2[i, j].scatter(param_train[:, 0], param_train[:, 1], c='k', marker='+')
            axs2[i, j].set_title(r'$\mu^*_{' + str(i + 1) + str(j + 1) + '}$')
            axs2[i, j].set_xlim(p1_range)
            axs2[i, j].set_ylim(p2_range)
            axs2[i, j].invert_yaxis()
            axs2[i, j].get_xaxis().set_visible(False)
            axs2[i, j].get_yaxis().set_visible(False)

            if (j == 0):
                axs1[i, j].set_ylabel(param_labels[1])
                axs1[i, j].get_yaxis().set_visible(True)
                axs2[i, j].set_ylabel(param_labels[1])
                axs2[i, j].get_yaxis().set_visible(True)
            if (i == plot_shape_[0] - 1):
                axs1[i, j].set_xlabel(param_labels[0])
                axs1[i, j].get_xaxis().set_visible(True)
                axs2[i, j].set_xlabel(param_labels[0])
                axs2[i, j].get_xaxis().set_visible(True)

    return

def heatmap2d(values, p1_grid, p2_grid, param_train, n_init, figsize=(10, 10), param_labels=['p1', 'p2'], title=''):
    assert(p1_grid.ndim == 1)
    assert(p2_grid.ndim == 1)
    assert(values.ndim == 2)
    assert(param_train.ndim == 2)

    n_p1 = len(p1_grid)
    n_p2 = len(p2_grid)
    assert(values.shape[0] == n_p1)
    assert(values.shape[1] == n_p2)

    fig, ax = plt.subplots(1, 1, figsize = figsize)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('rg', ['C0', 'w', 'C3'], N = 256)

    im = ax.imshow(values, cmap = cmap)
    fig.colorbar(im, ax = ax, fraction = 0.04)

    ax.set_xticks(np.arange(0, n_p1, 2), labels=np.round(p1_grid[::2], 2))
    ax.set_yticks(np.arange(0, n_p2, 2), labels=np.round(p2_grid[::2], 2))

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

    ax.set_xlabel(param_labels[0], fontsize=15)
    ax.set_ylabel(param_labels[1], fontsize=15)
    ax.set_title(title, fontsize=25)
    return