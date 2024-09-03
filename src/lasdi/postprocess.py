import numpy as np
import torch

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