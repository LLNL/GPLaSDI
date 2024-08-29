import numpy as np
import torch

def interpolate_coef_matrix_mean(gp_dictionnary, param, n_coef, sindy_coef):
    from .interp import eval_gp_obsolete

    coef_samples = []
    coef_x, coef_y = sindy_coef[0].shape
    if param.ndim == 1:
        param = param.reshape(1, -1)

    gp_pred = eval_gp_obsolete(gp_dictionnary, param, n_coef)


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
    from .sindy import simulate_sindy

    if coef_samples is None:
        coef_samples = interpolate_coef_matrix_mean(gp_dictionnary, param, n_coef, sindy_coef)

    Z0 = [z0]
    Z = simulate_sindy(coef_samples, Z0, t_grid)

    return Z

def simulate_interpolated_sindy_mean(param_grid, Z0, t_grid, Dt, Z, param_train, fd_type):
    from .sindy import compute_time_derivative, solve_sindy
    from .interp import build_interpolation_data, fit_gps_obsolete

    dZdt = compute_time_derivative(Z, Dt, fd_type)
    sindy_coef = solve_sindy(dZdt, Z)
    interpolation_data = build_interpolation_data(sindy_coef, param_train)
    gp_dictionnary = fit_gps_obsolete(interpolation_data)
    n_coef = interpolation_data['n_coef']

    coef_samples = [interpolate_coef_matrix_mean(gp_dictionnary, param_grid[i, :], n_coef, sindy_coef) for i in range(param_grid.shape[0])]

    Z_simulated = [simulate_uncertain_sindy_mean(gp_dictionnary, param_grid[i, 0], Z0[i], t_grid, sindy_coef, n_coef, coef_samples[i]) for i in range(param_grid.shape[0])]

    return Z_simulated, gp_dictionnary, interpolation_data, sindy_coef, n_coef, coef_samples

def compute_errors(n_a_grid, n_b_grid, Zis, autoencoder, X_test, Dt, Dx):

    '''

    Compute the maximum relative errors accross the parameter space grid

    '''

    max_e_residual = np.zeros([n_a_grid, n_b_grid])
    max_e_relative = np.zeros([n_a_grid, n_b_grid])
    max_e_relative_mean = np.zeros([n_a_grid, n_b_grid])
    max_std = np.zeros([n_a_grid, n_b_grid])

    m = 0

    for j in range(n_b_grid):
        for i in range(n_a_grid):

            Z_m = torch.Tensor(Zis[m])
            X_pred_m = autoencoder.decoder(Z_m).detach().numpy()
            e_relative_m = np.linalg.norm((X_test[m:m + 1, :, :] - X_pred_m), axis = 2) / np.linalg.norm(X_test[m:m + 1, :, :], axis = 2)
            e_relative_m_mean = np.linalg.norm((X_test[m, :, :] - X_pred_m.mean(0)), axis = 1) / np.linalg.norm(X_test[m, :, :], axis = 1)
            max_e_relative_m = e_relative_m.max()
            max_e_relative_m_mean = e_relative_m_mean.max()
            max_std_m = X_pred_m.std(0).max()

            X_pred_m = X_pred_m.mean(0)
            # TODO(kevin): detach physics here.
            # TODO(kevin): we're still using deprecated function here?
            _, e_residual_m = residual(X_pred_m.T, Dt, Dx)
            max_e_residual_m = e_residual_m.max()

            max_e_relative[j, i] = max_e_relative_m
            max_e_relative_mean[j, i] = max_e_relative_m_mean
            max_e_residual[j, i] = max_e_residual_m
            max_std[j, i] = max_std_m

            m += 1

    return max_e_residual, max_e_relative, max_e_relative_mean, max_std

def residual(U, Dt, Dx, n_ts = None):

    '''

    DEPRECATED

    '''

    if n_ts is None:
        dUdt = (U[:, 1:] - U[:, :-1]) / Dt
        dUdx = (U[1:, :] - U[:-1, :]) / Dx

        r = dUdt[:-1, :] - U[:-1, :-1] * dUdx[:, :-1]
        e = np.linalg.norm(r)

        return r, e

    else:
        nt = U.shape[1]
        time_steps = np.arange(0, nt - 1, 1)
        np.random.shuffle(time_steps)
        time_steps = time_steps[:n_ts]
        dUdt = (U[:, time_steps + 1] - U[:, time_steps]) / Dt
        dUdx2 = (U[2:, time_steps] - 2 * U[1:-1, time_steps] + U[:-2, time_steps]) / Dx / Dx
        r = dUdt[1:-1, :] - k * dUdx2
        e = np.linalg.norm(r).mean()

        return r, e 
    

def plot_prediction(param, autoencoder, gp_dictionnary, n_samples, z0, t_grid, sindy_coef, n_coef, t_mesh, x_mesh, scale, true, Dt, Dx):

    '''

    DEPRECATED

    '''
    from .sindy import simulate_uncertain_sindy
    import matplotlib.pyplot as plt

    Z = simulate_uncertain_sindy(gp_dictionnary, param, n_samples, z0, t_grid, sindy_coef, n_coef)

    n_z = Z.shape[2]

    pred = autoencoder.decoder(torch.Tensor(Z)).detach().numpy()
    pred_mean = pred.mean(0)
    pred_std = pred.std(0)

    r, e = residual(pred_mean.T, Dt, Dx)

    plt.figure()

    plt.subplot(231)
    for s in range(n_samples):
        for i in range(n_z):
            plt.plot(t_grid, Z[s, :, i], 'C' + str(i), alpha = 0.3)
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
    plt.contourf(t_mesh[:-1, :-1], x_mesh[:-1, :-1], r, 100, cmap = plt.cm.jet)
    plt.colorbar()
    plt.title('Residual')

    plt.tight_layout()