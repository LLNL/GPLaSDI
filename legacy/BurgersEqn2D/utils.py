import numpy as np
from scipy.integrate import odeint
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
import torch
import matplotlib.pyplot as plt
from solver_burgers import solver


def compute_sindy_data(Z, Dt):

    '''

    Builds the SINDy dataset, assuming only linear terms in the SINDy dataset. The time derivatives are computed through
    finite difference.

    Z is the encoder output (3D tensor), with shape [n_train, time_dim, space_dim]

    '''

    dZdt = (Z[:, 1:, :] - Z[:, :-1, :]) / Dt
    Z = Z[:, :-1, :]

    return dZdt, Z


def solve_sindy(dZdt, Z):

    '''

    Computes the SINDy coefficients for each training points.
    sindy_coef is the list of coefficient (length n_train), and each term in sindy_coef is a matrix of SINDy coefficients
    corresponding to each training points.

    '''

    sindy_coef = []
    n_train, time_dim, space_dim = dZdt.shape

    for i in range(n_train):
        dZdt_i = dZdt[i, :, :]
        Z_i = Z[i, :, :]
        Z_i = np.hstack((np.ones([time_dim, 1]), Z_i))

        c_i = np.linalg.lstsq(Z_i, dZdt_i)[0]
        sindy_coef.append(c_i)

    return sindy_coef


def simulate_sindy(sindy_coef, Z0, t_grid):

    '''

    Integrates each system of ODEs corresponding to each training points, given the initial condition Z0 = encoder(U0)

    '''

    n_sindy = len(sindy_coef)

    for i in range(n_sindy):

        c_i = sindy_coef[i].T
        dzdt = lambda z, t : c_i[:, 1:] @ z + c_i[:, 0]

        Z_i = odeint(dzdt, Z0[i], t_grid)
        Z_i = Z_i.reshape(1, Z_i.shape[0], Z_i.shape[1])

        if i == 0:
            Z_simulated = Z_i
        else:
            Z_simulated = np.concatenate((Z_simulated, Z_i), axis = 0)

    return Z_simulated


def build_interpolation_data(sindy_coef, params):

    '''

    Generates a regression training dataset dictionnary for each GP.
    For example, interpolation_data['coef_1'][X] is the tensor of FOM simulation parameters and interpolation_data['coef_1'][y]
    is a vector of the form [sindy_coef[0][0, 0], ... , sindy_coef[n_train][0, 0]]

    '''

    n_sindy = len(sindy_coef)
    coef_x, coef_y = sindy_coef[0].shape
    interpolation_data = {}

    k = 1
    for i in range(coef_x):
        for j in range(coef_y):
            interpolation_data['coef_' + str(k)] = {}
            interpolation_data['coef_' + str(k)]['X'] = params
            for l in range(n_sindy):
                if l == 0:
                    interpolation_data['coef_' + str(k)]['y'] = np.array(sindy_coef[l][i, j])
                else:
                    interpolation_data['coef_' + str(k)]['y'] = np.hstack((interpolation_data['coef_' + str(k)]['y'], np.array(sindy_coef[l][i, j])))
            k += 1

    interpolation_data['n_coef'] = coef_x * coef_y

    return interpolation_data



def fit_gps(interpolation_data):

    '''

    Trains each GP given the interpolation dataset.
    gp_dictionnary is a dataset containing the trained GPs (as sklearn objects)

    '''

    n_coef = interpolation_data['n_coef']
    X = interpolation_data['coef_1']['X']
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    gp_dictionnary = {}

    for i in range(n_coef):

        y = interpolation_data['coef_' + str(i + 1)]['y']

        #kernel = ConstantKernel() * Matern(length_scale_bounds = (0.1, 1e5), nu = 1.5)
        kernel = ConstantKernel() * RBF(length_scale_bounds = (0.1, 1e5))

        gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 10, random_state = 1)
        gp.fit(X, y)

        gp_dictionnary['coef_' + str(i + 1)] = gp

    return gp_dictionnary



def eval_gp(gp_dictionnary, param_grid, n_coef):

    '''

    Computes the GPs predictive mean and standard deviation for points of the parameter space grid

    '''

    gp_pred = {}

    for i in range(n_coef):

        gp = gp_dictionnary['coef_' + str(i + 1)]
        mean, std = gp.predict(param_grid, return_std = True)

        gp_pred['coef_' + str(i + 1)] = {}
        gp_pred['coef_' + str(i + 1)]['mean'] = mean
        gp_pred['coef_' + str(i + 1)]['std'] = std

    return gp_pred



def interpolate_coef_matrix(gp_dictionnary, param, n_samples, n_coef, sindy_coef):

    '''

    Generates sample sets of ODEs for a given parameter.
    coef_samples is a list of length n_samples, where each terms is a matrix of SINDy coefficients sampled from the GP predictive
    distributions

    '''

    coef_samples = []
    coef_x, coef_y = sindy_coef[0].shape
    if param.ndim == 1:
        param = param.reshape(1, -1)

    gp_pred = eval_gp(gp_dictionnary, param, n_coef)

    for _ in range(n_samples):
        coeff_matrix = np.zeros([coef_x, coef_y])
        k = 1
        for i in range(coef_x):
            for j in range(coef_y):
                mean = gp_pred['coef_' + str(k)]['mean']
                std = gp_pred['coef_' + str(k)]['std']

                coeff_matrix[i, j] = np.random.normal(mean, std)
                k += 1

        coef_samples.append(coeff_matrix)

    return coef_samples


def simulate_uncertain_sindy(gp_dictionnary, param, n_samples, z0, t_grid, sindy_coef, n_coef, coef_samples = None):

    '''

    Integrates each ODE samples for a given parameter.

    '''

    if coef_samples is None:
        coef_samples = interpolate_coef_matrix(gp_dictionnary, param, n_samples, n_coef, sindy_coef)

    Z0 = [z0 for _ in range(n_samples)]
    Z = simulate_sindy(coef_samples, Z0, t_grid)

    return Z


def simulate_interpolated_sindy(param_grid, Z0, t_grid, n_samples, Dt, Z, param_train):

    '''

    Integrates each ODE samples for each parameter of the parameter grid.
    Z_simulated is a list of length param_grid.shape[0], where each term is a 3D tensor of the form [n_samples, time_dim, n_z]

    '''

    dZdt, Z = compute_sindy_data(Z, Dt)
    sindy_coef = solve_sindy(dZdt, Z)
    interpolation_data = build_interpolation_data(sindy_coef, param_train)
    gp_dictionnary = fit_gps(interpolation_data)
    n_coef = interpolation_data['n_coef']

    coef_samples = [interpolate_coef_matrix(gp_dictionnary, param_grid[i, :], n_samples, n_coef, sindy_coef) for i in range(param_grid.shape[0])]

    Z_simulated = [simulate_uncertain_sindy(gp_dictionnary, param_grid[i, 0], n_samples, Z0[i], t_grid, sindy_coef, n_coef, coef_samples[i]) for i in range(param_grid.shape[0])]

    return Z_simulated, gp_dictionnary, interpolation_data, sindy_coef, n_coef, coef_samples



def plot_simlated_interpolated_sindy(Zis, index_of_param, param_grid, n_plot_x, n_plot_y, t_grid):

    '''

    Plot integrated SINDy sets of ODE - for debug purpose

    '''

    n_samples = Zis[0].shape[0]
    n_z = Zis[0].shape[2]

    l = 1
    plt.figure()
    for i in range(n_plot_y):
        for j in range(n_plot_x):
            plt.subplot(n_plot_y, n_plot_x, l)
            for o in range(n_z):
                for m in range(n_samples):
                    plt.plot(t_grid, Zis[index_of_param[l - 1]][m, :, o], color = 'C' + str(o), alpha = 0.3)
            plt.title('k = ' + str(np.round(param_grid[index_of_param[l - 1], :], 3)))
            l += 1

    plt.tight_layout()


def display_sindy_equations(index_of_k, k_grid, coef_samples):

    '''

    Display GP-interpolated SINDy equations - for debug purpose

    '''


    n_display = index_of_k.shape[0]

    for i in range(n_display):

        coef_i = np.array(coef_samples[i])
        mean = coef_i.mean(0).T
        std = coef_i.std(0).T

        up = mean + 1.96 * std
        low = mean - 1.96 * std

        n_z, n_coef = mean.shape

        print('\n~~~~~~ k = ' + str(round(k_grid[index_of_k[i]].item(), 4)) + ' ~~~~~~')
        print('\nMEAN COEFFICIENTS\n')
        for z in range(n_z):
            print('dz' + str(z + 1) + ' / dt =', end = '')
            for c in range(1, n_coef):
                if mean[z, c] >= 0:
                    print(' + ' + str(round(mean[z, c], 3)) + ' * z' + str(c), end = '')
                elif mean[z, c] < 0:
                    print(' - ' + str(round(np.abs(mean[z, c]), 3)) + ' * z' + str(c), end = '')

            if mean[z, 0] >= 0:
                print(' + ' + str(round(mean[z, 0], 3)))
            else:
                print(' - ' + str(round(np.abs(mean[z, 0]), 3)))

        print('\nLOWER BOUND\n')
        for z in range(n_z):
            print('dz' + str(z + 1) + ' / dt =', end = '')
            for c in range(1, n_coef):
                if low[z, c] >= 0:
                    print(' + ' + str(round(low[z, c], 3)) + ' * z' + str(c), end = '')
                elif low[z, c] < 0:
                    print(' - ' + str(round(np.abs(low[z, c]), 3)) + ' * z' + str(c), end = '')

            if low[z, 0] >= 0:
                print(' + ' + str(round(low[z, 0], 3)))
            else:
                print(' - ' + str(round(np.abs(low[z, 0]), 3)))

        print('\nUPPER BOUND\n')
        for z in range(n_z):
            print('dz' + str(z + 1) + ' / dt =', end = '')
            for c in range(1, n_coef):
                if up[z, c] >= 0:
                    print(' + ' + str(round(up[z, c], 3)) + ' * z' + str(c), end = '')
                elif up[z, c] < 0:
                    print(' - ' + str(round(np.abs(up[z, c]), 3)) + ' * z' + str(c), end = '')

            if up[z, 0] >= 0:
                print(' + ' + str(round(up[z, 0], 3)))
            else:
                print(' - ' + str(round(np.abs(up[z, 0]), 3)))



def plot_prediction(param, autoencoder, gp_dictionnary, n_samples, z0, t_grid, sindy_coef, n_coef, x_mesh, y_mesh, scale, Dt, Dx, Dy, nx, ny, Re, time_step, u_true, v_true):

    '''

    DEPRECATED

    '''

    Z = simulate_uncertain_sindy(gp_dictionnary, param, n_samples, z0, t_grid, sindy_coef, n_coef)

    n_z = Z.shape[2]

    pred = autoencoder.decoder(torch.Tensor(Z)).detach().numpy()
    pred_mean = pred.mean(0)
    pred_std = pred.std(0)

    r_u, r_v, _, _ = residual(pred_mean, Dt, Dx, Dy, nx, ny, Re)

    r_u = r_u[time_step, :, :]
    r_v = r_v[time_step, :, :]

    mean_u, mean_v = pred_mean[time_step, :nx * ny], pred_mean[time_step, nx * ny:]
    std_u, std_v = pred_std[time_step, :nx * ny], pred_std[time_step, nx * ny:]

    mean_u = mean_u.reshape(nx, ny)
    mean_v = mean_v.reshape(nx, ny)

    std_u = std_u.reshape(nx, ny)
    std_v = std_v.reshape(nx, ny)


    fig = plt.figure(figsize = (20, 5))

    plt.subplot(261)
    for s in range(n_samples):
        for i in range(n_z):
            plt.plot(t_grid, Z[s, :, i], 'C' + str(i), alpha = 0.3)
    plt.title('Latent Space')

    plt.subplot(262)
    plt.contourf(x_mesh, y_mesh, mean_u[::scale, ::scale], 100, cmap = plt.cm.jet)
    plt.colorbar()
    plt.title('Decoder Mean Prediction (U)')

    plt.subplot(263)
    plt.contourf(x_mesh, y_mesh, mean_v[::scale, ::scale], 100, cmap = plt.cm.jet)
    plt.colorbar()
    plt.title('Decoder Mean Prediction (V)')

    plt.subplot(264)
    plt.contourf(x_mesh, y_mesh, u_true[time_step, ::scale, ::scale], 100, cmap = plt.cm.jet)
    plt.colorbar()
    plt.title('Ground True (U)')

    plt.subplot(265)
    plt.contourf(x_mesh, y_mesh, v_true[time_step, ::scale, ::scale], 100, cmap = plt.cm.jet)
    plt.colorbar()
    plt.title('Ground True (V)')

    plt.subplot(268)
    abs_u = np.abs(u_true[time_step, ::scale, ::scale] - mean_u[::scale, ::scale])
    plt.contourf(x_mesh, y_mesh, abs_u, 100, cmap = plt.cm.jet)
    plt.colorbar()
    plt.title('Absolute Error (U)')

    plt.subplot(269)
    abs_v = np.abs(v_true[time_step, ::scale, ::scale] - mean_v[::scale, ::scale])
    plt.contourf(x_mesh, y_mesh, abs_v, 100, cmap = plt.cm.jet)
    plt.colorbar()
    plt.title('Absolute Error (V)')

    plt.subplot(2, 6, 10)
    plt.contourf(x_mesh, y_mesh, std_u[::scale, ::scale], 100, cmap = plt.cm.jet)
    plt.colorbar()
    plt.title('Decoder Standard Deviation (U)')

    plt.subplot(2, 6, 11)
    plt.contourf(x_mesh, y_mesh, std_v[::scale, ::scale], 100, cmap = plt.cm.jet)
    plt.colorbar()
    plt.title('Decoder Standard Deviation (V)')

    plt.subplot(2, 6, 6)
    plt.contourf(x_mesh[1:-1, 1:-1], y_mesh[1:-1, 1:-1], r_u, 100, cmap = plt.cm.jet)
    plt.colorbar()
    plt.title('Residual (U)')

    plt.subplot(2, 6, 12)
    plt.contourf(x_mesh[1:-1, 1:-1], y_mesh[1:-1, 1:-1], r_v, 100, cmap = plt.cm.jet)
    plt.colorbar()
    plt.title('Residual (V)')

    plt.tight_layout()

    return fig


def plot_gps(gp_dictionnary, param_train, sindy_coef, a_grid, b_grid, n_a_grid, n_b_grid, param_grid, n_coef):

    '''

    Plots GP predictive means and standard deviations

    '''

    gp_pred = eval_gp(gp_dictionnary, param_grid, n_coef)

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



def initial_condition_latent(param_grid, initial_condition, ic, nx, ny, autoencoder):

    '''

    Outputs the initial condition in the latent space: Z0 = encoder(U0)

    '''


    n_param = param_grid.shape[0]
    Z0 = []

    for i in range(n_param):

        u0 = initial_condition(ic, param_grid[i, 0], param_grid[i, 1], nx, ny)
        u0 = u0.reshape(1, 1, -1)
        u0 = torch.Tensor(u0)
        z0 = autoencoder.encoder(u0)
        z0 = z0[0, 0, :].detach().numpy()
        Z0.append(z0)

    return Z0



def find_sindy_coef(Z, Dt, n_train, time_dim, loss_function):

    '''

    Computes the SINDy loss, reconstruction loss, and sindy coefficients

    '''


    loss_sindy = 0
    loss_coef = 0

    dZdt, Z = compute_sindy_data(Z, Dt)
    sindy_coef = []

    for i in range(n_train):

        dZdt_i = dZdt[i, :, :]
        Z_i = torch.cat([torch.ones(time_dim - 1, 1), Z[i, :, :]], dim = 1)
        coef_matrix_i = Z_i.pinverse() @ dZdt_i

        loss_sindy += loss_function(dZdt_i, Z_i @ coef_matrix_i)
        loss_coef += torch.norm(coef_matrix_i)

        sindy_coef.append(coef_matrix_i.detach().numpy())

    return loss_sindy, loss_coef, sindy_coef


def get_residual_error(autoencoder, Zis, n_a_grid, n_b_grid, n_samples, Dt, Dx, metric = 'mean'):

    '''

    DEPRECATED

    '''

    if metric == 'mean':
        max_mean_error_residual = 0
    elif metric == 'max':
        max_max_error_residual = 0

    m = 0

    for j in range(n_b_grid):
        for i in range(n_a_grid):

            Z_m = torch.Tensor(Zis[m])
            X_pred_m = autoencoder.decoder(Z_m).detach().numpy()

            error_residual = np.zeros(n_samples)
            for s in range(n_samples):
                U_s = X_pred_m[s, :, :].T
                _, e = residual(U_s, Dt, Dx)
                error_residual[s] = e

            mean_error_residual = error_residual.mean()
            max_error_residual = error_residual.max()

            if metric == 'mean':
                if mean_error_residual > max_mean_error_residual:
                    a_index = i
                    b_index = j
                    m_index = m
                    max_mean_error_residual = mean_error_residual
            elif metric == 'max':
                if max_error_residual > max_max_error_residual:
                    a_index = i
                    b_index = j
                    m_index = m
                    max_max_error_residual = max_error_residual

            m += 1

    return a_index, b_index, m_index


def get_max_std(autoencoder, Zis, n_a_grid, n_b_grid):

    '''

    Computes the maximum standard deviation accross the parameter space grid and finds the corresponding parameter location

    '''

    max_std = 0
    m = 0

    for j in range(n_b_grid):
        for i in range(n_a_grid):

            Z_m = torch.Tensor(Zis[m])
            X_pred_m = autoencoder.decoder(Z_m).detach().numpy()
            X_pred_m_std = X_pred_m.std(0)
            max_std_m = X_pred_m_std.max()


            if max_std_m > max_std:
                    a_index = i
                    b_index = j
                    m_index = m
                    max_std = max_std_m

            m += 1

    return a_index, b_index, m_index


def get_new_param(m_index, X_train, param_train, param_grid, time_dim, space_dim, ic, Re, nx, ny, nt, dt, nxy, dx, dy, maxitr, tol):

    '''

    Generates a new FOM data point for the parameter yielding the largest uncertainty

    '''

    new_a, new_w = param_grid[m_index, 0], param_grid[m_index, 1]
    new_param = np.array([[new_a, new_w]])

    u, v, _ = solver(ic, new_a, new_w, Re, nx, ny, nt, dt, nxy, dx, dy, maxitr, tol)
    new_X = np.hstack((u, v))
    new_X = new_X.reshape(1, time_dim, space_dim)
    new_X = torch.Tensor(new_X)

    X_train = torch.cat([X_train, new_X], dim = 0)
    param_train = np.vstack((param_train, new_param))

    return X_train, param_train


def residual(X_pred, Dt, Dx, Dy, nx, ny, Re):

    '''

    DEPRECATED

    '''

    u = X_pred[:, :nx * ny]
    v = X_pred[:, nx * ny:]

    u = u.reshape(-1, nx, ny)
    v = v.reshape(-1, nx, ny)

    dudt = (u[1:, :, :] - u[:-1, :, :]) / Dt
    dvdt = (v[1:, :, :] - v[:-1, :, :]) / Dt

    dudx = (u[:, 1:, :] - u[:, :-1, :]) / Dx
    dvdx = (v[:, 1:, :] - v[:, :-1, :]) / Dx
    dudy = (u[:, :, 1:] - u[:, :, :-1]) / Dy
    dvdy = (v[:, :, 1:] - v[:, :, :-1]) / Dy

    dudx2 = (u[:, 2:, :] - 2 * u[:, 1:-1, :] + u[:, :-2, :]) / Dx / Dx
    dvdx2 = (v[:, 2:, :] - 2 * v[:, 1:-1, :] + v[:, :-2, :]) / Dx / Dx
    dudy2 = (u[:, :, 2:] - 2 * u[:, :, 1:-1] + u[:, :, :-2]) / Dy / Dy
    dvdy2 = (v[:, :, 2:] - 2 * v[:, :, 1:-1] + v[:, :, :-2]) / Dy / Dy

    convection_u = u[:-1, 1:-1, 1:-1] * dudx[:-1, 1:, 1:-1] + v[:-1, 1:-1, 1:-1] * dudy[:-1, 1:-1, 1:]
    convection_v = u[:-1, 1:-1, 1:-1] * dvdx[:-1, 1:, 1:-1] + v[:-1, 1:-1, 1:-1] * dvdy[:-1, 1:-1, 1:]

    diffusion_u = dudx2[:-1, :, 1:-1] + dudy2[:-1, 1:-1, :]
    diffusion_v = dvdx2[:-1, :, 1:-1] + dvdy2[:-1, 1:-1, :]

    r_u = dudt[:, 1:-1, 1:-1] + convection_u - diffusion_u / Re
    r_v = dvdt[:, 1:-1, 1:-1] + convection_v - diffusion_v / Re

    e_u = np.linalg.norm(r_u, axis = (1, 2))
    e_v = np.linalg.norm(r_v, axis = (1, 2))

    return r_u, r_v, e_u, e_v


def compute_errors(n_a_grid, n_b_grid, Zis, autoencoder, X_test):

    '''

    Compute the maximum relative errors accross the parameter space grid

    '''

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

            max_e_relative[j, i] = max_e_relative_m
            max_e_relative_mean[j, i] = max_e_relative_m_mean
            max_std[j, i] = max_std_m

            m += 1

    return max_e_relative, max_e_relative_mean, max_std

def plot_errors(error, n_a_grid, n_b_grid, a_grid, b_grid, param_train, n_init, normalize = False, percent = True, title = None):

    '''

    Plot errors, "error" can be either the max residual error, relative errors, or standard deviations

    '''

    if title is None:
        title = 'Error'

    title += '\n'

    if percent:
        error *= 100

    fig, ax = plt.subplots(figsize = (15, 15))
    im = ax.imshow(error, cmap = plt.cm.bwr)
    plt.colorbar(im)

    ax.set_xticks(np.arange(0, n_a_grid, 2), labels = np.round(a_grid[0, ::2], 2))
    ax.set_yticks(np.arange(0, n_b_grid, 2), labels = np.round(b_grid[::2, 0], 2))

    for i in range(n_a_grid):
        for j in range(n_b_grid):
            if normalize is True:
                ax.text(j, i, round(error[i, j] / error.max(), 2), ha = 'center', va = 'center', color = 'k')
            else:
                ax.text(j, i, round(error[i, j], 2), ha = 'center', va = 'center', color = 'k')

    grid_square_x = np.arange(-0.5, n_a_grid, 1)
    grid_square_y = np.arange(-0.5, n_b_grid, 1)

    n_train = param_train.shape[0]
    for i in range(n_train):
        a_index = np.sum((a_grid[0, :] < param_train[i, 0]) * 1)
        b_index = np.sum((b_grid[:, 0] < param_train[i, 1]) * 1)

        if i < n_init:
            color = 'w'
        else:
            color = 'k'

        ax.plot([grid_square_x[a_index], grid_square_x[a_index]], [grid_square_y[b_index], grid_square_y[b_index] + 1], c = color, linewidth = 2)
        ax.plot([grid_square_x[a_index] + 1, grid_square_x[a_index] + 1], [grid_square_y[b_index], grid_square_y[b_index] + 1], c = color, linewidth = 2)
        ax.plot([grid_square_x[a_index], grid_square_x[a_index] + 1], [grid_square_y[b_index], grid_square_y[b_index]], c = color, linewidth = 2)
        ax.plot([grid_square_x[a_index], grid_square_x[a_index] + 1], [grid_square_y[b_index] + 1, grid_square_y[b_index] + 1], c = color, linewidth = 2)

    ax.set_xlabel('a', fontsize = 15)
    ax.set_ylabel('w', fontsize = 15)
    ax.set_title(title, fontsize = 25)


def direct_mean_prediction(a, b, u0, autoencoder, t_grid, gp_dictionnary, sindy_coef, device):

    '''

    Makes a ROM prediction for a given parameter using the predictive mean of the GPs (i.e. integrates only one system of ODEs,
    instead of integrating multiple samples and making a forward pass for each of them)

    '''

    coef_matrix = np.zeros_like(sindy_coef[0])
    coef_y, coef_x = coef_matrix.shape

    param = np.array([[a, b]])

    c = 1
    for i in range(coef_y):
        for j in range(coef_x):
            coef_matrix[i, j] = gp_dictionnary['coef_' + str(c)].predict(param).item()
            c += 1

    coef_matrix = coef_matrix.T
    dzdt = lambda z, t: coef_matrix[:, 1:] @ z + coef_matrix[:, 0]

    u0 = torch.Tensor(u0.reshape(1, 1, -1)).to(device)
    Z0 = autoencoder.encoder(u0)
    Z0 = Z0[0, 0, :].cpu().detach().numpy()

    Z = odeint(dzdt, Z0, t_grid)
    Z = Z.reshape(1, Z.shape[0], Z.shape[1])
    Z = torch.Tensor(Z).to(device)

    Pred = autoencoder.decoder(Z).cpu()
    Pred = Pred[0, :, :].T.detach().numpy()

    return Pred
