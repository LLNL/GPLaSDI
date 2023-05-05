import numpy as np
from scipy.integrate import odeint
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
import torch
import matplotlib.pyplot as plt
import os
import time
from solver import *


def compute_sindy_data(Z, Dt):

    '''

    Builds the SINDy dataset, assuming linear terms. The time derivatives are computed through finite difference.

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



def fit_gps(interpolation_data, normalize = False):

    '''

    Trains each GP given the interpolation dataset.
    gp_dictionnary is a dataset containing the trained GPs (as sklearn objects)

    '''

    n_coef = interpolation_data['n_coef']
    X = interpolation_data['coef_1']['X']
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if normalize:
        mean = X.mean(0)
        std = X.std(0)
        X = (X - mean) / std

    gp_dictionnary = {}

    for i in range(n_coef):

        y = interpolation_data['coef_' + str(i + 1)]['y']

        kernel = ConstantKernel() * Matern(length_scale_bounds = (0.1, 1e5), nu = 1.5)
        # kernel = ConstantKernel() * RBF(length_scale_bounds = (0.1, 1e5))
        gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 10, random_state = 1)
        gp.fit(X, y)

        gp_dictionnary['coef_' + str(i + 1)] = gp

    if normalize:
        gp_dictionnary['param_mean'] = mean
        gp_dictionnary['param_std'] = std

    return gp_dictionnary



def eval_gp(gp_dictionnary, param_grid, n_coef, normalize = False):

    '''

    Computes the GPs predictive mean and standard deviation for points of the parameter space grid

    '''

    if normalize:
        mean = gp_dictionnary['param_mean']
        std = gp_dictionnary['param_std']
        param_grid = (param_grid - mean) / std

    gp_pred = {}

    for i in range(n_coef):

        gp = gp_dictionnary['coef_' + str(i + 1)]
        mean, std = gp.predict(param_grid, return_std = True)

        gp_pred['coef_' + str(i + 1)] = {}
        gp_pred['coef_' + str(i + 1)]['mean'] = mean
        gp_pred['coef_' + str(i + 1)]['std'] = std

    if normalize:
        gp_pred['param_mean'] = mean
        gp_pred['param_std'] = std

    return gp_pred



def interpolate_coef_matrix(gp_dictionnary, param, n_samples, n_coef, sindy_coef, normalize = False):

    '''

    Generates sample sets of ODEs for a given parameter.
    coef_samples is a list of length n_samples, where each terms is a matrix of SINDy coefficients sampled from the GP predictive
    distributions

    '''

    coef_samples = []
    coef_x, coef_y = sindy_coef[0].shape
    if param.ndim == 1:
        param = param.reshape(1, -1)

    gp_pred = eval_gp(gp_dictionnary, param, n_coef, normalize)

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


def simulate_uncertain_sindy(gp_dictionnary, param, n_samples, z0, t_grid, sindy_coef, n_coef, coef_samples = None, normalize = False):

    '''

    Integrates each ODE samples for a given parameter.

    '''

    if coef_samples is None:
        coef_samples = interpolate_coef_matrix(gp_dictionnary, param, n_samples, n_coef, sindy_coef, normalize)

    Z0 = [z0 for _ in range(n_samples)]
    Z = simulate_sindy(coef_samples, Z0, t_grid)

    return Z


def simulate_interpolated_sindy(param_grid, Z0, t_grid, n_samples, Dt, Z, param_train, normalize = False):

    '''

    Integrates each ODE samples for each parameter of the parameter grid.
    Z_simulated is a list of length param_grid.shape[0], where each term is a 3D tensor of the form [n_samples, time_dim, n_z]

    '''

    dZdt, Z = compute_sindy_data(Z, Dt)
    sindy_coef = solve_sindy(dZdt, Z)
    interpolation_data = build_interpolation_data(sindy_coef, param_train)
    gp_dictionnary = fit_gps(interpolation_data, normalize)
    n_coef = interpolation_data['n_coef']

    coef_samples = [interpolate_coef_matrix(gp_dictionnary, param_grid[i, :], n_samples, n_coef, sindy_coef, normalize) for i in range(param_grid.shape[0])]

    Z_simulated = [simulate_uncertain_sindy(gp_dictionnary, param_grid[i, 0], n_samples, Z0[i], t_grid, sindy_coef, n_coef, coef_samples[i], normalize) for i in range(param_grid.shape[0])]

    return Z_simulated, gp_dictionnary, interpolation_data, sindy_coef, n_coef, coef_samples


def initial_condition_latent(param_grid, initial_condition, autoencoder):

    '''

    Outputs the initial condition in the latent space: Z0 = encoder(U0)

    '''

    n_param = param_grid.shape[0]
    Z0 = []

    for i in range(n_param):

        u0 = initial_condition(param_grid[i, 0], param_grid[i, 1])
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


def get_new_param(m_index, X_train, param_train, param_grid, time_dim, space_dim, n_proc, noise):

    '''

    Generates a new FOM data point for the parameter yielding the largest uncertainty, using a HyPar simulation

    '''

    new_a, new_w = param_grid[m_index, 0], param_grid[m_index, 1]
    new_param = np.array([[new_a, new_w]])

    write_files(new_a, new_w, [n_proc, n_proc])
    time.sleep(1)
    run_hypar(n_proc)

    new_X = post_process_data(time_dim)
    new_X = normalize_add_noise(new_X, noise)
    new_X = new_X.reshape(1, time_dim, space_dim)
    new_X = torch.Tensor(new_X)

    X_train = torch.cat([X_train, new_X], dim = 0)
    param_train = np.vstack((param_train, new_param))

    data_bin = 'op_{t:05d}.bin'
    data_dat = 'op_{t:05d}.dat'

    for t in range(time_dim):
        os.system('rm ' + data_bin.format(t=t))
        os.system('rm ' + data_dat.format(t=t))

    return X_train, param_train
