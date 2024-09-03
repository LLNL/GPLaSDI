import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
from sklearn.gaussian_process import GaussianProcessRegressor

def fit_gps(X, Y):

    '''
    Trains each GP given the interpolation dataset.
    X: (n_train, n_param) numpy 2d array
    Y: (n_train, n_coef) numpy 2d array
    We assume each target coefficient is independent with each other.
    gp_dictionnary is a dataset containing the trained GPs (as sklearn objects)

    '''

    n_coef = 1 if (Y.ndim == 1) else Y.shape[1]
    if (n_coef > 1):
        Y = Y.T

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    gp_dictionnary = []

    for yk in Y:
        # kernel = ConstantKernel() * Matern(length_scale_bounds = (0.01, 1e5), nu = 1.5)
        kernel = ConstantKernel() * RBF(length_scale_bounds = (0.1, 1e5))

        gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 10, random_state = 1)
        gp.fit(X, yk)

        gp_dictionnary += [gp]

    return gp_dictionnary

def eval_gp(gp_dictionnary, param_grid):

    '''

    Computes the GPs predictive mean and standard deviation for points of the parameter space grid

    '''

    n_coef = len(gp_dictionnary)
    if (param_grid.ndim == 1):
        param_grid = param_grid.reshape(1, -1)
    n_points = param_grid.shape[0]

    pred_mean, pred_std = np.zeros([n_points, n_coef]), np.zeros([n_points, n_coef])
    for k, gp in enumerate(gp_dictionnary):
        pred_mean[:, k], pred_std[:, k] = gp.predict(param_grid, return_std = True)

    return pred_mean, pred_std

def sample_coefs(gp_dictionnary, param, n_samples):

    '''

    Generates sample sets of ODEs for one given parameter.
    coef_samples is a list of length n_samples, where each terms is a matrix of SINDy coefficients sampled from the GP predictive
    distributions

    '''

    n_coef = len(gp_dictionnary)
    coef_samples = np.zeros([n_samples, n_coef])

    if param.ndim == 1:
        param = param.reshape(1, -1)
    n_points = param.shape[0]
    assert(n_points == 1)

    pred_mean, pred_std = eval_gp(gp_dictionnary, param)
    pred_mean, pred_std = pred_mean[0], pred_std[0]

    for s in range(n_samples):
        for k in range(n_coef):
            coef_samples[s, k] = np.random.normal(pred_mean[k], pred_std[k])

    return coef_samples