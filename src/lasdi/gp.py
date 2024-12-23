import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
from sklearn.gaussian_process import GaussianProcessRegressor

def fit_gps(X, Y):

    '''
    Trains each GP given the interpolation dataset.
    X: (n_train, n_param) numpy 2d array
    Y: (n_train, n_coef) numpy 2d array
    We assume each target coefficient is independent with each other.
    gp_dictionary is a dataset containing the trained GPs (as sklearn objects)

    '''
    import GPy

    n_coef = 1 if (Y.ndim == 1) else Y.shape[1]
    if (Y.ndim == 1):
        Y = Y.reshape(-1, 1)

    if (X.ndim == 1):
        X = X.reshape(-1, 1)

    gp_dictionary = []

    for k in range(n_coef):
        kernel = GPy.kern.RBF(input_dim=X.shape[1], ARD=False, variance=1., lengthscale=1.)
        # NOTE(kevin): using Y[:, k:k+1] keeps the shape of (n_samples, 1)
        m = GPy.models.GPRegression(X, Y[:, k:k+1], kernel, noise_var=1e-10)
        m.optimize_restarts(num_restarts=10, messages=False)
        # m.optimize(messages=True)

        gp_dictionary += [m]

    return gp_dictionary

def fit_gps_botorch(X, Y):

    '''
    Trains each GP given the interpolation dataset.
    X: (n_train, n_param) numpy 2d array
    Y: (n_train, n_coef) numpy 2d array
    We assume each target coefficient is independent with each other.
    gp_dictionary is a dataset containing the trained GPs (as sklearn objects)

    '''
    import torch
    from botorch.models.gp_regression import SingleTaskGP
    from gpytorch.kernels import RBFKernel, ScaleKernel
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood

    n_coef = 1 if (Y.ndim == 1) else Y.shape[1]
    Y = torch.tensor(Y, dtype=torch.float64)
    if (Y.ndim == 1):
        Y = Y.view(-1, 1)

    # Y[:, 0:1] keeps the shape as (nsample, 1)
    Yvar = torch.full_like(Y[:, 0:1], 1e-6)

    X = torch.tensor(X, dtype=torch.float64)
    if (X.ndim == 1):
        X = X.view(-1, 1)

    gp_dictionary = []

    for k in range(n_coef):
        cov_module = ScaleKernel(RBFKernel())

        model = SingleTaskGP(X, Y[:, k:k+1], Yvar,
                            # outcome_transform=None,
                            covar_module=cov_module)

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        gp_dictionary += [model]

    return gp_dictionary

def fit_gps_sklearn(X, Y):

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

def eval_gp(gp_dictionary, param_grid):

    '''

    Computes the GPs predictive mean and standard deviation for points of the parameter space grid

    '''
    import torch

    n_coef = len(gp_dictionary)
    if (param_grid.ndim == 1):
        param_grid = param_grid.reshape(1, -1)
    n_points = param_grid.shape[0]

    pred_mean, pred_std = np.zeros([n_points, n_coef]), np.zeros([n_points, n_coef])
    for k, gp in enumerate(gp_dictionary):
        avgk, stdk = gp.predict(param_grid)
        pred_mean[:, k], pred_std[:, k] = avgk.flatten(), stdk.flatten()

    return pred_mean, pred_std

def eval_gp_botorch(gp_dictionnary, param_grid):

    '''

    Computes the GPs predictive mean and standard deviation for points of the parameter space grid

    '''
    import torch

    n_coef = len(gp_dictionnary)
    param_grid = torch.tensor(param_grid, dtype=torch.float64)
    if (param_grid.ndim == 1):
        param_grid = param_grid.view(1, -1)
    n_points = param_grid.shape[0]

    pred_mean, pred_std = np.zeros([n_points, n_coef]), np.zeros([n_points, n_coef])
    for k, gp in enumerate(gp_dictionnary):
        pred = gp(param_grid)
        pred_mean[:, k] = pred.mean.detach().numpy()
        pred_std[:, k] = pred.stddev.detach().numpy()

    return pred_mean, pred_std

def eval_gp_sklearn(gp_dictionnary, param_grid):

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

def sample_coefs(gp_dictionary, param, n_samples):

    '''

    Generates sample sets of ODEs for one given parameter.
    coef_samples is a list of length n_samples, where each terms is a matrix of SINDy coefficients sampled from the GP predictive
    distributions

    '''

    n_coef = len(gp_dictionary)
    coef_samples = np.zeros([n_samples, n_coef])

    if param.ndim == 1:
        param = param.reshape(1, -1)
    n_points = param.shape[0]
    assert(n_points == 1)

    pred_mean, pred_std = eval_gp(gp_dictionary, param)
    pred_mean, pred_std = pred_mean[0], pred_std[0]

    for s in range(n_samples):
        for k in range(n_coef):
            coef_samples[s, k] = np.random.normal(pred_mean[k], pred_std[k])

    return coef_samples