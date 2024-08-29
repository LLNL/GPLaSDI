import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
from sklearn.gaussian_process import GaussianProcessRegressor

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

def fit_gps_obsolete(interpolation_data):

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

        # kernel = ConstantKernel() * Matern(length_scale_bounds = (0.01, 1e5), nu = 1.5)
        kernel = ConstantKernel() * RBF(length_scale_bounds = (0.1, 1e5))

        gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 10, random_state = 1)
        gp.fit(X, y)

        gp_dictionnary['coef_' + str(i + 1)] = gp

    return gp_dictionnary

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

def eval_gp_obsolete(gp_dictionnary, param_grid, n_coef):

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

def interpolate_coef_matrix(gp_dictionnary, param, n_samples, n_coef, sindy_coef):

    '''
    OBSOLETE
    
    Generates sample sets of ODEs for a given parameter.
    coef_samples is a list of length n_samples, where each terms is a matrix of SINDy coefficients sampled from the GP predictive
    distributions

    '''

    coef_samples = []
    coef_x, coef_y = sindy_coef[0].shape
    if param.ndim == 1:
        param = param.reshape(1, -1)

    gp_pred = eval_gp_obsolete(gp_dictionnary, param, n_coef)

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