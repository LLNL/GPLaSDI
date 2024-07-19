import numpy as np
import torch
from scipy.integrate import odeint
from .fd import SBP12, SBP24, SBP36, SBP48

FDdict = {'sbp12': SBP12(),
          'sbp24': SBP24(),
          'sbp36': SBP36(),
          'sbp48': SBP48()}

def compute_time_derivative(Z, Dt, fd_type):

    '''

    Builds the SINDy dataset, assuming only linear terms in the SINDy dataset. The time derivatives are computed through
    finite difference.

    Z is the encoder output (3D tensor), with shape [n_train, time_dim, space_dim]
    Dt is the size of timestep (assumed to be a uniform scalar)
    fd_type is the string that specifies finite-difference scheme for time derivative:
        - 'sbp12': summation-by-parts 1st/2nd (boundary/interior) order operator
        - 'sbp24': summation-by-parts 2nd/4th order operator
        - 'sbp36': summation-by-parts 3rd/6th order operator
        - 'sbp48': summation-by-parts 4th/8th order operator

    The output dZdt is a 3D tensor with the same shape of Z.

    '''

    fd = FDdict[fd_type]
    oper, _, _ = fd.getOperators(Z.size(1))
    
    ''' Is full vectorization possible? '''
    dZdt = torch.zeros(Z.size())
    for k, Zk in enumerate(Z):
        dZdt[k] = 1. / Dt * torch.sparse.mm(oper, Zk) 

    return dZdt

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
        # Z_i = Z[i, :, :]
        # Z_i = np.hstack((np.ones([time_dim, 1]), Z_i))
        Z_i = torch.cat([torch.ones(time_dim, 1), Z[i, :, :]], dim = 1)

        # c_i = np.linalg.lstsq(Z_i, dZdt_i)[0]
        c_i = torch.linalg.lstsq(Z_i.detach(), dZdt_i.detach()).solution.numpy()
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

def simulate_uncertain_sindy(gp_dictionnary, param, n_samples, z0, t_grid, sindy_coef, n_coef, coef_samples = None):

    '''

    Integrates each ODE samples for a given parameter.

    '''

    if coef_samples is None:
        from .interp import interpolate_coef_matrix
        coef_samples = interpolate_coef_matrix(gp_dictionnary, param, n_samples, n_coef, sindy_coef)

    Z0 = [z0 for _ in range(n_samples)]
    Z = simulate_sindy(coef_samples, Z0, t_grid)

    return Z

def simulate_interpolated_sindy(param_grid, Z0, t_grid, n_samples, Dt, Z, param_train, fd_type):

    '''

    Integrates each ODE samples for each parameter of the parameter grid.
    Z_simulated is a list of length param_grid.shape[0], where each term is a 3D tensor of the form [n_samples, time_dim, n_z]

    '''

    from .interp import build_interpolation_data, fit_gps, interpolate_coef_matrix

    dZdt = compute_time_derivative(Z, Dt, fd_type)
    sindy_coef = solve_sindy(dZdt, Z)
    interpolation_data = build_interpolation_data(sindy_coef, param_train)
    gp_dictionnary = fit_gps(interpolation_data)
    n_coef = interpolation_data['n_coef']

    coef_samples = [interpolate_coef_matrix(gp_dictionnary, param_grid[i, :], n_samples, n_coef, sindy_coef) for i in range(param_grid.shape[0])]

    Z_simulated = [simulate_uncertain_sindy(gp_dictionnary, param_grid[i, 0], n_samples, Z0[i], t_grid, sindy_coef, n_coef, coef_samples[i]) for i in range(param_grid.shape[0])]

    return Z_simulated, gp_dictionnary, interpolation_data, sindy_coef, n_coef, coef_samples