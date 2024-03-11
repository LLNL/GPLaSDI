import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags
from scipy.integrate import odeint
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
import torch
import matplotlib.pyplot as plt


def initial_condition(a, w, x_grid):

    return a * np.exp(-x_grid ** 2 / 2 / w / w)

def solver(u0, maxk, convergence_threshold, nt, nx, Dt, Dx):
    '''

    Solves 1D Burgers equation for generating the data
    from https://github.com/LLNL/gLaSDI and https://github.com/LLNL/LaSDI

    '''

    c = Dt / Dx

    idxn1 = np.zeros(nx - 1, dtype = 'int')
    idxn1[1:] = np.arange(nx - 2)
    idxn1[0] = nx - 2

    u = np.zeros((nt + 1, nx))
    u[0] = u0

    for n in range(nt):
        uw = u[n, :-1].copy()
        r = residual_burgers(u[n, :-1], uw, c, idxn1)

        for k in range(maxk):
            J = jacobian(uw, c, idxn1, nx)
            duw = spsolve(J, -r)
            uw = uw + duw
            r = residual_burgers(u[n, :-1], uw, c, idxn1)

            rel_residual = np.linalg.norm(r) / np.linalg.norm(u[n, :-1])
            if rel_residual < convergence_threshold:
                u[n + 1, :-1] = uw.copy()
                u[n + 1, -1] = u[n + 1, 0]
                break

    return u

def residual_burgers(un, uw, c, idxn1):

    '''

    Compute 1D Burgers equation residual for generating the data
    from https://github.com/LLNL/gLaSDI and https://github.com/LLNL/LaSDI

    '''

    f = c * (uw ** 2 - uw * uw[idxn1])
    r = -un + uw + f

    return r


def jacobian(u, c, idxn1, nx):

    '''

    Compute 1D Burgers equation jacobian for generating the data
    from https://github.com/LLNL/gLaSDI and https://github.com/LLNL/LaSDI

    '''

    diag_comp = 1.0 + c * (2 * u - u[idxn1])
    subdiag_comp = np.ones(nx - 1)
    subdiag_comp[:-1] = -c * u[1:]
    data = np.array([diag_comp, subdiag_comp])
    J = spdiags(data, [0, -1], nx - 1, nx - 1, format='csr')
    J[0, -1] = -c * u[0]

    return J





def generate_initial_data(U0, nt, nx, Dt, Dx):

    '''

    Generates 1D Burgers equation initial training dataset

    '''

    maxk = 10
    convergence_threshold = 1e-8

    n_init = len(U0)

    for i in range(n_init):

        U = solver(U0[i], maxk, convergence_threshold, nt - 1, nx, Dt, Dx)
        U = U.reshape(1, nt, nx)

        if i == 0:
            X_train = U
        else:
            X_train = np.concatenate((X_train, U), axis = 0)

        print(i)

    return X_train



