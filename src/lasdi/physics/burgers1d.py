import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags

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

def initial_condition(a, w, x_grid):

    return a * np.exp(-x_grid ** 2 / 2 / w / w)

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

def initial_train_data(config):

    time_dim = config['time_dim']
    space_dim = config['space_dim']

    xmin = config['xmin']
    xmax = config['xmax']
    tmax = config['tmax']

    Dx = (xmax - xmin) / (space_dim - 1)
    Dt = tmax / (time_dim - 1)
    assert(Dx > 0.)
    assert(Dt > 0.)

    x_grid = np.linspace(xmin, xmax, space_dim)
    t_grid = np.linspace(0, tmax, time_dim)

    # TODO(kevin): generalize parameters
    a_min = config['initial_train']['a_min']
    a_max = config['initial_train']['a_max']
    w_min = config['initial_train']['w_min']
    w_max = config['initial_train']['w_max']

    a_train = np.array([a_min, a_max])
    w_train = np.array([w_min, w_max])

    a_train, w_train = np.meshgrid(a_train, w_train)
    param_train = np.hstack((a_train.reshape(-1, 1), w_train.reshape(-1, 1)))
    n_train = param_train.shape[0]

    n_a_grid = config['initial_train']['n_a_grid']
    n_w_grid = config['initial_train']['n_w_grid']
    a_grid = np.linspace(a_min, a_max, n_a_grid)
    w_grid = np.linspace(w_min, w_max, n_w_grid)
    a_grid, w_grid = np.meshgrid(a_grid, w_grid)
    param_grid = np.hstack((a_grid.reshape(-1, 1), w_grid.reshape(-1, 1)))
    n_test = param_grid.shape[0]

    U0 = [initial_condition(param_train[i, 0], param_train[i, 1], x_grid) for i in range(n_train)]
    X_train = generate_initial_data(U0, time_dim, space_dim, Dt, Dx)

    data_train = {'param_train' : param_train, 'X_train' : X_train, 'n_train' : n_train, 'U0' : U0}
    train_filename = config['initial_train']['train_data']
    from os.path import dirname
    from pathlib import Path
    Path(dirname(train_filename)).mkdir(parents=True, exist_ok=True)
    np.save(train_filename, data_train)

    U0 = [initial_condition(param_grid[i, 0], param_grid[i, 1], x_grid) for i in range(n_test)]
    X_test = generate_initial_data(U0, time_dim, space_dim, Dt, Dx)

    data_test = {'param_grid' : param_grid, 'X_test' : X_test, 'n_test' : n_test, 'U0' : U0}
    test_filename = config['initial_train']['test_data']
    Path(dirname(test_filename)).mkdir(parents=True, exist_ok=True)
    np.save(test_filename, data_test)

    return data_train