import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags
import torch
from ..inputs import InputParser
from . import Physics
from ..fd import FDdict

class Burgers1D(Physics):
    a_idx = None # parameter index for a
    w_idx = None # parameter index for w

    def __init__(self, cfg, param_name=None):
        super().__init__(cfg, param_name)

        self.qdim = 1
        self.dim = 1

        assert('burgers1d' in cfg)
        parser = InputParser(cfg['burgers1d'], name="burgers1d_input")

        self.offline = parser.getInput(['offline_driver'], fallback=False)

        self.nt = parser.getInput(['number_of_timesteps'], datatype=int)
        self.grid_size = parser.getInput(['grid_size'], datatype=list)
        self.qgrid_size = self.grid_size
        assert(self.dim == len(self.grid_size))

        self.xmin = parser.getInput(['xmin'], datatype=float)
        self.xmax = parser.getInput(['xmax'], datatype=float)
        self.dx = (self.xmax - self.xmin) / (self.grid_size[0] - 1)
        assert(self.dx > 0.)

        self.tmax = parser.getInput(['simulation_time'])
        self.dt = self.tmax / (self.nt - 1)

        self.x_grid = np.linspace(self.xmin, self.xmax, self.grid_size[0])
        self.t_grid = np.linspace(0, self.tmax, self.nt)

        self.maxk = parser.getInput(['maxk'], fallback=10)
        self.convergence_threshold = parser.getInput(['convergence_threshold'], fallback=1.e-8)

        if (self.param_name is not None):
            if 'a' in self.param_name:
                self.a_idx = self.param_name.index('a')
            if 'w' in self.param_name:
                self.w_idx = self.param_name.index('w')
        return
    
    def initial_condition(self, param):
        a, w = 1.0, 1.0
        if 'a' in self.param_name:
            a = param[self.a_idx]
        if 'w' in self.param_name:
            w = param[self.w_idx]

        return a * np.exp(- self.x_grid ** 2 / 2 / w / w)
    
    def solve(self, param):
        u0 = self.initial_condition(param)

        new_X = solver(u0, self.maxk, self.convergence_threshold, self.nt - 1, self.grid_size[0], self.dt, self.dx)
        new_X = new_X.reshape(1, self.nt, self.grid_size[0])
        return torch.Tensor(new_X)
    
    def export(self):
        dict_ = {'t_grid' : self.t_grid, 'x_grid' : self.x_grid, 'dt' : self.dt, 'dx' : self.dx}
        return dict_
    
    def residual(self, Xhist):
        # first axis is time index, and second index is spatial index.
        dUdx = (Xhist[:, 1:] - Xhist[:, :-1]) / self.dx
        dUdt = (Xhist[1:, :] - Xhist[:-1, :]) / self.dt

        r = dUdt[:, :-1] - Xhist[:-1, :-1] * dUdx[:-1, :]
        e = np.linalg.norm(r)

        return r, e

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

def main():
    import argparse
    import yaml
    import h5py
    import sys
    parser = argparse.ArgumentParser(description = "",
                                 formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config_file', metavar='string', type=str,
                        help='config file to run LasDI workflow.\n')
    args = parser.parse_args(sys.argv[1:])
    print("config file: %s" % args.config_file)

    # Read config file
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
        cfg_parser = InputParser(config, name='main')

    # initialize parameter space and physics class
    from ..param import ParameterSpace
    param_space = ParameterSpace(config)
    physics = Burgers1D(config['physics'], param_space.param_name)

    # read training parameter points
    train_param_file = cfg_parser.getInput(['workflow', 'offline_greedy_sampling', 'train_param_file'], datatype=str)
    train_sol_file = cfg_parser.getInput(['workflow', 'offline_greedy_sampling', 'train_sol_file'], datatype=str)
    with h5py.File(train_param_file, 'r') as f:
        new_train_params = f['train_params'][...]

    # generate and write FOM solution
    new_X = physics.generate_solutions(new_train_params)
    with h5py.File(train_sol_file, 'w') as f:
        f.create_dataset("train_sol", new_X.shape, data=new_X)

    # check if test parameter points exist
    test_param_file = cfg_parser.getInput(['workflow', 'offline_greedy_sampling', 'test_param_file'], datatype=str)
    import os.path
    if (os.path.isfile(test_param_file)):
        # read test parameter points
        test_sol_file = cfg_parser.getInput(['workflow', 'offline_greedy_sampling', 'test_sol_file'], datatype=str)
        with h5py.File(test_param_file, 'r') as f:
            new_test_params = f['test_params'][...]

        # generate and write FOM solution
        new_X = physics.generate_solutions(new_test_params)
        with h5py.File(test_sol_file, 'w') as f:
            f.create_dataset("test_sol", new_X.shape, data=new_X)

    return