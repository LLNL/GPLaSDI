# -------------------------------------------------------------------------------------------------
# Inputs
# -------------------------------------------------------------------------------------------------

import  numpy               as      np
from    scipy.sparse.linalg import  spsolve
from    scipy.sparse        import  spdiags
import  torch

from    ..inputs            import  InputParser
from    .                   import  Physics
from    ..fd                import  FDdict



# -------------------------------------------------------------------------------------------------
# Burgers 1D class
# -------------------------------------------------------------------------------------------------

class Burgers1D(Physics):
    # Class variables
    a_idx = None # parameter index for a
    w_idx = None # parameter index for w


    
    def __init__(self, cfg : dict, param_name : list[str] = None) -> None:
        """
        This is the initializer for the Burgers Physics class. This class essentially acts as a 
        wrapper around a 1D Burgers solver.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        cfg: A dictionary housing the settings for the Burgers object. This should be the "physics"
        sub-dictionary of the configuration file. 

        param_name: A list of strings. There should be one list item for each parameter. The i'th 
        element of this list should be a string housing the name of the i'th parameter.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Call the super class initializer.
        super().__init__(cfg, param_name)

        # The solution to Burgers' equation is scalar valued, so the qdim is 1. Likewise, since 
        # there is only one spatial dimension in the 1D burgers example, dim is also 1.
        self.qdim   : int   = 1
        self.dim    : int   = 1

        # Make sure the configuration dictionary is actually for Burgers' equation.
        assert('burgers1d' in cfg)
        
        # Now, get a parser for cfg.
        parser : InputParser = InputParser(cfg['burgers1d'], name = "burgers1d_input")

        # Fetch variables from the configuration. 
        self.offline    : bool      = parser.getInput(['offline_driver'],       fallback = False)   # TODO: ??? What does this do ???
        self.nt         : int       = parser.getInput(['number_of_timesteps'],  datatype = int)     # number of time steps when solving 
        self.grid_size  : list[int] = parser.getInput(['grid_size'],            datatype = list)    # number of grid points along each spatial axis
        self.qgrid_size : list[int] = self.grid_size                
        
        # If there are n spatial dimensions, then the grid needs to have n axes (one for each 
        # dimension). Make sure this is the case.
        assert(self.dim == len(self.grid_size))

        # Fetch more variables from the 
        self.xmin = parser.getInput(['xmin'], datatype = float)         # Minimum value of the spatial variable in the problem domain
        self.xmax = parser.getInput(['xmax'], datatype = float)         # Maximum value of the spatial variable in the problem domain
        self.dx = (self.xmax - self.xmin) / (self.grid_size[0] - 1)     # Spacing between grid points along the spatial axis.
        assert(self.dx > 0.)

        self.tmax   : float     = parser.getInput(['simulation_time'])  # Final simulation time. We solve form t = 0 to t = tmax
        self.dt     : float     = self.tmax / (self.nt - 1)             # step size between successive time steps/the time step we use when solving.

        # Set up the spatial, temporal grid.
        self.x_grid : np.ndarray    = np.linspace(self.xmin, self.xmax, self.grid_size[0])
        self.t_grid : np.ndarray    = np.linspace(0, self.tmax, self.nt)

        self.maxk                   : int   = parser.getInput(['maxk'], fallback = 10)      # TODO: ??? What is this ???
        self.convergence_threshold  : float = parser.getInput(['convergence_threshold'], fallback = 1.e-8)

        # Determine which index corresponds to 'a' and 'w' (we pass an array of parameter values, 
        # we need this information to figure out which element corresponds to which variable).
        if (self.param_name is not None):
            if 'a' in self.param_name:
                self.a_idx = self.param_name.index('a')
            if 'w' in self.param_name:
                self.w_idx = self.param_name.index('w')
        
        # All done!
        return
    


    def initial_condition(self, param : np.ndarray) -> np.ndarray:
        """
        Evaluates the initial condition along the spatial grid. For this class, we use the 
        following initial condition:
            u(x, 0) = a*exp(-x^2 / (2*w^2))
        where a and w are the corresponding parameter values.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param: A 1d numpy.ndarray object with two elements corresponding to the values of the w 
        and a parameters. self.a_idx and self.w_idx tell us which index corresponds to which 
        variable.
        

        -------------------------------------------------------------------------------------------
        Returns 
        -------------------------------------------------------------------------------------------

        A 1d numpy.ndarray object of length self.grid_size[0] (the number of grid points along the 
        spatial axis).
        """

        # Fetch the parameter values.
        a, w = 1.0, 1.0
        if 'a' in self.param_name:
            a = param[self.a_idx]
        if 'w' in self.param_name:
            w = param[self.w_idx]  

        # Compute the initial condition and return!
        return a * np.exp(- self.x_grid ** 2 / 2 / w / w)
    


    def solve(self, param : np.ndarray) -> torch.Tensor:
        """
        Solves the 1d burgers equation when the IC uses the parameters in the param array.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param: A 1d numpy.ndarray object with two elements corresponding to the values of the w 
        and a parameters. self.a_idx and self.w_idx tell us which index corresponds to which 
        variable.
        

        -------------------------------------------------------------------------------------------
        Returns 
        -------------------------------------------------------------------------------------------

        A 3d torch.Tensor object of shape (1, nt, nx), where nt is the number of points along the 
        temporal grid and nx is the number along the spatial grid.
        """
        
        # Fetch the initial condition.
        u0 : np.ndarray = self.initial_condition(param)

        # Solve the PDE and then reshape the result to be a 3d tensor with a leading dimension of 
        # size 1.
        new_X = solver(u0, self.maxk, self.convergence_threshold, self.nt - 1, self.grid_size[0], self.dt, self.dx)
        new_X = new_X.reshape(1, self.nt, self.grid_size[0])

        # All done!
        return torch.Tensor(new_X)
    


    def export(self) -> dict:
        """
        Returns a dictionary housing self's internal state. You can use this dictionary to 
        effectively serialize self.
        """

        dict_ : dict = {'t_grid' : self.t_grid, 'x_grid' : self.x_grid, 'dt' : self.dt, 'dx' : self.dx}
        return dict_
    

    
    def residual(self, Xhist : np.ndarray) -> tuple[np.ndarray, float]:
        """
        This function computes the PDE residual (difference between the left and right hand side
        of Burgers' equation when we substitute in the solution in Xhist).


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Xhist: A 2d numpy.ndarray object of shape (nt, nx), where nt is the number of points along
        the temporal axis and nx is the number of points along the spatial axis. The i,j element of
        this array should have the j'th component of the solution at the i'th time step.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        A two element tuple. The first is a numpy.ndarray object of shape (nt - 2, nx - 2) whose 
        i, j element holds the residual at the i + 1'th temporal grid point and the j + 1'th 
        spatial grid point. 
        """
        
        # First, approximate the spatial and teporal derivatives.
        # first axis is time index, and second index is spatial index.
        dUdx = (Xhist[:, 1:] - Xhist[:, :-1]) / self.dx
        dUdt = (Xhist[1:, :] - Xhist[:-1, :]) / self.dt

        # compute the residual + the norm of the residual.
        r   : np.ndarray    = dUdt[:, :-1] - Xhist[:-1, :-1] * dUdx[:-1, :]
        e   : float         = np.linalg.norm(r)

        # All done!
        return r, e



# -------------------------------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------------------------------

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
    J = spdiags(data, [0, -1], nx - 1, nx - 1, format = 'csr')
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



# -------------------------------------------------------------------------------------------------
# Main function (if running this file as a script).
# -------------------------------------------------------------------------------------------------

def main():
    import argparse
    import yaml
    import h5py
    import sys
    parser = argparse.ArgumentParser(description = "",
                                 formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config_file', metavar = 'string', type = str,
                        help = 'config file to run LasDI workflow.\n')
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
    train_param_file = cfg_parser.getInput(['workflow', 'offline_greedy_sampling', 'train_param_file'], datatype = str)
    train_sol_file = cfg_parser.getInput(['workflow', 'offline_greedy_sampling', 'train_sol_file'], datatype = str)
    with h5py.File(train_param_file, 'r') as f:
        new_train_params = f['train_params'][...]

    # generate and write FOM solution
    new_X = physics.generate_solutions(new_train_params)
    with h5py.File(train_sol_file, 'w') as f:
        f.create_dataset("train_sol", new_X.shape, data = new_X)

    # check if test parameter points exist
    test_param_file = cfg_parser.getInput(['workflow', 'offline_greedy_sampling', 'test_param_file'], datatype = str)
    import os.path
    if (os.path.isfile(test_param_file)):
        # read test parameter points
        test_sol_file = cfg_parser.getInput(['workflow', 'offline_greedy_sampling', 'test_sol_file'], datatype = str)
        with h5py.File(test_param_file, 'r') as f:
            new_test_params = f['test_params'][...]

        # generate and write FOM solution
        new_X = physics.generate_solutions(new_test_params)
        with h5py.File(test_sol_file, 'w') as f:
            f.create_dataset("test_sol", new_X.shape, data = new_X)

    return