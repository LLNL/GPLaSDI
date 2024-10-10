import numpy as np
import torch

class Physics:
    # Physical space dimension
    dim = -1
    # solution (denoted as q) dimension
    qdim = -1
    # grid_size is the shape of the grid nd-array.
    grid_size = []
    # the shape of the solution nd-array.
    qgrid_size = []

    '''
        numpy nd-array, assuming the shape of:
        - 1d: (space_dim[0],)
        - 2d: (2, space_dim[0], space_dim[1])
        - 3d: (3, space_dim[0], space_dim[1], space_dim[2])
        - higher dimension...
    '''
    x_grid = np.array([])

    # the number of time steps, as a positive integer.
    nt = -1
    # time step size. assume constant for now.
    dt = -1.
    # time grid in numpy 1d array
    t_grid = np.array([])

    # Need an offline FOM simulation or not.
    # Set at the initialization.
    offline = False

    # list of parameter names to parse parameters.
    param_name = None

    def __init__(self, cfg, param_name=None):
        self.param_name = param_name
        return
    
    def initial_condition(self, param):
        raise RuntimeError("Abstract method Physics.initial_condition!")
        return np.array
    
    def solve(self, param):
        raise RuntimeError("Abstract method Physics.solve!")
        return torch.Tensor
    
    def export(self):
        raise RuntimeError("Abstract method Physics.export!")
        return dict
    
    def generate_solutions(self, params):
        '''
        Given 2d-array of params,
        generate solutions of size params.shape[0].
        params.shape[1] must match the required size of
        parameters for the specific physics.
        '''
        assert(params.ndim == 2)
        n_param = len(params)
        print("Generating %d samples" % n_param)

        X_train = None
        for k, param in enumerate(params):
            new_X = self.solve(param)
            assert(new_X.size(0) == 1) # should contain one parameter case.
            if (X_train is None):
                X_train = new_X
            else:
                X_train = torch.cat([X_train, new_X], dim = 0)

            print("%d/%d complete" % (k+1, n_param))
        
        return X_train

    def residual(self, Xhist):
        raise RuntimeError("Abstract method Physics.residual!")
        return res, res_norm
    
class OfflineFOM(Physics):
    def __init__(self, cfg, param_name=None):
        super().__init__(cfg, param_name)
        self.offline = True

        assert('offline_fom' in cfg)
        from ..inputs import InputParser
        parser = InputParser(cfg['offline_fom'], name="offline_fom_input")

        self.dim = parser.getInput(['space_dimension'], datatype=int)
        self.qdim = parser.getInput(['solution_dimension'], datatype=int)

        self.grid_size = parser.getInput(['grid_size'], datatype=list)
        self.qgrid_size = self.grid_size
        if (self.qdim > 1):
            self.qgrid_size = [self.qdim] + self.qgrid_size
        assert(self.dim == len(self.grid_size))

        #TODO(kevin): a general file loading for spatial grid
        #             There can be unstructured grids as well.
        self.x_grid = None

        # Assume uniform time stepping for now.
        self.nt = parser.getInput(['number_of_timesteps'], datatype=int)
        self.dt = parser.getInput(['timestep_size'], datatype=float)
        self.t_grid = np.linspace(0.0, (self.nt-1) * self.dt, self.nt)

        return
    
    def generate_solutions(self, params):
        raise RuntimeError("OfflineFOM does not support generate_solutions!!")
        return

    def export(self):
        dict_ = {'t_grid' : self.t_grid, 'x_grid' : self.x_grid, 'dt' : self.dt}
        return dict_

