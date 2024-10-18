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

    # ParameterSpace object to parse parameters.
    param_space = None



    def __init__(self, param_space, cfg):
        self.param_space = param_space
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
        n_param = len(params)
        print("Generating %d samples" % n_param)

        X_train = None
        for k, param in enumerate(params):
            new_X = self.solve(param)
            if (X_train is None):
                X_train = new_X
            else:
                X_train = torch.cat([X_train, new_X], dim = 0)

            print("%d/%d complete" % (k+1, n_param))
        
        return X_train



    def residual(self, Xhist):
        raise RuntimeError("Abstract method Physics.residual!")
        return res, res_norm