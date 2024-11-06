# -------------------------------------------------------------------------------------------------
# Imports  
# -------------------------------------------------------------------------------------------------

import  numpy   as np
import  torch



# -------------------------------------------------------------------------------------------------
# Physics class 
# -------------------------------------------------------------------------------------------------

class Physics:
    # Physical space dimension
    dim : int= -1

    # The fom solution can be vector valued. If it is, then qdim specifies the dimensionality of 
    # the fom solution at each point. If the solution is scalar valued, then qdim = -1. 
    qdim : int = -1
    
    # grid_size is the shape of the grid nd-array.
    grid_size : list[int] = []
    
    # the shape of the solution nd-array. This is just the qgrid_size with the qdim prepended onto 
    # it.
    qgrid_size : list[int] = []
    
    '''
        numpy nd-array, assuming the shape of:
        - 1d: (space_dim[0],)
        - 2d: (2, space_dim[0], space_dim[1])
        - 3d: (3, space_dim[0], space_dim[1], space_dim[2])
        - higher dimension...
    '''
    x_grid : np.ndarray = np.array([])

    # the number of time steps, as a positive integer.
    nt : int = -1

    # time step size. assume constant for now. 
    dt : float = -1.

    # time grid in numpy 1d array. 
    t_grid : np.ndarray = np.array([])

    # Need an offline FOM simulation or not.
    # Set at the initialization.
    offline : bool = False

    # list of parameter names to parse parameters.
    param_name : list[str] = None



    def __init__(self, cfg : dict, param_name : list[str] = None) -> None:
        """
        A Physics object acts as a wrapper around a solver for a particular equation. The initial 
        condition in that function can have named parameters. Each physics object should have a 
        solve method to solve the underlying equation for a given set of parameters, and an 
        initial condition function to recover the equation's initial condition for a specific set 
        of parameters.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        cfg: A dictionary housing the settings for the Physics object. This should be the "physics"
        sub-dictionary of the main configuration file. 

        param_name: A list of strings. There should be one list item for each parameter. The i'th 
        element of this list should be a string housing the name of the i'th parameter.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """
        
        self.param_name = param_name
        return
    


    def initial_condition(self, param : np.ndarray) -> np.ndarray:
        """
        The user should write an instance of this method for their specific Physics sub-class.
        It should evaluate and return the initial condition along the spatial grid.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param: A 1d numpy.ndarray object holding the value of self's parameters (necessary to 
        specify the IC).
        

        -------------------------------------------------------------------------------------------
        Returns 
        -------------------------------------------------------------------------------------------

        A d-dimensional numpy.ndarray object of shape self.grid_size, where 
        d = len(self.grid_size). This should hold the IC evaluated on self's spatial grid 
        (self.x_grid)
        """

        raise RuntimeError("Abstract method Physics.initial_condition!")
        return np.array
    


    def solve(self, param : np.ndarray) -> torch.Tensor:
        """
        The user should write an instance of this method for their specific Physics sub-class.
        This function should solve the underlying equation when the IC uses the parameters in 
        param.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param: A 1d numpy.ndarray object with two elements corresponding to the values of the 
        initial condition parameters.
        

        -------------------------------------------------------------------------------------------
        Returns 
        -------------------------------------------------------------------------------------------

        A (ns + 2)-dimensional torch.Tensor object of shape (1, nt, nx[0], .. , nx[ns - 1]), 
        where nt is the number of points along the temporal grid and nx = self.grid_size specifies 
        the number of grid points along the axes in the spatial grid.
        """

        raise RuntimeError("Abstract method Physics.solve!")
        return torch.Tensor
    


    def export(self) -> dict:
        """
        This function should return a dictionary that houses self's state. I
        """
        raise RuntimeError("Abstract method Physics.export!")
        return dict
    


    def generate_solutions(self, params : np.ndarray) -> torch.Tensor:
        """
        Given 2d-array of params, generate solutions of size params.shape[0]. params.shape[1] must 
        match the required size of parameters for the specific physics.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param: a 2d numpy.ndarray object of shape (np, n), where np is the number of combinations 
        of parameters we want to test and n denotes the number of parameters in self's initial 
        condition function.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        A torch.Tensor object of shape (np, nt, nx[0], .. , nx[ns - 1]), where nt is the number of 
        points along the temporal grid and nx = self.grid_size specifies the number of grid points 
        along the axes in the spatial grid.
        """

        # Make sure we have a 2d grid of parameter values.
        assert(params.ndim == 2)
        n_param : int = len(params)

        # Report
        print("Generating %d samples" % n_param)

        # Cycle through the parameters.
        X_train = None
        for k, param in enumerate(params):
            # Solve the underlying equation using the current set of parameter values.
            new_X : torch.Tensor = self.solve(param)

            # Now, add this solution to the set of solutions.
            assert(new_X.size(0) == 1) # should contain one parameter case.
            if (X_train is None):
                X_train = new_X
            else:
                X_train = torch.cat([X_train, new_X], dim = 0)

            print("%d/%d complete" % (k+1, n_param))    
        
        # All done!
        return X_train



    def residual(self, Xhist : np.ndarray) -> tuple[np.ndarray, float]:
        """
        The user should write an instance of this method for their specific Physics sub-class.
        This function should compute the PDE residual (difference between the left and right hand 
        side of of the underlying physics equation when we substitute in the solution in Xhist).


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Xhist: A (ns + 1)-dimensional numpy.ndarray object of shape self.grid_size  = (nt, nx[0], 
        ... , nx[ns - 1]), where nt is the number of points along the temporal grid and nx = 
        self.grid_size specifies the number of grid points along the axes in the spatial grid. 
        The i,j(0), ... , j(ns - 1) element of this array should hold the value of the solution at 
        the i'th time step and the spatial grid point with index (j(0), ... , j(ns - 1)).


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        A two element tuple. The first is a numpy.ndarray object holding the residual on the 
        spatial and temporal grid. The second should be a float holding the norm of the residual.
        """

        raise RuntimeError("Abstract method Physics.residual!")
        return res, res_norm



# -------------------------------------------------------------------------------------------------
# Offline full order model class
# -------------------------------------------------------------------------------------------------

class OfflineFOM(Physics):
    def __init__(self, cfg, param_name = None):
        super().__init__(cfg, param_name)
        self.offline = True

        assert('offline_fom' in cfg)
        from ..inputs import InputParser
        parser = InputParser(cfg['offline_fom'], name = "offline_fom_input")

        self.dim = parser.getInput(['space_dimension'], datatype = int)
        self.qdim = parser.getInput(['solution_dimension'], datatype = int)

        self.grid_size = parser.getInput(['grid_size'], datatype = list)
        self.qgrid_size = self.grid_size
        if (self.qdim > 1):
            self.qgrid_size = [self.qdim] + self.qgrid_size
        assert(self.dim == len(self.grid_size))

        #TODO(kevin): a general file loading for spatial grid
        #             There can be unstructured grids as well.
        self.x_grid = None

        # Assume uniform time stepping for now.
        self.nt = parser.getInput(['number_of_timesteps'], datatype = int)
        self.dt = parser.getInput(['timestep_size'], datatype = float)
        self.t_grid = np.linspace(0.0, (self.nt-1) * self.dt, self.nt)

        return
    


    def generate_solutions(self, params):
        raise RuntimeError("OfflineFOM does not support generate_solutions!!")
        return



    def export(self):
        dict_ = {'t_grid' : self.t_grid, 'x_grid' : self.x_grid, 'dt' : self.dt}
        return dict_

