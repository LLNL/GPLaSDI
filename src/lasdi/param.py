# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import  numpy   as      np
from    .inputs import  InputParser



# -------------------------------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------------------------------

def get_1dspace_from_list(config : dict):
    Nx          = len(config['list'])
    paramRange  = np.array(config['list'])

    return Nx, paramRange



def create_uniform_1dspace(config : dict):
    Nx      = config['sample_size']
    minval  = config['min']
    maxval  = config['max']

    if (config['log_scale']):
        paramRange = np.exp(np.linspace(np.log(minval), np.log(maxval), Nx))
    else:
        paramRange = np.linspace(minval, maxval, Nx)
    
    return Nx, paramRange



getParam1DSpace = {'list'       : get_1dspace_from_list,
                   'uniform'    : create_uniform_1dspace}



# -------------------------------------------------------------------------------------------------
# ParameterSpace Class
# -------------------------------------------------------------------------------------------------

class ParameterSpace:
    # Initialize class variables
    param_list      = []
    param_name      = []
    train_space     = None
    test_space      = None
    n_test          = 0
    n_train         = 0
    n_init          = 0
    test_grid_sizes = []
    test_meshgrid   = None



    def __init__(self, config : dict) -> None:
        """
        Initializes a ParameterSpace object using the settings passed in the conf dictionary (which 
        should have been read from a yaml file).

        
        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        config: This is a dictionary that houses the settings we want to use to run the code. This 
        should have been read from a yaml file. We assume it contains the following keys. If one 
        or more keys are tabbed over relative to one key above them, then the one above is a 
        dictionary and the ones below should be keys within that dictionary.
            - parameter_space
                - parameters
            - TODO List keys we assume are in config.
        """

        # Make sure the configuration dictionary has a "parameter_space" setting. This should house 
        # information about which variables are present in the code, as well as how we want to test
        # the various possible parameter values.
        assert('parameter_space' in config);

        # Load the parameter_space settings into an InputParser object (which makes it easy to 
        # fetch setting values). We then fetch the list of parameters. Each parameters has a name, 
        # min and max, and information on how many instances we want.
        parser                          = InputParser(config['parameter_space'],    name        = "param_space_input")
        self.param_list : list[dict]    = parser.getInput(['parameters'],           datatype    = list)

        # Fetch the parameter names.
        self.param_name : list[str]     = []
        for param in self.param_list:
            self.param_name += [param['name']];

        # 
        self.train_space =  self.createInitialTrainSpace(self.param_list)
        self.n_init = self.train_space.shape[0]
        self.n_train = self.n_init

        test_space_type = parser.getInput(['test_space', 'type'], datatype = str)
        if (test_space_type == 'grid'):
            self.test_grid_sizes, self.test_meshgrid, self.test_space = self.createTestGridSpace(self.param_list)
            self.n_test = self.test_space.shape[0]

        return
    


    def createInitialTrainSpace(self, param_list : list[dict]) -> np.ndarray:
        """
        Sets up a grid of parameter values to test. Note that we only use the min and max value
        of each parameter when setting up this grid.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param_list: A list of parameter dictionaries. Each entry should be a dictionary with the 
        following keys:
            - name
            - min
            - max
        
            
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        A 2d array of shape (2)^k x k, where k is the number of parameters (k == len(param_list)).
        The i'th column is the flattened i'th mesh_grid array we when we create a mesh grid using 
        the min and max value of each parameter as the argument. See "createHyperMeshGrid" for 
        details. 
        
        Specifically, we return exactly what "createHyperGridSpace" returns. See the doc-string 
        for that function for further details. 
        """

        # We need to know the min and max value for each parameter to set up the grid of possible 
        # parameter values.
        paramRanges : list[np.ndarray] = []

        for param in param_list:    
            # Fetch the min, max value of the current parameter. 
            minval  : float = param['min']
            maxval  : float = param['max']
            
            # Store these values in an array and add them to the list.
            paramRanges += [np.array([minval, maxval])]

        # Use the ranges to set up a grid of possible parameter values.
        mesh_grids : tuple[np.ndarray]  = self.createHyperMeshGrid(paramRanges)

        # Now combine these grids into a 2d 
        return self.createHyperGridSpace(mesh_grids)
    


    def createTestGridSpace(self, param_list):
        paramRanges = []
        gridSizes = []

        for param in param_list:
            Nx, paramRange = getParam1DSpace[param['test_space_type']](param)
            gridSizes += [Nx]
            paramRanges += [paramRange]

        mesh_grids = self.createHyperMeshGrid(paramRanges)
        return gridSizes, mesh_grids, self.createHyperGridSpace(mesh_grids)
    


    def getParameter(self, param_vector):
        '''
            convert numpy array parameter vector to a dict.
            Physics class takes the dict for solve/initial_condition.
        '''
        assert(param_vector.size == len(self.param_name))

        param = {}
        for k, name in enumerate(self.param_name):
            param[name] = param_vector[k]

        return param
    


    def createHyperMeshGrid(self, param_ranges : list[np.ndarray]):
        '''


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param_ranges: list of numpy 1d arrays, each corresponding to 1d parameter grid space. The 
        i'th element of this list should be a 2-element numpy.ndarray object housing the max and 
        min value for the i'th parameter. The list size should equal the number of parameters. 
                        

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        the "paramSpaces" tuple. This is a tuple of numpy ndarray objects, the i'th one of which 
        gives the grid of parameter values for the i'th parameter. Specifically, if there are 
        k parameters and if the i'th parameter range object has Ni values, then the j'th return 
        array has shape (N1, ... , Nk) and the i(1), ... , i(k) element of this array houses the 
        i(j)'th value of the j'th parameter.

        Thus, if there are k parameters, the returned tuple has k elements, each one of 
        which is an array of shape N1, ... , Nk.
        '''

        # Fetch the ranges, add them to a tuple (this is what the meshgrid function needs).
        args = ()
        for paramRange in param_ranges:
            args += (paramRange,)

        # Use numpy's meshgrid function to generate the grids of parameter values.
        paramSpaces = np.meshgrid(*args, indexing='ij')

        # All done!
        return paramSpaces
    


    def createHyperGridSpace(self, mesh_grids : tuple[np.ndarray]):
        '''
        Flattens the mesh_grid numpy.ndarray objects returned by createHyperMeshGrid and combines 
        them into a single 2d array of shape (grid size) x (number of parameters) (see below).


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        mesh_grids: tuple of numpy nd arrays, corresponding to each parameter. This should ALWAYS
        be the output of the "CreateHyperMeshGrid" function. See the return section of that 
        function's docstring for details.
    
        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        The param_grid. This is a 2d numpy.ndarray object of shape (grid size) x (number of 
        parameters). If each element of mesh_grids is a numpy.ndarray object of shape N(1), ... , 
        N(k) (k parameters), then (grid size) = N(1)*N(2)*...*N(k) and (number of parameters) = k.
        '''

        # For each parameter, we flatten its mesh_grid into a 1d array (of length (grid size)). We
        # horizontally stack these flattened grids to get the final param_grid array.
        param_grid = None
        for k, paramSpace in enumerate(mesh_grids):
            # Special treatment for the first parameter to initialize param_grid
            if (k == 0):
                param_grid = paramSpace.reshape(-1, 1)
                continue

            # Flatten the new mesh grid and append it to the grid.
            param_grid = np.hstack((param_grid, paramSpace.reshape(-1, 1)))

        # All done!
        return param_grid
    


    def appendTrainSpace(self, param):
        assert(self.train_space.shape[1] == param.size)

        self.train_space = np.vstack((self.train_space, param))
        self.n_train = self.train_space.shape[0]
        return
    


    def export(self):
        dict_ = {'final_param_train': self.train_space,
                 'param_grid': self.test_space,
                 'test_grid_sizes': self.test_grid_sizes,
                 'test_meshgrid': self.test_meshgrid,
                 'n_init': self.n_init}
        return dict_
    


    def loadTrainSpace(self):
        raise RuntimeError("ParameterSpace.loadTrainSpace is not implemented yet!")
        return
