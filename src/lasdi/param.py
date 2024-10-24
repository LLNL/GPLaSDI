# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import  numpy   as      np
from    .inputs import  InputParser



# -------------------------------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------------------------------

def get_1dspace_from_list(param_dict : dict) -> tuple[int, np.ndarray]:
    """
    This function generates the parameter range (set of possible parameter values) for a parameter 
    that uses the list type test space. That is, "test_space_type" should be a key for the 
    parameter dictionary and the corresponding value should be "list". The parameter dictionary 
    should also have a "list" key whose value is a list of the possible parameter values.

    We parse this list and turn it into a numpy ndarray.

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    param_dict: A dictionary specifying one of the parameters. We should fetch this from the 
    configuration yaml file. It must have a "list" key whose corresponding value is a list of 
    floats. 


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Two arguments: Nx and paramRange. paramRange is a 1d numpy ndarray (whose ith value is the 
    i'th element of param_dict["list"]). Nx is the length of paramRange. 
    """

    # In this case, the parameter dictionary should have a "list" attribute which should list the 
    # parameter values we want to test. Fetch it (it's length is Nx) and use it to generate an
    # array of possible values.
    Nx          : int           = len(param_dict['list'])
    paramRange  : np.ndarray    = np.array(param_dict['list'])

    # All done!
    return Nx, paramRange



def create_uniform_1dspace(param_dict : dict) -> tuple[int, np.ndarray]:
    """
    This function generates the parameter range (set of possible parameter values) for a parameter 
    that uses the uniform type test space. That is, "test_space_type" should be a key for the 
    parameter dictionary and the corresponding value should be "uniform". The parameter dictionary 
    should also have the following keys:
        "min"
        "max"
        "sample_size"
        "log_scale"
    "min" and "max" specify the minimum and maximum value of the parameter, respectively. 
    "sample_size" specifies the number of parameter values we generate. Finally, log_scale, if 
    true, specifies if we should use a uniform or logarithmic spacing between samples of the 
    parameter.
    
    The values corresponding to "min" and "max" should be floats while the values corresponding to 
    "sample_size" and "log_scale" should be an int and a bool, respectively. 

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    param_dict: A dictionary specifying one of the parameters. We should fetch this from the 
    configuration yaml file. It must have a "min", "max", "sample_size", and "log_scale" 
    keys (see above).


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Two arguments: Nx and paramRange. paramRange is a 1d numpy ndarray (whose ith value is the 
    i'th possible value of the parameter. Thus, paramRange[0] = param_dict["min"] and 
    paramRange[-1] = param_dict["max"]). Nx is the length of paramRange or, equivalently 
    param_dict["sample_size"]. 
    """

    # Fetch the number of samples and the min/max value for this parameter.
    Nx      : int   = param_dict['sample_size']
    minval  : float = param_dict['min']
    maxval  : float = param_dict['max']

    # Generate the range of parameter values. Note that we have to generate a uniform grid in the 
    # log space, then exponentiate it to generate logarithmic spacing.
    if (param_dict['log_scale']):
        paramRange : np.ndarray = np.exp(np.linspace(np.log(minval), np.log(maxval), Nx))
    else:
        paramRange : np.ndarray = np.linspace(minval, maxval, Nx)
    
    # All done! 
    return Nx, paramRange



# A macro that allows us to switch function we use to generate generate a parameter's range. 
getParam1DSpace : dict[str, callable]    = {'list'       : get_1dspace_from_list,
                                            'uniform'    : create_uniform_1dspace}



# -------------------------------------------------------------------------------------------------
# ParameterSpace Class
# -------------------------------------------------------------------------------------------------

class ParameterSpace:
    # Initialize class variables
    param_list      : list[dict]        = []    # A list housing the parameter dictionaries (from the yml file)
    param_name      : list[str]         = []    # A list housing the parameter names.
    n_param         : int               = 0     # The number of parameters.
    train_space     : np.ndarray        = None  # A 2D array of shape (n_train, n_param) whose i,j element is the j'th parameter value in the i'th combination of training parameters.
    test_space      : np.ndarray        = None  # A 2D array of shape (n_test, n_param) whose i,j element is the j'th parameter value in the i'th combination of testing parameters.
    n_init          : int               = 0     # The number of combinations of parameters in the training set???
    test_grid_sizes : list[int]         = []    # A list whose i'th element is the number of different values of the i'th parameter in the test instances.
    test_meshgrid   : tuple[np.ndarray] = None



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
                - parameters (this should have at least one parameter defined!)
            - test_space
                - type (should be "grid")
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

        # First, let's fetch the set of possible parameter values. This yields a 2^k x k matrix,
        # where k is the number of parameters. The i,j entry of this matrix gives the value of the 
        # j'th parameter on the i'th instance.
        self.train_space    = self.createInitialTrainSpace(self.param_list)
        self.n_init         = self.train_space.shape[0]

        # Next, let's make a set of possible parameter values to test at.
        test_space_type = parser.getInput(['test_space', 'type'], datatype = str)
        if (test_space_type == 'grid'):
            # Generate the set possible parameter combinations. See the docstring for 
            # "createTestGridSpace" for details.
            self.test_grid_sizes, self.test_meshgrid, self.test_space = self.createTestGridSpace(self.param_list)

        # All done!
        return
    


    def n_train(self) -> int:
        """
        Returns the number of combinations of parameters in the training set.
        """

        return self.train_space.shape[0]
    


    def n_test(self) -> int:
        """
        Returns the number of combinations of parameters in the testing set.
        """

        return self.test_space.shape[0]



    def createInitialTrainSpace(self, param_list : list[dict]) -> np.ndarray:
        """
        Sets up a grid of parameter values to train at. Note that we only use the min and max value
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

        A 2d array of shape ((2)^k, k), where k is the number of parameters (k == len(param_list)).
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
    


    def createTestGridSpace(self, param_list : list[dict]) -> tuple[list[int], tuple[np.ndarray], np.ndarray]:
        """
        This function sets up a grid of parameter values to test at. 


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param_list: A list of parameter dictionaries. Each dictionary should either use the 
        "uniform" or "list" format. See create_uniform_1dspace and get_1dspace_from_list, 
        respectively.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        A three element tuple. 
        
        The first is a list whose i'th element specifies the number of distinct values of the i'th 
        parameter we consider (this is the length of the i'th element of "paramRanges" below).

        The second is a a tuple of k numpy ndarrays (where k = len(param_list)), the i'th one of 
        which is a k-dimensional array with shape (N0, ... , N{k - 1}), where Ni = 
        param_list[i].size whose i(0), ... , i(k - 1) element specifies the value of the i'th 
        parameter in the i(0), ... , i(k - 1)'th unique combination of parameter values.

        The third one is a 2d array of parameter values. It has shape (M, k), where 
        M = \prod_{i = 0}^{k - 1} param_list[i].size. 
        """

        # Set up arrays to hold the parameter values + number of parameter values for each 
        # parameter.
        paramRanges : np.ndarray    = []
        gridSizes   : list[int]     = []

        # Cycle through the parameters        
        for param in param_list:
            # Fetch the set of possible parameter values (paramRange) + the size of this set (Nx)
            Nx, paramRange  = getParam1DSpace[param['test_space_type']](param)

            # Add Nx, ParamRange to their corresponding lists
            gridSizes      += [Nx]
            paramRanges    += [paramRange]

        # Now that we have the set of parameter values for each parameter, turn it into a grid.
        mesh_grids : tuple[np.ndarray] = self.createHyperMeshGrid(paramRanges)

        # All done. Return a list specifying the number of possible values for each parameter, the
        # grids of parameter values, and the flattened/2d version of it. 
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
    


    def createHyperMeshGrid(self, param_ranges : list[np.ndarray]) -> tuple[np.ndarray]:
        '''
        This function generates arrays of parameter values. Specifically, if there are k 
        parameters (param_ranges has k elements), then we return k k-d arrays, the i'th one of 
        which is a k-dimensional array whose i(0), ... , i(k - 1) element specifies the value of 
        the i'th variable in the i(0), ... , i(k - 1)'th unique combination of parameter values.


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
        k parameters and if param_range[i].size = Ni, then the j'th return array has shape 
        (N0, ... , N{k - 1}) and the i(0), ... , i(k - 1) element of this array houses the i(j)'th 
        value of the j'th parameter.

        Thus, if there are k parameters, the returned tuple has k elements, each one of 
        which is an array of shape (N0, ... , N{k - 1}).
        '''

        # Fetch the ranges, add them to a tuple (this is what the meshgrid function needs).
        args : tuple[np.ndarray] = ()
        for paramRange in param_ranges:
            args += (paramRange,)

        # Use numpy's meshgrid function to generate the grids of parameter values.
        paramSpaces : tuple[np.ndarray] = np.meshgrid(*args, indexing='ij')

        # All done!
        return paramSpaces
    


    def createHyperGridSpace(self, mesh_grids : tuple[np.ndarray]) -> np.ndarray:
        '''
        Flattens the mesh_grid numpy.ndarray objects returned by createHyperMeshGrid and combines 
        them into a single 2d array of shape (grid size, number of parameters) (see below).


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        mesh_grids: tuple of numpy nd arrays, corresponding to each parameter. This should ALWAYS
        be the output of the "CreateHyperMeshGrid" function. See the return section of that 
        function's docstring for details.
    
        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        The param_grid. This is a 2d numpy.ndarray object of shape (grid size, number of 
        parameters). If each element of mesh_grids is a numpy.ndarray object of shape (N(1), ... , 
        N(k)) (k parameters), then (grid size) = N(1)*N(2)*...*N(k) and (number of parameters) = k.
        '''

        # For each parameter, we flatten its mesh_grid into a 1d array (of length (grid size)). We
        # horizontally stack these flattened grids to get the final param_grid array.
        param_grid : np.ndarray = None
        for k, paramSpace in enumerate(mesh_grids):
            # Special treatment for the first parameter to initialize param_grid
            if (k == 0):
                param_grid : np.ndarray = paramSpace.reshape(-1, 1)
                continue

            # Flatten the new mesh grid and append it to the grid.
            param_grid : np.ndarray = np.hstack((param_grid, paramSpace.reshape(-1, 1)))

        # All done!
        return param_grid
    


    def appendTrainSpace(self, param : np.ndarray) -> None:
        """
        Adds a new parameter to self's train space attribute.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param: A 1d numpy ndarray object. It should have shape (n_param) and should hold a 
        parameter value that we want to add to the training set.



        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Make sure param has n_param components/can be appended to the set of training parameters.
        assert(self.train_space.shape[1] == param.size)

        # Add the new parameter to the training space by appending it as a new row to 
        # self.train_space
        self.train_space    : np.ndarray    = np.vstack((self.train_space, param))
        
        # All done!
        return
    


    def export(self) -> dict:
        """
        This function packages the testing/training examples into a dictionary, which it returns.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        None!

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        A dictionary with 4 keys. Below is a list of the keys with a short description of each 
        corresponding value. 
            train_space: self.train_space, a 2d array of shape (n_train, n_param) whose i,j element 
            holds the value of the j'th parameter in the i'th training case.

            test_space: self.test_space, a 2d array of shape (n_test, n_param) whose i,j element 
            holds the value of the j'th parameter in the i'th testing case.

            test_grid_sizes: A list whose i'th element specifies how many distinct parameter values
            we use for the i'th parameter. 

            test_meshgrid: a tuple of n_param numpy.ndarray array objects whose i'th element is a
            n_param-dimensional array whose i(1), i(2), ... , i(n_param) element holds the value of 
            the i'th parameter in the i(1), ... , i(n_param) combination of parameter values in the 
            testing test. 

            n_init: The number of combinations of training parameters in the training set.     
        """

        # Build the dictionary
        dict_ = {'train_space'      : self.train_space,
                 'test_space'       : self.test_space,
                 'test_grid_sizes'  : self.test_grid_sizes,
                 'test_meshgrid'    : self.test_meshgrid,
                 'n_init'           : self.n_init}
        
        # All done!
        return dict_
    


    def load(self, dict_ : dict) -> None:
        """
        This function builds a parameter space object from a dictionary. This dictionary should 
        be one that was returned by the export method, or a loaded copy of a dictionary that was 
        returned by the export method. 


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        dict_: This should be a dictionary with the following keys: 
            - train_space
            - test_space
            - test_grid_sizes
            - test_meshgrid
            - n_init
        This dictionary should have been returned by the export method. We use the values in this 
        dictionary to set up self.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Extract information from the dictionary.
        self.train_space        : np.ndarray        = dict_['train_space']
        self.test_space         : np.ndarray        = dict_['test_space']
        self.test_grid_sizes    : list[int]         = dict_['test_grid_sizes']
        self.test_meshgrid      : tuple[np.ndarray] = dict_['test_meshgrid']

        # Run checks
        assert(self.n_init                  == dict_['n_init'])
        assert(self.train_space.shape[1]    == self.n_param)
        assert(self.test_space.shape[1]     == self.n_param)

        # All done!
        return
