import numpy as np
from scipy.spatial import Delaunay
from .inputs import InputParser

def get_1dspace_from_list(config):
    Nx = len(config['list'])
    paramRange = np.array(config['list'])
    return Nx, paramRange

def create_uniform_1dspace(config):
    Nx = config['sample_size']
    minval = config['min']
    maxval = config['max']
    if (config['log_scale']):
        paramRange = np.exp(np.linspace(np.log(minval), np.log(maxval), Nx))
    else:
        paramRange = np.linspace(minval, maxval, Nx)
    return Nx, paramRange

getParam1DSpace = {'list': get_1dspace_from_list,
                   'uniform': create_uniform_1dspace}

class ParameterSpace:
    param_list = []
    param_name = []
    n_param = 0
    train_space = None
    test_space = None
    n_init = 0
    test_grid_sizes = []
    test_meshgrid = None

    def __init__(self, config):
        assert('parameter_space' in config)
        parser = InputParser(config['parameter_space'], name="param_space_input")

        self.param_list = parser.getInput(['parameters'], datatype=list)
        self.n_param = len(self.param_list)

        self.param_name = []
        for param in self.param_list:
            self.param_name += [param['name']]

        test_space_type = parser.getInput(['test_space', 'type'], datatype=str)
        if (test_space_type == 'grid'):
            self.train_space = self.createInitialTrainSpace(self.param_list)
            self.n_init = self.train_space.shape[0]

            self.test_grid_sizes, self.test_meshgrid, self.test_space = self.createTestGridSpace(self.param_list)
        if (test_space_type == 'hull'):
            assert self.n_param >=2, 'Must have at least 2 parameters if test_space is \'hull\' '
            self.train_space = self.createInitialTrainSpaceForHull(self.param_list)
            self.n_init = self.train_space.shape[0]

            self.test_grid_sizes, self.test_meshgrid, self.test_space = self.createTestHullSpace(self.param_list)

        return
    
    def n_train(self):
        return self.train_space.shape[0]
    
    def n_test(self):
        return self.test_space.shape[0]
    
    def createInitialTrainSpace(self, param_list):
        paramRanges = []

        for param in param_list:
            minval = param['min']
            maxval = param['max']
            paramRanges += [np.array([minval, maxval])]

        mesh_grids = self.createHyperMeshGrid(paramRanges)
        return self.createHyperGridSpace(mesh_grids)
    
    def createInitialTrainSpaceForHull(self, param_list):

        """
        If test_space is 'hull', then the provided training parameters must be 
        points on the exterior of our training space. This function concatenates the provided 
        training points into a 2D array.

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        param_list: A list of parameter dictionaries. Each entry should be a dictionary with the 
        following keys:
            - name
            - min
            - max
            - sample_size
            - list
            - log_scale false
            
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        A 2d array of shape (d, k), where d is the number of points provided on the exterior of
        the training space and k is the number of parameters (k == len(param_list)).
        """

        #A list we will use to store all of the provided training points.
        paramRanges = []

        for k, param in enumerate(param_list):

            # Fetch the training points associated with each parameter which are given by a list.
            _, paramRange = getParam1DSpace['list'](param)
            # Store the training points into the list.
            paramRanges += [paramRange]

            if k > 0:
                assert (len(paramRanges[k])==len(paramRanges[k - 1])), (f'Training parameters {k} and {k-1} have '
                                                            'different lengths. All training parameters '
                                                            'must have same length when test_space is \'hull\'.')
                
        
        # Stack all the provided training points into an array.
        mesh_grids = np.vstack((paramRanges)).T
        return mesh_grids
    
    def createTestGridSpace(self, param_list):
        paramRanges = []
        gridSizes = []

        for param in param_list:
            Nx, paramRange = getParam1DSpace[param['test_space_type']](param)
            gridSizes += [Nx]
            paramRanges += [paramRange]

        mesh_grids = self.createHyperMeshGrid(paramRanges)
        return gridSizes, mesh_grids, self.createHyperGridSpace(mesh_grids)
    
    def createTestGridSpaceForHull(self, param_list):
        """
        This function sets up an initial grid for the testing parameters when the test_space is 
        'hull'. Here, we form a uniform grid over the given training parameters based on the 
        provided min and max values of each parameter and specified number of samples. The function
        'createTestSpaceFromHull' will later be used to keep testing point which are in the
        convex hull of training points.

        This function is similar to the function 'createTestGridSpace', except we do not specify 
        the 'test_space_type' value for any parameter.

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        param_list: A list of parameter dictionaries. Each entry should be a dictionary with the 
        following keys:
            - name
            - min
            - max
            - sample_size
            - list
            - log_scale false
            
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

        paramRanges = []
        gridSizes = []

        for param in param_list:
            Nx, paramRange = getParam1DSpace['uniform'](param)
            gridSizes += [Nx]
            paramRanges += [paramRange]

        mesh_grids = self.createHyperMeshGrid(paramRanges)
        return gridSizes, mesh_grids, self.createHyperGridSpace(mesh_grids)
    
    def createTestHullSpace(self, param_list):
        """
        This function sets up an initial grid for the testing parameters when the test_space is 
        'hull'. Here, we form a uniform grid over the giving training parameters based on the 
        provided min and max values of each parameter and specified number of samples. The function
        'createTestSpaceFromHull' will later be used to only keep values of this grid which are in
        the convex hull of our training parameters.

        This function is similar to the function 'createTestGridSpace', except we do not specify 
        the 'test_space_type' value for any parameter.

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        param_list: A list of parameter dictionaries. Each entry should be a dictionary with the 
        following keys:
            - name
            - min
            - max
            - sample_size
            - list
            - log_scale false
            
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

        The third one is a 2d array of parameter values. It has shape (M, k), where M is the
        number of testing points after removing points outside the convex hull of training 
        parameters. 
        """

        # Get the initial uniform grid over the training parameters
        gridSizes, mesh_grids, test_space = self.createTestGridSpaceForHull(param_list)

        # Mesh the training space. This will be slow in higher dimensions
        cloud = Delaunay(self.train_space)
        # Determine which test points are contained in the convex hull of training points
        mask = cloud.find_simplex(test_space)>=0
        # Only keep testing points in the convex hull of training points
        test_space = test_space[mask]

        return gridSizes, mesh_grids, test_space
    
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
    
    def createHyperMeshGrid(self, param_ranges):
        '''
            param_ranges: list of numpy 1d arrays, each corresponding to 1d parameter grid space.
                          The list size is equal to the number of parameters.
            
            Output: paramSpaces
                - tuple of numpy nd arrays, corresponding to each parameter.
                  Dimension of the array equals to the number of parameters
        '''
        args = ()
        for paramRange in param_ranges:
            args += (paramRange,)

        paramSpaces = np.meshgrid(*args, indexing='ij')
        return paramSpaces
    
    def createHyperGridSpace(self, mesh_grids):
        '''
            mesh_grids: tuple of numpy nd arrays, corresponding to each parameter.
                        Dimension of the array equals to the number of parameters
            
            Output: param_grid
                - numpy 2d array of size (grid size x number of parameters).

                grid size is the size of a numpy nd array.
        '''
        param_grid = None
        for k, paramSpace in enumerate(mesh_grids):
            if (k == 0):
                param_grid = paramSpace.reshape(-1, 1)
                continue

            param_grid = np.hstack((param_grid, paramSpace.reshape(-1, 1)))

        return param_grid
    
    def appendTrainSpace(self, param):
        assert(self.train_space.shape[1] == param.size)

        self.train_space = np.vstack((self.train_space, param))
        return
    
    def export(self):
        dict_ = {'train_space': self.train_space,
                 'test_space': self.test_space,
                 'test_grid_sizes': self.test_grid_sizes,
                 'test_meshgrid': self.test_meshgrid,
                 'n_init': self.n_init}
        return dict_
    
    def load(self, dict_):
        self.train_space = dict_['train_space']
        self.test_space = dict_['test_space']
        self.test_grid_sizes = dict_['test_grid_sizes']
        self.test_meshgrid = dict_['test_meshgrid']

        assert(self.n_init == dict_['n_init'])
        assert(self.train_space.shape[1] == self.n_param)
        assert(self.test_space.shape[1] == self.n_param)
        return
