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

def get_1dspace_for_exterior(config):
    Nx = config['sample_size']
    paramRange = np.array(config['list'])
    return Nx, paramRange

getParam1DSpace = {'list': get_1dspace_from_list,
                   'uniform': create_uniform_1dspace,
                   'exterior': get_1dspace_for_exterior}

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

            self.test_grid_sizes, self.test_meshgrid, self.test_space = self.createTestSpaceFromHull(self.param_list)

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
        '''
        If test_space is 'hull', then the provided training parameters must be 
        points on the exterior of our training space. So, we form the provided points
        into an array.
        '''

        paramRanges = []

        k = 0
        for param in param_list:
            assert (param['test_space_type'] == 'exterior'), ('test_space_type for all parameters must '
                                                            'be \'exterior\' when test_space is \'hull\'. ')
            
            _, paramRange = getParam1DSpace[param['test_space_type']](param)
            paramRanges += [paramRange]

            if k > 0:
                assert (len(paramRanges[k])==len(paramRanges[k - 1])), (f'Training parameters {k} and {k-1} have '
                                                            'different lengths. All training parameters '
                                                            'must have same length when test_space is \'hull\'.')
            k = k + 1


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
        '''
        This is similar to createTestGridSpace, but with some different variables.
        We take the min/max value of each parameter, and create a uniform rectangular grid
        over the parameter space with 'sample_size' points in each dimension. 
        '''

        paramRanges = []
        gridSizes = []

        for param in param_list:
            Nx, _ = getParam1DSpace[param['test_space_type']](param)
            minval = param['min']
            maxval = param['max']
            gridSizes += [Nx]
            paramRanges += [np.linspace(minval, maxval, Nx)]

        mesh_grids = self.createHyperMeshGrid(paramRanges)
        return gridSizes, mesh_grids, self.createHyperGridSpace(mesh_grids)
    
    def createTestSpaceFromHull(self, param_list):
        #get the initial grid over the parameters
        gridSizes, mesh_grids, test_space = self.createTestGridSpaceForHull(self.param_list)


        #mesh training space. This will be slow in higher dimensions
        cloud = Delaunay(self.train_space)
        #Determine if each point is in/out of convex Hull
        mask = cloud.find_simplex(test_space)>=0
        #Only keep points in convex Hull
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
