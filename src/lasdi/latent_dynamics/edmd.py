import numpy as np
import torch
from scipy.integrate import odeint
from . import LatentDynamics
from ..inputs import InputParser
from ..fd import FDdict
from scipy.linalg import logm
import importlib 

def get_function_from_string(func_str):
    # Split the string into module and function
    module_name, func_name = func_str.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)

class edmd(LatentDynamics):
    fd_type = ''
    fd = None
    fd_oper = None

    def __init__(self, dim, high_order_terms, rand_functions, nt, config):
        super().__init__(dim, nt) 

        # Defining the higher order terms
        self.high_order_terms = high_order_terms
        self.rand_functions = rand_functions

        # Number of coefficients depend upon the basis functions used
        self.ncoefs = ((len(self.rand_functions) + self.high_order_terms)*self.dim + self.dim) ** 2

        assert('edmd' in config)
        parser = InputParser(config['edmd'], name='edmd_input')

        '''
            fd_type is the string that specifies finite-difference scheme for time derivative:
                - 'sbp12': summation-by-parts 1st/2nd (boundary/interior) order operator
                - 'sbp24': summation-by-parts 2nd/4th order operator
                - 'sbp36': summation-by-parts 3rd/6th order operator
                - 'sbp48': summation-by-parts 4th/8th order operator
        '''
        self.fd_type = parser.getInput(['fd_type'], fallback='sbp12')
        self.fd = FDdict[self.fd_type]
        self.fd_oper, _, _ = self.fd.getOperators(self.nt)

        # NOTE(kevin): by default, this will be L1 norm.
        self.coef_norm_order = parser.getInput(['coef_norm_order'], fallback=1)

        # TODO(kevin): other loss functions
        self.MSE = torch.nn.MSELoss()

        return
    
    def calibrate(self, Z, dt, compute_loss=True, numpy=False):
        ''' loop over all train cases, if Z dimension is 3 '''
        if (Z.dim() == 3):
            n_train = Z.size(0)

            if (numpy):
                coefs = np.zeros([n_train, self.ncoefs])
            else:
                coefs = torch.zeros([n_train, self.ncoefs])
            loss_edmd, loss_coef = 0.0, 0.0

            for i in range(n_train):
                result = self.calibrate(Z[i], dt, compute_loss, numpy)
                if (compute_loss):
                    coefs[i] = result[0]
                    loss_edmd += result[1]
                    loss_coef += result[2]
                else:
                    coefs[i] = result
            
            if (compute_loss):
                return coefs, loss_edmd, loss_coef
            else:
                return coefs

        ''' evaluate for one train case '''
        assert(Z.dim() == 2)
        
        # Creating a copy!
        Z_i = Z

        # Adding higher order terms! Running a for loop based on how many higher order terms you want to add!
        for i in range(self.high_order_terms):

            # Append to the candidtate library the higher order expressions
            Z_i = torch.cat([Z_i, Z**(i+2)], dim = 1)

        # Adding trignometric functions
        for i in self.rand_functions:
            Z_i = torch.cat([Z_i, get_function_from_string(i)(Z)], dim = 1)

        # reshaping the Z to have columns as snapshots!
        Z_i = torch.transpose(Z_i,0,1)

        # Get the Z' matrix!
        Z_plus = Z_i[:,1:]
        Z_minus = Z_i[:,0:-1]

        # Get the A operator: Using lstsq since that is more stable then pseudo inverse!
        A = (torch.linalg.lstsq(Z_minus.T,Z_plus.T).solution).T
        #A = Z_plus @ torch.linalg.pinv(Z_minus)

	# Compute the losses!
        if (compute_loss):

	    # NOTE(khushant): This loss is different from what is used in SINDy.
            loss_edmd = self.MSE(Z_plus, A @ Z_minus)

            # NOTE(kevin): by default, this will be L1 norm.
            loss_coef = torch.norm(A, self.coef_norm_order)

        # output of lstsq is not contiguous in memory.
        coefs = A.detach().flatten()
        if (numpy):
            coefs = coefs.numpy()

        if (compute_loss):
            return coefs, loss_edmd, loss_coef
        else:
            return coefs

    def simulate(self, coefs, z0, t_grid):

        '''

        Integrates each system of ODEs corresponding to each training points, given the initial condition Z0 = encoder(U0)

        '''
        # copy is inevitable for numpy==1.26. removed copy=False temporarily.
        A = coefs.reshape([(self.high_order_terms + len(self.rand_functions)) * self.dim + self.dim, (self.high_order_terms + len(self.rand_functions)) * self.dim + self.dim])

        Z_i = np.zeros((len(t_grid), self.dim))
        Z_i[0,:] = z0
	
	# Performing the integration

        for i in range(1,len(t_grid)):

            # Making a copy of the z
            z_new = Z_i[i-1,:]

            # Add the higher order terms!
            for j in range(self.high_order_terms):

                # Get the higher order terms if any
                new_terms = np.power(Z_i[i-1,:],j+2)

                # Stack the initial conditions!
                z_new = np.hstack((z_new,new_terms))

            # Add the trignometric funtions to the candidate library!
            for k in self.rand_functions:

                # Get the trig terms!
                new_terms = get_function_from_string(k)(torch.from_numpy(Z_i[i-1,:]))

                # Stack the initial conditions!
                z_new = np.hstack((z_new, new_terms.detach().cpu().numpy()))
	    
	    # Integrate and store!
            Z_i[i,:] = (A @ z_new)[:self.dim]

        return Z_i
    
    def export(self):
        param_dict = super().export()
        param_dict['fd_type'] = self.fd_type
        param_dict['coef_norm_order'] = self.coef_norm_order
        return param_dict

