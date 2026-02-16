import numpy as np
import torch
from scipy.integrate import odeint
from . import LatentDynamics
from ..inputs import InputParser
from ..fd import FDdict
import importlib

# Defining a function for converting the custom functions from .yml (in str format) to modules

def get_function_from_string(func_str):
    # Split the string into module and function
    module_name, func_name = func_str.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)

class SINDy(LatentDynamics):
    fd_type = ''
    fd = None
    fd_oper = None

    def __init__(self, dim, high_order_terms, trig_functions, nt, config):
        super().__init__(dim, nt)
        print('Higher order sindy')
        #TODO(kevin): generalize for high-order dynamics! Updating this! Only works for first order terms!
        #self.ncoefs = self.dim * (self.dim + 1)

        # Logic: total terms in candidate set: (higher_order_terms + first order terms + constants) * reduced dim size
        self.high_order_terms = high_order_terms
        self.trig_functions = trig_functions

        self.ncoefs = ((len(self.trig_functions) + self.high_order_terms)*self.dim + self.dim + 1) * self.dim

        assert('sindy' in config)
        parser = InputParser(config['sindy'], name='sindy_input')

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
            loss_sindy, loss_coef = 0.0, 0.0

            for i in range(n_train):

                # Iterate over individual training cases!
                result = self.calibrate(Z[i], dt, compute_loss, numpy)

                # Compute the loss and other terms based on the flag!
                if (compute_loss):
                    coefs[i] = result[0]
                    loss_sindy += result[1]
                    loss_coef += result[2]
                else:
                    coefs[i] = result
            
            if (compute_loss):
                return coefs, loss_sindy, loss_coef
            else:
                return coefs

        ''' evaluate for one train case '''
        assert(Z.dim() == 2)
        dZdt = self.compute_time_derivative(Z, dt)
        time_dim, space_dim = dZdt.shape
        
        # Build up the library functions: ones + the latent terms. 
        Z_i = torch.cat([torch.ones(time_dim, 1), Z], dim = 1)

        # Adding higher order terms! Running a for loop based on how many higher order terms you want to add!
        for i in range(self.high_order_terms):

            # Append to the candidtate library the higher order expressions
            Z_i = torch.cat([Z_i, Z**(i+2)], dim = 1)

        # Adding trignometric functions 
        for i in self.trig_functions:
            Z_i = torch.cat([Z_i, get_function_from_string(i)(Z)], dim = 1)

        # Find the coefficients using least squares
        coefs = torch.linalg.lstsq(Z_i, dZdt).solution
        # Returning different stuff!
        if (compute_loss):
            loss_sindy = self.MSE(dZdt, Z_i @ coefs)
            # NOTE(kevin): by default, this will be L1 norm.
            loss_coef = torch.norm(coefs, self.coef_norm_order)

        # output of lstsq is not contiguous in memory.
        coefs = coefs.detach().flatten()
        if (numpy):
            coefs = coefs.numpy()

        if (compute_loss):
            return coefs, loss_sindy, loss_coef
        else:
            return coefs

    def compute_time_derivative(self, Z, Dt):

        '''

        Builds the SINDy dataset, assuming only linear terms in the SINDy dataset. The time derivatives are computed through
        finite difference.

        Z is the encoder output (2D tensor), with shape [time_dim, space_dim]
        Dt is the size of timestep (assumed to be a uniform scalar)

        The output dZdt is a 2D tensor with the same shape of Z.

        '''
        return 1. / Dt * torch.sparse.mm(self.fd_oper, Z)

    def simulate(self, coefs, z0, t_grid):

        '''

        Integrates each system of ODEs corresponding to each training points, given the initial condition Z0 = encoder(U0)

        '''
        # copy is inevitable for numpy==1.26. removed copy=False temporarily.

        # Updating the coefficients reshape based on higher order terms logic!
        # c_i = coefs.reshape([self.dim+1, self.dim]).T

        c_i = coefs.reshape([(self.high_order_terms + len(self.trig_functions)) * self.dim + self.dim + 1, self.dim]).T

        # Update the integrate function to account for the additional initial conditions when using higher order terms!
        # dzdt = lambda z, t : c_i[:, 1:] @ z + c_i[:, 0]

        def dzdt(z,t):
            
            # Making a copy of the z
            z_new = z

            # Add the higher order terms!
            for i in range(self.high_order_terms):

                # Get the higher order terms if any
                new_terms = np.power(z,i+2)

                # Stack the initial conditions!
                z_new = np.hstack((z_new,new_terms))

            # Add the random funtions to the candidate library!
            for i in self.trig_functions:
                
                # Get the trig terms!
                new_terms = get_function_from_string(i)(torch.from_numpy(z))

                # Stack the initial conditions!
                z_new = np.hstack((z_new, new_terms.detach().cpu().numpy()))

            # Return the final initial conditions for the candidtate library!
            return c_i[:,1:] @ z_new + c_i[:,0]
        
        # Integrate!
        Z_i = odeint(dzdt, z0, t_grid)

        return Z_i
    
    def export(self):
        param_dict = super().export()
        param_dict['fd_type'] = self.fd_type
        param_dict['coef_norm_order'] = self.coef_norm_order
        return param_dict
