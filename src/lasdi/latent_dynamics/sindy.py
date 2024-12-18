import numpy as np
import torch
from scipy.integrate import odeint
from . import LatentDynamics
from ..inputs import InputParser
from ..fd import FDdict

class SINDy(LatentDynamics):
    fd_type = ''
    fd = None
    fd_oper = None

    def __init__(self, dim, nt, config):
        super().__init__(dim, nt)

        #TODO(kevin): generalize for high-order dynamics
        self.ncoefs = self.dim * (self.dim + 1)

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
                result = self.calibrate(Z[i], dt, compute_loss, numpy)
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

        Z_i = torch.cat([torch.ones(time_dim, 1), Z], dim = 1)
        coefs = torch.linalg.lstsq(Z_i, dZdt).solution

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
        c_i = coefs.reshape([self.dim+1, self.dim]).T
        dzdt = lambda z, t : c_i[:, 1:] @ z + c_i[:, 0]

        Z_i = odeint(dzdt, z0, t_grid)

        return Z_i
    
    def export(self):
        param_dict = super().export()
        param_dict['fd_type'] = self.fd_type
        param_dict['coef_norm_order'] = self.coef_norm_order
        return param_dict
