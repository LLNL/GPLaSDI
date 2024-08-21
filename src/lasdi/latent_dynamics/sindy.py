import numpy as np
import torch
from scipy.integrate import odeint
from . import LatentDynamics
from ..inputs import InputParser
from ..fd import SBP12, SBP24, SBP36, SBP48

FDdict = {'sbp12': SBP12(),
          'sbp24': SBP24(),
          'sbp36': SBP36(),
          'sbp48': SBP48()}

class SINDy(LatentDynamics):
    fd_type = ''
    fd = None

    def __init__(self, dim, config):
        super().__init__(dim)

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

        # NOTE(kevin): by default, this will be L1 norm.
        self.coef_norm_order = parser.getInput(['coef_norm_order'], fallback=1)

        # TODO(kevin): other loss functions
        self.MSE = torch.nn.MSELoss()

        return
    
    def calibrate(self, Z, dt, compute_loss=True, numpy=False):

        if (Z.dim() == 3):
            coefs = []
            loss_sindy, loss_coef = 0.0, 0.0

            n_train = Z.size(0)
            for i in range(n_train):
                result = self.calibrate(Z[i], dt, compute_loss, numpy)
                if (compute_loss):
                    coefs += result[0]
                    loss_sindy += result[1]
                    loss_coef += result[2]
                else:
                    coefs += result
            
            if (compute_loss):
                return coefs, loss_sindy, loss_coef
            else:
                return coefs

        assert(Z.dim() == 2)
        dZdt = self.compute_time_derivative(Z, dt)
        time_dim, space_dim = dZdt.shape

        Z_i = torch.cat([torch.ones(time_dim, 1), Z], dim = 1)

        if (numpy):
            Z_i = Z_i.detach()
            dZdt = dZdt.detach()

        coefs = torch.linalg.lstsq(Z_i, dZdt).solution

        if (compute_loss):
            loss_sindy = self.MSE(dZdt, Z_i @ coefs)
            # NOTE(kevin): by default, this will be L1 norm.
            loss_coef = torch.norm(coefs, self.coef_norm_order)

        if (numpy):
            coefs = coefs.detach().numpy()

        if (compute_loss):
            return [coefs], loss_sindy, loss_coef
        else:
            return [coefs]

    def compute_time_derivative(self, Z, Dt):

        '''

        Builds the SINDy dataset, assuming only linear terms in the SINDy dataset. The time derivatives are computed through
        finite difference.

        Z is the encoder output (2D tensor), with shape [time_dim, space_dim]
        Dt is the size of timestep (assumed to be a uniform scalar)

        The output dZdt is a 2D tensor with the same shape of Z.

        '''

        oper, _, _ = self.fd.getOperators(Z.size(0))
        return 1. / Dt * torch.sparse.mm(oper, Z)

    def simulate(self, coefs, z0, t_grid):

        '''

        Integrates each system of ODEs corresponding to each training points, given the initial condition Z0 = encoder(U0)

        '''

        c_i = coefs.T
        dzdt = lambda z, t : c_i[:, 1:] @ z + c_i[:, 0]

        Z_i = odeint(dzdt, z0, t_grid)

        return Z_i
    
    def export(self):
        param_dict = super().export()
        param_dict['fd_type'] = self.fd_type
        param_dict['coef_norm_order'] = self.coef_norm_order
        return param_dict

def simulate_uncertain_sindy(gp_dictionnary, param, n_samples, z0, t_grid, sindy_coef, n_coef, coef_samples = None):

    '''

    Integrates each ODE samples for a given parameter.

    '''

    if coef_samples is None:
        from .interp import interpolate_coef_matrix
        coef_samples = interpolate_coef_matrix(gp_dictionnary, param, n_samples, n_coef, sindy_coef)

    Z0 = [z0 for _ in range(n_samples)]
    Z = simulate_sindy(coef_samples, Z0, t_grid)

    return Z

def simulate_interpolated_sindy(param_grid, Z0, t_grid, n_samples, Dt, Z, param_train, fd_type):

    '''

    Integrates each ODE samples for each parameter of the parameter grid.
    Z_simulated is a list of length param_grid.shape[0], where each term is a 3D tensor of the form [n_samples, time_dim, n_z]

    '''

    from .interp import build_interpolation_data, fit_gps, interpolate_coef_matrix

    dZdt = compute_time_derivative(Z, Dt, fd_type)
    sindy_coef = solve_sindy(dZdt, Z)
    interpolation_data = build_interpolation_data(sindy_coef, param_train)
    gp_dictionnary = fit_gps(interpolation_data)
    n_coef = interpolation_data['n_coef']

    coef_samples = [interpolate_coef_matrix(gp_dictionnary, param_grid[i, :], n_samples, n_coef, sindy_coef) for i in range(param_grid.shape[0])]

    Z_simulated = [simulate_uncertain_sindy(gp_dictionnary, param_grid[i, 0], n_samples, Z0[i], t_grid, sindy_coef, n_coef, coef_samples[i]) for i in range(param_grid.shape[0])]

    return Z_simulated, gp_dictionnary, interpolation_data, sindy_coef, n_coef, coef_samples