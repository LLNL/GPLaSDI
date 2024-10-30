# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  numpy               as      np
import  torch
from    scipy.integrate     import  odeint

from    .                   import  LatentDynamics
from    ..inputs            import  InputParser
from    ..fd                import  FDdict



# -------------------------------------------------------------------------------------------------
# SINDy class
# -------------------------------------------------------------------------------------------------

class SINDy(LatentDynamics):
    fd_type     = ''
    fd          = None
    fd_oper     = None



    def __init__(self, 
                 dim        : int, 
                 nt         : int, 
                 config     : dict) -> None:
        """
        Initializes a SINDy object. This is a subclass of the LatentDynamics class which uses the 
        SINDy algorithm as its model for the ODE governing the latent state. Specifically, we 
        assume there is a library of functions, f_1(z), ... , f_N(z), each one of which is a 
        monomial of the components of the latent space, z, and a set of coefficients c_{i,j}, 
        i = 1, 2, ... , dim and j = 1, 2, ... , N such that
            z_i'(t) = \sum_{j = 1}^{N} c_{i,j} f_j(z)
        In this case, we assume that f_1, ... , f_N consists of the set of order <= 1 monomials. 
        That is, f_1(z), ... , f_N(z) = 1, z_1, ... , z_{dim}.
            

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        dim: The number of dimensions in the latent space, where the latent dynamics takes place.

        nt: The number of time steps we want to generate when solving (numerically) the latent 
        space dynamics.

        config: A dictionary housing the settings we need to set up a SINDy object. Specifically, 
        this dictionary should have a key called "sindy" whose corresponding value is another 
        dictionary with the following two keys:
            - fd_type: A string specifying which finite-difference scheme we should use when
            approximating the time derivative of the solution to the latent dynamics at a 
            specific time. Currently, the following options are allowed:
                - 'sbp12': summation-by-parts 1st/2nd (boundary/interior) order operator
                - 'sbp24': summation-by-parts 2nd/4th order operator
                - 'sbp36': summation-by-parts 3rd/6th order operator
                - 'sbp48': summation-by-parts 4th/8th order operator
            - coef_norm_order: A string specifying which norm we want to use when computing
            the coefficient loss.
        """

        # Run the base class initializer. The only thing this does is set the dim and nt 
        # attributes.
        super().__init__(dim, nt)

        # We only allow library terms of order <= 1. If we let z(t) \in \mathbb{R}^{dim} denote the 
        # latent state at some time, t, then the possible library terms are 1, z_1(t), ... , 
        # z_{dim}(t). Since each component function gets its own set of coefficients, there must 
        # be dim*(dim + 1) total coefficients.
        #TODO(kevin): generalize for high-order dynamics
        self.ncoefs = self.dim * (self.dim + 1)

        # Now, set up an Input parser to process the contents of the config['sindy'] dictionary. 
        assert('sindy' in config)
        parser = InputParser(config['sindy'], name = 'sindy_input')

        """
        Determine which finite difference scheme we should use to approximate the time derivative
        of the latent space dynamics. Currently, we allow the following values for "fd_type":
            - 'sbp12': summation-by-parts 1st/2nd (boundary/interior) order operator
            - 'sbp24': summation-by-parts 2nd/4th order operator
            - 'sbp36': summation-by-parts 3rd/6th order operator
            - 'sbp48': summation-by-parts 4th/8th order operator
        """
        self.fd_type    : str       = parser.getInput(['fd_type'], fallback = 'sbp12')
        self.fd         : callable  = FDdict[self.fd_type]

        # RESUME HERE 
        # RESUME HERE 
        # RESUME HERE 
        # RESUME HERE 
        # RESUME HERE 
        # RESUME HERE 
        # RESUME HERE 
        # RESUME HERE 
        # RESUME HERE 
        # RESUME HERE 
        # RESUME HERE 
        # RESUME HERE 
        # RESUME HERE 
        # RESUME HERE 
        # RESUME HERE 
        # RESUME HERE 
        # RESUME HERE 
        # RESUME HERE 
        # RESUME HERE 
        # RESUME HERE 
        # RESUME HERE 
        # RESUME HERE 
        # RESUME HERE 
        self.fd_oper, _, _          = self.fd.getOperators(self.nt)

        # NOTE(kevin): by default, this will be L1 norm.
        self.coef_norm_order = parser.getInput(['coef_norm_order'], fallback=1)

        # TODO(kevin): other loss functions
        self.MSE = torch.nn.MSELoss()

        # All done!
        return
    


    def calibrate(self, Z, dt, compute_loss=True, numpy=False):
        ''' loop over all train cases, if Z dimension is 3 '''
        if (Z.dim() == 3):
            n_train = Z.size(0)

            if (numpy):
                coefs = np.zeros([n_train, self.ncoefs])
            else:
                coefs = torch.Tensor([n_train, self.ncoefs])
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