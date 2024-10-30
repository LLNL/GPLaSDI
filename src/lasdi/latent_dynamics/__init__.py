# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import numpy as np
import torch



# -------------------------------------------------------------------------------------------------
# LatentDynamics base class
# -------------------------------------------------------------------------------------------------

class LatentDynamics:
    # Class variables
    dim     : int           = -1        # Dimensionality of the latent space
    nt      : int           = -1        # Number of time steps when solving the latent dynamics
    ncoefs  : int           = -1        # Number of coefficients in the latent space dynamics

    # TODO(kevin): do we want to store coefficients as an instance variable?
    coefs   : torch.Tensor  = torch.Tensor([])



    def __init__(self, dim_ : int, nt_ : int) -> None:
        """
        Initializes a LatentDynamics object. Each LatentDynamics object needs to have a 
        dimensionality (dim), a number of time steps, a model for the latent space dynamics, and 
        set of coefficients for that model. The model should describe a set of ODEs in 
        \mathbb{R}^{dim}. We 


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        dim_: The number of dimensions in the latent space, where the latent dynamics takes place.

        nt_: The number of time steps we want to generate when solving (numerically) the latent 
        space dynamics.
        """

        # Set class variables.
        self.dim    : int   = dim_
        self.nt     : int   = nt_

        # There must be at least one latent dimension and there must be at least 1 time step.
        assert(self.dim > 0)
        assert(self.nt > 0)

        # All done!
        return
    


    def calibrate(self, 
                  Z             : torch.Tensor, 
                  dt            : int, 
                  compute_loss  : bool          = True, 
                  numpy         : bool          = False) -> np.ndarray:
        '''
            calibrate coefficients of the latent dynamics and compute loss
            for one given time series Z.

            Z is the encoder output (2D/3D tensor), with shape [time_dim, space_dim] or [n_train, time_dim, space_dim]
        '''
        raise RuntimeError('Abstract function LatentDynamics.calibrate!')
        if (compute_loss):
            return coefs, loss
        else:
            return coefs
    


    def simulate(self, coefs, z0, t_grid):
        '''
            time-integrate with one initial condition z0 at time points t_grid,
            for one given set of coefficients
        '''
        raise RuntimeError('Abstract function LatentDynamics.simulate!')
        return zhist
    


    def sample(self, coefs_sample, z0_sample, t_grid):
        '''
            Sample time series for given sample initial conditions and coefficients.
        '''
        assert(len(coefs_sample) == len(z0_sample))

        for i in range(len(coefs_sample)):
            Z_i = self.simulate(coefs_sample[i], z0_sample[i], t_grid)
            Z_i = Z_i.reshape(1, Z_i.shape[0], Z_i.shape[1])

            if (i == 0):
                Z_simulated = Z_i
            else:
                Z_simulated = np.concatenate((Z_simulated, Z_i), axis = 0)

        return Z_simulated
    


    def export(self):
        param_dict = {'dim': self.dim, 'ncoefs': self.ncoefs}
        return param_dict
    

    
    # SINDy does not need to load parameters.
    # Other latent dynamics might need to.
    def load(self, dict_):
        assert(self.dim == dict_['dim'])
        assert(self.ncoefs == dict_['ncoefs'])
        return
    