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
        \mathbb{R}^{dim}. These ODEs should contain a set of unknown coefficients. We learn those 
        coefficients using the calibrate function. Once we have learned the coefficients, we can 
        solve the corresponding set of ODEs forward in time using the simulate function.


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
        """
        The user must implement this class on any latent dynamics sub-class. Each latent dynamics 
        object should parameterize a model for the dynamics in the latent space. Thus, to specify
        the latent dynamics, we need to find a set of coefficients that parameterize the equations
        in the latent dynamics model. The calibrate function is supposed to do that. 

        Specifically, this function should take in a sequence (or sequences) of latent states, a
        time step, dt, which specifies the time step between successive terms in the sequence(s) of
        latent states, and some optional booleans which control what information we return. 

        The function should always return at least one argument: a numpy.ndarray object holding the
        optimal coefficients in the latent dynamics model using the data contained in Z. 


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Z: A 2d or 3d tensor. If Z is a 2d tensor, then it has shape (Nt, Nz), where Nt specifies 
        the number of time steps in each sequence of latent states and Nz is the dimension of the 
        latent space. In this case, the i,j entry of Z holds the j'th component of the latent state 
        at the time t_0 + i*dt. If it is a 3d tensor, then it has shape (Np, Nt, Nz). In this case, 
        we assume there at Np different combinations of parameter values. The i, j, k entry of Z in 
        this case holds the k'th component of the latent encoding at time t_0 + j*dt when we use 
        he i'th combination of parameter values. 

        dt: The time step between time steps. See the description of the "Z" argument. 

        compute_loss: A boolean which, if True, this function should return the coefficients and 
        additional losses based on the set of coefficients we learn. If False, this function should
        only return the optimal coefficients for the latent dynamics model using the data in Z. 

        numpy: A boolean. If True, this function should return the coefficient matrix as a 
        numpy.ndarray object. If False, this function should return it as a torch.Tensor object.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
     
        A tensor or ndarray (depending on the value of the "numpy" argument) holding the optimal 
        coefficients for the latent space dynamics given the data stored in Z. If Z is 2d, then
        the returned tensor will only contain one set of coefficients. If Z is 3d, with a leading 
        dimension size of Np (number of combinations of parameter values) then we will return 
        an array/tensor with a leading dimension of size Np whose i'th entry holds the coefficients
        for the sequence of latent states stored in Z[:, ...].
        """

        raise RuntimeError('Abstract function LatentDynamics.calibrate!')
        if (compute_loss):
            return coefs, loss
        else:
            return coefs
    


    def simulate(self, coefs : np.ndarray, z0 : np.ndarray, t_grid : np.ndarray) -> np.ndarray:
        """
        Time integrates the latent dynamics when it uses the coefficients specified in coefs and 
        starts from the (single) initial condition in z0.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        coefs: A one dimensional numpy.ndarray object holding the coefficients we want to use 
        to solve the latent dynamics forward in time. 

        z0: A numpy ndarray object of shape nz representing the initial condition for the latent 
        dynamics. Thus, the i'th component of this array should hold the i'th component of the 
        latent dynamics initial condition.

        t_grid: A 1d numpy ndarray object whose i'th entry holds the value of the i'th time value 
        where we want to compute the latent solution. The elements of this array should be in 
        ascending order.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------        
        
        A 2d numpy.ndarray object holding the solution to the latent dynamics at the time values 
        specified in t_grid when we use the coefficients in coefs to characterize the latent 
        dynamics model. Specifically, this is a 2d array of shape (nt, nz), where nt is the 
        number of time steps (size of t_grid) and nz is the latent space dimension (self.dim). 
        Thus, the i,j element of this matrix holds the j'th component of the latent solution at 
        the time stored in the i'th element of t_grid. 
        """

        raise RuntimeError('Abstract function LatentDynamics.simulate!')
        return zhist
    


    def sample(self, coefs_sample : np.ndarray, z0_sample : np.ndarray, t_grid : np.ndarray) -> np.ndarray:
        """
        simulate's the latent dynamics for a set of coefficients/initial conditions.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        coefs_sample: A numpy.ndarray object whose leading dimension has size ns (the number of 
        sets of coefficients/initial conditions/simulations we run).

        z0_sample: A 2d numpy.ndarray object of shape (ns, nz) (where ns is the number of samples
        and nz is the dimensionality of the latent space). The i,j entry of z0_sample should hold 
        the j'th component of the i'th initial condition.

        t_grid: A 1d numpy ndarray object whose i'th entry holds the value of the i'th time value 
        where we want to compute each latent solution. The elements of this array should be in 
        ascending order. We use the same array for each set of coefficients.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------


        A 3d numpy ndarray object of shape (ns, nt, nz), where ns = the number of samples (the 
        leading dimension of z0_sample and coefs_sample), nt = the number of time steps (size of 
        t_grid) and nz is the dimension of the latent space. The i, j, k element of this array 
        holds the k'th component of the solution of the latent dynamics at the j'th time step (j'th
        element of t_grid) when we use the i'th set of coefficients/initial conditions. 
        """

        # There needs to be as many initial conditions as sets of coefficients.
        assert(len(coefs_sample) == len(z0_sample))

        # Cycle through the set of coefficients
        for i in range(len(coefs_sample)):
            # Simulate the latent dynamics when we use the i'th set of coefficients + ICs
            Z_i : np.ndarray = self.simulate(coefs_sample[i], z0_sample[i], t_grid)

            # Append a leading dimension of size 1.
            Z_i = Z_i.reshape(1, Z_i.shape[0], Z_i.shape[1])

            # Append the latest trajectory onto the Z_simulated array.
            if (i == 0):
                Z_simulated = Z_i
            else:
                Z_simulated = np.concatenate((Z_simulated, Z_i), axis = 0)

        # All done!
        return Z_simulated
    


    def export(self) -> dict:
        param_dict = {'dim': self.dim, 'ncoefs': self.ncoefs}
        return param_dict
    

    
    # SINDy does not need to load parameters.
    # Other latent dynamics might need to.
    def load(self, dict_ : dict) -> None:
        assert(self.dim == dict_['dim'])
        assert(self.ncoefs == dict_['ncoefs'])
        return
    