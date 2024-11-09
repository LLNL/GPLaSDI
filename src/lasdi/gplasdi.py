# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import  time

import  torch
import  numpy                       as      np
from    torch.optim                 import  Optimizer
from    sklearn.gaussian_process    import  GaussianProcessRegressor

from    .gp                         import  eval_gp, sample_coefs, fit_gps
from    .latent_space               import  initial_condition_latent, Autoencoder
from    .enums                      import  NextStep, Result
from    .physics                    import  Physics
from    .latent_dynamics            import  LatentDynamics     
from    .timing                     import  Timer
from    .param                      import  ParameterSpace



# -------------------------------------------------------------------------------------------------
# Simulate latent dynamics
# -------------------------------------------------------------------------------------------------

def average_rom(autoencoder     : Autoencoder, 
                physics         : Physics, 
                latent_dynamics : LatentDynamics, 
                gp_list         : list[GaussianProcessRegressor], 
                param_grid      : np.ndarray):
    """
    This function simulates the latent dynamics for a collection of testing parameters by using
    the mean of the posterior distribution for each coefficient's posterior distribution. 
    Specifically, for each parameter combination, we determine the mean of the posterior 
    distribution for each coefficient. We then use this mean to simulate the latent dynamics 
    forward in time (starting from the latent encoding of the fom initial condition for that 
    combination of coefficients).

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    autoencoder: The actual autoencoder object that we use to map the ICs into the latent space.

    physics: A "Physics" object that stores the datasets for each parameter combination. 
    
    latent_dynamics: A LatentDynamics object which describes how we specify the dynamics in the
    Autoencoder's latent space.    

    gp_list: a list of trained GP regressor objects. The number of elements in this list should 
    match the number of columns in param_grid. The i'th element of this list is a GP regressor 
    object that predicts the i'th coefficient. 

    param_grid: A 2d numpy.ndarray object of shape (number of parameter combination) x (number of 
    parameters). The i,j element of this array holds the value of the j'th parameter in the i'th 
    combination of parameters. 


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------
    
    A 3d numpy ndarray whose i, j, k element holds the k'th component of the j'th time step of 
    the solution to the latent dynamics when we use the latent encoding of the initial condition 
    from the i'th combination of parameter values
    """

    # The param grid needs to be two dimensional, with the first axis corresponding to which 
    # instance of the parameter values we are using. If there is only one parameter, it may be 1d. 
    # We can fix that by adding on an axis with size 1. 
    if (param_grid.ndim == 1):
        param_grid = param_grid.reshape(1, -1)

    # Now fetch the number of combinations of parameter values.
    n_param : int = param_grid.shape[0]

    # For each parameter in param_grid, fetch the corresponding initial condition and then encode
    # it. This gives us a list whose i'th element holds the encoding of the i'th initial condition.
    Z0      : list[np.ndarray] = initial_condition_latent(param_grid, physics, autoencoder)

    # Evaluate each GP at each combination of parameter values. This returns two arrays, the 
    # first of which is a 2d array whose i,j element specifies the mean of the posterior 
    # distribution for the j'th coefficient at the i'th combination of parameter values.
    pred_mean, _ = eval_gp(gp_list, param_grid)

    # For each testing parameter, cycle through the mean value of each coefficient from each 
    # posterior distribution. For each set of coefficients (combination of parameter values), solve
    # the latent dynamics forward in time (starting from the corresponding IC value) and store the
    # resulting solution frames in Zis, a 3d array whose i, j, k element holds the k'th component 
    # of the j'th time step fo the latent solution when we use the coefficients from the posterior 
    # distribution for the i'th combination of parameter values.
    Zis : np.ndarray = np.zeros([n_param, physics.nt, autoencoder.n_z])
    for i in range(n_param):
        Zis[i] = latent_dynamics.simulate(pred_mean[i], Z0[i], physics.t_grid)

    # All done!
    return Zis



def sample_roms(autoencoder     : Autoencoder, 
                physics         : Physics, 
                latent_dynamics : LatentDynamics, 
                gp_list         : list[GaussianProcessRegressor], 
                param_grid      : np.ndarray, 
                n_samples       : int) ->           np.ndarray:
    '''
    This function samples the latent coefficients, solves the corresponding latent dynamics, and 
    then returns the resulting latent solutions. 
    
    Specifically, for each combination of parameter values in the param_grid, we draw n_samples 
    samples of the latent coefficients (from the coefficient posterior distributions evaluated at 
    that parameter value). This gives us a set of n_samples latent dynamics coefficients. For each 
    set of coefficients, we solve the corresponding latent dynamics forward in time and store the 
    resulting solution frames. We do this for each sample and each combination of parameter values,
    resulting in an (n_param, n_sample, n_t, n_z) array of solution frames, which is what we 
    return.

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    autoencoder: An autoencoder. We use this to map the fom IC's (stored in Physics) to the 
    latent space using the autoencoder's encoder.

    physics: A "Physics" object that stores the ICs for each parameter combination. 
    
    latent_dynamics: A LatentDynamics object which describes how we specify the dynamics in the
    Autoencoder's latent space. We use this to simulate the latent dynamics forward in time.

    gp_list: a list of trained GP regressor objects. The number of elements in this list should 
    match the number of columns in param_grid. The i'th element of this list is a GP regressor 
    object that predicts the i'th coefficient. 

    param_grid: A 2d numpy.ndarray object of shape (number of parameter combination) x (number of 
    parameters). The i,j element of this array holds the value of the j'th parameter in the i'th 
    combination of parameters. 

    n_samples: The number of samples we want to draw from each posterior distribution for each 
    coefficient evaluated at each combination of parameter values.
    

    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------
    
    A np.array of size [n_test, n_samples, physics.nt, autoencoder.n_z]. The i, j, k, l element 
    holds the l'th component of the k'th frame of the solution to the latent dynamics when we use 
    the j'th sample of latent coefficients drawn from the posterior distribution for the i'th 
    combination of parameter values (i'th row of param_grid).
    '''

    # The param grid needs to be two dimensional, with the first axis corresponding to which 
    # instance of the parameter values we are using. If there is only one parameter, it may be 1d. 
    # We can fix that by adding on an axis with size 1. 
    if (param_grid.ndim == 1):
        param_grid = param_grid.reshape(1, -1)
    
    # Now fetch the number of combinations of parameter values (rows of param_grid).
    n_param : int = param_grid.shape[0]

    # For each parameter in param_grid, fetch the corresponding initial condition and then encode
    # it. This gives us a list whose i'th element holds the encoding of the i'th initial condition.
    Z0      : list[np.ndarray] = initial_condition_latent(param_grid, physics, autoencoder)

    # Now, for each combination of parameters, draw n_samples samples from the posterior
    # distributions for each coefficient at that combination of parameters. We store these samples 
    # in a list of numpy arrays. The k'th list element is a (n_sample, n_coef) array whose i, j 
    # element stores the i'th sample from the posterior distribution for the j'th coefficient at 
    # the k'th combination of parameter values.
    coef_samples : list[np.ndarray] = [sample_coefs(gp_list, param_grid[i], n_samples) for i in range(n_param)]

    # For each testing parameter, cycle through the samples of the coefficients for that 
    # combination of parameter values. For each set of coefficients, solve the corresponding latent 
    # dynamics forward in time and store the resulting frames in Zis. This is a 4d array whose i, 
    # j, k, l element holds the l'th component of the k'th frame of the solution to the latent 
    # dynamics when we use the j'th sample of latent coefficients drawn from the posterior 
    # distribution for the i'th combination of parameter values.
    Zis = np.zeros([n_param, n_samples, physics.nt, autoencoder.n_z])
    for i, Zi in enumerate(Zis):
        z_ic = Z0[i]
        for j, coef_sample in enumerate(coef_samples[i]):
            Zi[j] = latent_dynamics.simulate(coef_sample, z_ic, physics.t_grid)

    # All done!
    return Zis



def get_fom_max_std(autoencoder : Autoencoder, Zis : np.ndarray) -> int:
    """
    Computes the maximum standard deviation across the trajectories in Zis and returns the
    corresponding parameter index. Specifically, Zis is a 4d tensor of shape (n_test, n_samples, 
    n_t, n_z). The first axis specifies which parameter combination we're using. For each 
    combination of parameters, we assume that we drew n_samples of the posterior distribution of
    the coefficients at that parameter value, simulated the corresponding dynamics for n_t time 
    steps, and then recorded the results in Zis[i]. Thus, Zis[i, j, k, :] represents the k'th 
    time step of the solution to the latent dynamics when we use the coefficients from the j'th 
    sample of the posterior distribution for the i'th set of parameters. 
    
    Let i \in {1, 2, ... , n_test} and k \in {1, 2, ... , n_t}. For each j, we map the k'th frame
    of the j'th solution trajectory for the i'th parameter combination (Zi[i, j, k, :]) to a fom
    frame. We do this for each j (the set of samples), which gives us a collection of n_sample 
    fom frames, representing samples of the distribution of fom frames at the k'th time step 
    when we use the posterior distribution for the i'th set of parameters. For each l \in {1, 2, 
    ... , n_fom}, we compute the STD of the set of l'th components of these n_sample fom frames.
    We do this for each i and k and then figure out which i, k, l combination gives the largest
    STD. We return the corresponding i index. 
    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    autoencoder: The autoencoder. We assume the solved dynamics (whose frames are stored in Zis) 
    take place in the autoencoder's latent space. We use this to decode the solution frames.

    Zis: A 4d numpy array of shape (n_test, n_samples, n_t, n_z) whose i, j, k, l element holds 
    the l'th component of the k'th frame of the solution to the latent dynamics when we use the 
    j'th sample of latent coefficients drawn from the posterior distribution for the i'th testing 
    parameter.

    

    -----------------------------------------------------------------------------------------------
    Returns:
    -----------------------------------------------------------------------------------------------

    An integer. The index of the testing parameter that gives the largest standard deviation. 
    Specifically, for each testing parameter, we compute the STD of each component of the fom 
    solution at each frame generated by samples from the posterior coefficient distribution for 
    that parameter. We compute the maximum of these STDs and pair that number with the parameter. 
    We then return the index of the parameter whose corresponding maximum std (the number we pair
    with it) is greatest.
    """
    # TODO(kevin): currently this evaluate point-wise maximum standard deviation.
    #              is this a proper metric? we might want to consider an average, or L2 norm of std.


    max_std : float = 0.0

    # Cycle through the testing parameters.
    for m, Zi in enumerate(Zis):
        # Zi is a 3d tensor of shape (n_samples, n_t, n_z), where n_samples is the number of 
        # samples of the posterior distribution per parameter, n_t is the number of time steps in
        # the latent dynamics solution, and n_z is the dimension of the latent space. The i,j,k
        # element of Zi is the k'th component of the j'th frame of the solution to the latent 
        # dynamics when the latent dynamics uses the i'th set of sampled parameter values.
        Z_m             : torch.Tensor  = torch.Tensor(Zi)

        # Now decode the frames.
        X_pred_m        : np.ndarray    = autoencoder.decoder(Z_m).detach().numpy()

        # Compute the standard deviation across the sample axis. This gives us an array of shape 
        # (n_t, n_fom) whose i,j element holds the (sample) standard deviation of the j'th component 
        # of the i'th frame of the fom solution. In this case, the sample distribution consists of 
        # the set of j'th components of i'th frames of fom solutions (one for each sample of the 
        # coefficient posterior distributions).
        X_pred_m_std    : np.ndarray    = X_pred_m.std(0)

        # Now compute the maximum standard deviation across frames/fom components.
        max_std_m       : np.float32    = X_pred_m_std.max()

        # If this is bigger than the biggest std we have seen so far, update the maximum.
        if max_std_m > max_std:
            m_index : int   = m
            max_std : float = max_std_m

    # Report the index of the testing parameter that gave the largest maximum std.
    return m_index



# -------------------------------------------------------------------------------------------------
# BayesianGLaSDI class
# -------------------------------------------------------------------------------------------------

# move optimizer parameters to device
def optimizer_to(optim : Optimizer, device : str) -> None:
    """
    This function moves an optimizer object to a specific device. 


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    optim: The optimizer whose device we want to change.

    device: The device we want to move optim onto. 


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Nothing.
    """

    # Cycle through the optimizer's parameters.
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)



class BayesianGLaSDI:
    X_train : torch.Tensor = torch.Tensor([])
    X_test  : torch.Tensor = torch.Tensor([])

    def __init__(self, 
                 physics            : Physics, 
                 autoencoder        : Autoencoder, 
                 latent_dynamics    : LatentDynamics, 
                 param_space        : ParameterSpace, 
                 config             : dict):
        """
        This class runs a full GPLaSDI training. As input, it takes the autoencoder defined as a 
        torch.nn.Module object, a Physics object to recover fom ICs + information on the time 
        discretization, a 

        The "train" method runs the active learning training loop, computes the reconstruction and 
        SINDy loss, trains the GPs, and samples a new FOM data point.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        physics: A "Physics" object that we use to fetch the fom initial conditions (which we 
        encode into latent ICs). Each Physics object has
        a corresponding PDE with parameters, and a way to generate a solution to that equation 
        given a particular set of parameter values (and an IC, BCs). We use this object to generate
        fom solutions which we then use to train the autoencoder/latent dynamics.
         
        Autoencoder: An autoencoder object that we use to compress the fom state to a reduced, 
        latent state.

        latent_dynamics: A LatentDynamics object which describes how we specify the dynamics in the
        Autoencoder's latent space.

        param_space: A Parameter space object which holds the set of testing and training 
        parameters. 

        config: A dictionary housing the settings we wna to use to train the model on 
        the data generated by physics.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        self.autoencoder                    = autoencoder
        self.latent_dynamics                = latent_dynamics
        self.physics                        = physics
        self.param_space                    = param_space

        # Initialize a timer object. We will use this while training.
        self.timer                          = Timer()

        # Extract training/loss hyperparameters from the configuration file. 
        self.n_samples          : int       = config['n_samples']       # Number of samples to draw per coefficient per combination of parameters
        self.lr                 : float     = config['lr']              # Learning rate for the optimizer.
        self.n_iter             : int       = config['n_iter']          # Number of iterations for one train and greedy sampling
        self.max_iter           : int       = config['max_iter']        # Maximum iterations for overall training
        self.max_greedy_iter    : int       = config['max_greedy_iter'] # Maximum iterations for greedy sampling
        self.ld_weight          : float     = config['ld_weight']       # Weight of the SINDy loss in the loss function. \beta_2 in the paper.
        self.coef_weight        : float     = config['coef_weight']     # Weight of the norm of matrix of latent dynamics coefficients. \beta_3 in the paper.

        # Set up the optimizer and loss function.
        self.optimizer          : Optimizer = torch.optim.Adam(autoencoder.parameters(), lr = self.lr)
        self.MSE                            = torch.nn.MSELoss()

        # Set paths for checkpointing. 
        self.path_checkpoint    : str       = config['path_checkpoint']
        self.path_results       : str       = config['path_results']

        # Make sure the checkpoints and results directories exist.
        from os.path import dirname
        from pathlib import Path
        Path(dirname(self.path_checkpoint)).mkdir(  parents = True, exist_ok = True)
        Path(dirname(self.path_results)).mkdir(     parents = True, exist_ok = True)

        # Set the device to train on. We default to cpu.
        device = config['device'] if 'device' in config else 'cpu'
        if (device == 'cuda'):
            assert(torch.cuda.is_available())
            self.device = device
        elif (device == 'mps'):
            assert(torch.backends.mps.is_available())
            self.device = device
        else:
            self.device = 'cpu'

        # Set up variables to aide checkpointing.
        self.best_loss      : float         = np.inf            # The lowest testing loss we have found so far
        self.best_coefs     : np.ndarray    = None              # The best coefficients from the iteration with lowest testing loss
        self.restart_iter   : int           = 0                 # Iteration number at the end of the last training period
        
        # Set placeholder tensors to hold the testing and training data. We expect to set up 
        # X_train to be a tensor of shape (Np, Nt, Nx[0], ... , Nx[Nd - 1]), where Np is the number 
        # of parameter combinations in the training set, Nt is the number of time steps per fom 
        # solution, and Nx[0], ... , Nx[Nd - 1] represent the number of steps along the spatial 
        # axes. X_test has an analogous shape, but it's leading dimension has a size matching the 
        # number of combinations of parameters in the testing set.
        self.X_train        : torch.Tensor  = torch.Tensor([])  
        self.X_test         : torch.Tensor  = torch.Tensor([])

        # All done!
        return



    def train(self) -> None:
        """
        Runs a round of training on the autoencoder.

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        None!


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Make sure we have at least one training data point (the 0 axis of X_Train corresponds 
        # which combination of training parameters we use).
        assert(self.X_train.size(0) > 0)
        assert(self.X_train.size(0) == self.param_space.n_train())

        # Map everything to self's device.
        device              : str               = self.device
        autoencoder_device  : Autoencoder       = self.autoencoder.to(device)
        X_train_device      : torch.Tensor      = self.X_train.to(device)

        # Make sure the checkpoints and results directories exist.
        from pathlib import Path
        Path(self.path_checkpoint).mkdir(   parents = True, exist_ok = True)
        Path(self.path_results).mkdir(      parents = True, exist_ok = True)

        ps                  : ParameterSpace    = self.param_space
        n_train             : int               = ps.n_train()
        ld                  : LatentDynamics    = self.latent_dynamics

        # Determine number of iterations we should run in this round of training.
        next_iter   : int = min(self.restart_iter + self.n_iter, self.max_iter)
        
        # Run the iterations!
        for iter in range(self.restart_iter, next_iter):
            # Begin timing the current training step.            
            self.timer.start("train_step")

            # Zero out the gradients. 
            self.optimizer.zero_grad()


            # -------------------------------------------------------------------------------------
            # Forward pass

            # Run the forward pass. This results in a tensor of shape (Np, Nt, Nz), where Np is the 
            # number of parameters, Nt is the number of time steps in the time series, and Nz is 
            # the latent space dimension. X_Pred, should have the same shape as X_Train, (Np, Nt, 
            # Nx[0], .... , Nx[Nd - 1]). 
            Z               : torch.Tensor  = autoencoder_device.encoder(X_train_device)
            X_pred          : torch.Tensor  = autoencoder_device.decoder(Z)
            Z               : torch.Tensor  = Z.cpu()
            
            # Compute the autoencoder loss. 
            loss_ae         : torch.Tensor  = self.MSE(X_train_device, X_pred)

            # Compute the latent dynamics and coefficient losses. Also fetch the current latent
            # dynamics coefficients for each training point. The latter is stored in a 3d array 
            # called "coefs" of shape (n_train, N_z, N_l), where N_{\mu} = n_train = number of 
            # training parameter combinations, N_z = latent space dimension, and N_l = number of 
            # terms in the SINDy library.
            coefs, loss_ld, loss_coef       = ld.calibrate(Z, self.physics.dt, compute_loss = True, numpy = True)
            max_coef        : np.float32    = np.abs(coefs).max()

            # Compute the final loss.
            loss = loss_ae + self.ld_weight * loss_ld / n_train + self.coef_weight * loss_coef / n_train


            # -------------------------------------------------------------------------------------
            # Backward Pass

            #  Run back propagation and update the model parameters. 
            loss.backward()
            self.optimizer.step()

            # Check if we hit a new minimum loss. If so, make a checkpoint, record the loss and 
            # the iteration number. 
            if loss.item() < self.best_loss:
                torch.save(autoencoder_device.cpu().state_dict(), self.path_checkpoint + '/' + 'checkpoint.pt')
                autoencoder_device  : Autoencoder   = self.autoencoder.to(device)
                self.best_coefs     : np.ndarray    = coefs
                self.best_loss      : float         = loss.item()

            # -------------------------------------------------------------------------------------
            # Report Results from this iteration 

            # Report the current iteration number and losses
            print("Iter: %05d/%d, Loss: %3.10f, Loss AE: %3.10f, Loss LD: %3.10f, Loss COEF: %3.10f, max|c|: %04.1f, "
                  % (iter + 1, self.max_iter, loss.item(), loss_ae.item(), loss_ld.item(), loss_coef.item(), max_coef),
                  end = '')

            # If there are fewer than 6 training examples, report the set of parameter combinations.
            if n_train < 6:
                print('Param: ' + str(np.round(ps.train_space[0, :], 4)), end = '')

                for i in range(1, n_train - 1):
                    print(', ' + str(np.round(ps.train_space[i, :], 4)), end = '')
                print(', ' + str(np.round(ps.train_space[-1, :], 4)))

            # Otherwise, report the final 6 parameter combinations.
            else:
                print('Param: ...', end = '')
                for i in range(5):
                    print(', ' + str(np.round(ps.train_space[-6 + i, :], 4)), end = '')
                print(', ' + str(np.round(ps.train_space[-1, :], 4)))

            # We have finished a training step, stop the timer.
            self.timer.end("train_step")
        
        # We are ready to wrap up the training procedure.
        self.timer.start("finalize")

        # Now that we have completed another round of training, update the restart iteration.
        self.restart_iter += self.n_iter

        # Recover the autoencoder + coefficients which attained the lowest loss. If we recorded 
        # our bess loss in this round of training, then we replace the autoencoder's parameters 
        # with those from the iteration that got the best loss. Otherwise, we use the current 
        # set of coefficients and serialize the current model.
        if ((self.best_coefs is not None) and (self.best_coefs.shape[0] == n_train)):
            state_dict  = torch.load(self.path_checkpoint + '/' + 'checkpoint.pt')
            self.autoencoder.load_state_dict(state_dict)
        else:
            self.best_coefs : np.ndarray = coefs
            torch.save(autoencoder_device.cpu().state_dict(), self.path_checkpoint + '/' + 'checkpoint.pt')

        # Report timing information.
        self.timer.end("finalize")
        self.timer.print()

        # All done!
        return



    def get_new_sample_point(self) -> np.ndarray:
        """
        This function uses a greedy process to sample a new parameter value. Specifically, it runs 
        through each combination of parameters in in self.param_space. For the i'th combination of 
        parameters, we generate a collection of samples of the coefficients in the latent dynamics.
        We draw the k'th sample of the j'th coefficient from the posterior distribution for the 
        j'th coefficient at the i'th combination of parameters. We map the resulting solution back 
        into the real space and evaluate the standard deviation of the fom frames. We return the 
        combination of parameters which engenders the largest standard deviation (see the function
        get_fom_max_std).


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        None!

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        a 2d numpy ndarray object of shape (1, n_param) whose (0, j) element holds the value of 
        the j'th parameter in the new sample.
        """


        self.timer.start("new_sample")
        assert(self.X_test.size(0)      >  0)
        assert(self.X_test.size(0)      == self.param_space.n_test())
        assert(self.best_coefs.shape[0] == self.param_space.n_train())
        coefs : np.ndarray = self.best_coefs

        print('\n~~~~~~~ Finding New Point ~~~~~~~')
        # TODO(kevin): william, this might be the place for new sampling routine.

        # Move the model to the cpu (this is where all the GP stuff happens) and load the model 
        # from the last checkpoint. This should be the one that obtained the best loss so far. 
        # Remember that coefs should specify the coefficients from that iteration. 
        ae          : Autoencoder       = self.autoencoder.cpu()
        ps          : ParameterSpace    = self.param_space
        n_test      : int               = ps.n_test()
        ae.load_state_dict(torch.load(self.path_checkpoint + '/' + 'checkpoint.pt'))

        # Map the initial conditions for the fom to initial conditions in the latent space.
        Z0 : list[np.ndarray] = initial_condition_latent(ps.test_space, self.physics, ae)

        # Train the GPs on the training data, get one GP per latent space coefficient.
        gp_list : list[GaussianProcessRegressor] = fit_gps(ps.train_space, coefs)

        # For each combination of parameter values in the testing set, for each coefficient, 
        # draw a set of samples from the posterior distribution for that coefficient evaluated at
        # the testing parameters. We store the samples for a particular combination of parameter 
        # values in a 2d numpy.ndarray of shape (n_sample, n_coef), whose i, j element holds the 
        # i'th sample of the j'th coefficient. We store the arrays for different parameter values 
        # in a list of length (number of combinations of parameters in the testing set). 
        coef_samples : list[np.ndarray] = [sample_coefs(gp_list, ps.test_space[i], self.n_samples) for i in range(n_test)]

        # Now, solve the latent dynamics forward in time for each set of coefficients in 
        # coef_samples. There are n_test combinations of parameter values, and we have n_samples 
        # sets of coefficients for each combination of parameter values. For each of those, we want 
        # to solve the corresponding latent dynamics for n_t time steps. Each one of the frames 
        # in that solution live in \mathbb{R}^{n_z}. Thus, we need to store the results in a 4d 
        # array of shape (n_test, n_samples, n_t, n_z) whose i, j, k, l element holds the l'th 
        # component of the k'th frame of the solution to the latent dynamics when we use the 
        # j'th sample of the coefficients for the i'th testing parameter value and when the latent
        # dynamics uses the encoding of the i'th fom IC as its IC. 
        Zis : np.ndarray = np.zeros([n_test, self.n_samples, self.physics.nt, ae.n_z])
        for i, Zi in enumerate(Zis):
            z_ic = Z0[i]
            for j, coef_sample in enumerate(coef_samples[i]):
                Zi[j] = self.latent_dynamics.simulate(coef_sample, z_ic, self.physics.t_grid)

        # Find the index of the parameter with the largest std.
        m_index : int = get_fom_max_std(ae, Zis)

        # We have found the testing parameter we want to add to the training set. Fetch it, then
        # stop the timer and return the parameter. 
        new_sample : np.ndarray = ps.test_space[m_index, :].reshape(1, -1)
        print('New param: ' + str(np.round(new_sample, 4)) + '\n')
        self.timer.end("new_sample")

        # All done!
        return new_sample



    def export(self) -> dict:
        """
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        A dictionary housing most of the internal variables in self. You can pass this dictionary 
        to self (after initializing it using ParameterSpace, Autoencoder, and LatentDynamics 
        objects) to make a GLaSDI object whose internal state matches that of self.
        """

        dict_ = {'X_train'          : self.X_train, 
                 'X_test'           : self.X_test, 
                 'lr'               : self.lr, 
                 'n_iter'           : self.n_iter,
                 'n_samples'        : self.n_samples, 
                 'best_coefs'       : self.best_coefs, 
                 'max_iter'         : self.max_iter,
                 'max_iter'         : self.max_iter, 
                 'ld_weight'        : self.ld_weight, 
                 'coef_weight'      : self.coef_weight,
                 'restart_iter'     : self.restart_iter, 
                 'timer'            : self.timer.export(), 
                 'optimizer'        : self.optimizer.state_dict()}
        return dict_



    def load(self, dict_ : dict) -> None:
        """
        Modifies self's internal state to match the one whose export method generated the dict_ 
        dictionary.


        -------------------------------------------------------------------------------------------
        Arguments 
        -------------------------------------------------------------------------------------------

        dict_: This should be a dictionary returned by calling the export method on another 
        GLaSDI object. We use this to make self hav the same internal state as the object that 
        generated dict_. 
        

        -------------------------------------------------------------------------------------------
        Returns  
        -------------------------------------------------------------------------------------------
        
        Nothing!
        """

        # Extract instance variables from dict_.
        self.X_train        : torch.Tensor  = dict_['X_train']
        self.X_test         : torch.Tensor  = dict_['X_test']
        self.best_coefs     : np.ndarray    = dict_['best_coefs']
        self.restart_iter   : int           = dict_['restart_iter']

        # Load the timer / optimizer. 
        self.timer.load(dict_['timer'])
        self.optimizer.load_state_dict(dict_['optimizer'])
        if (self.device != 'cpu'):
            optimizer_to(self.optimizer, self.device)

        # All done!
        return
    