import  numpy                               as      np
from    sklearn.gaussian_process.kernels    import  ConstantKernel, Matern, RBF
from    sklearn.gaussian_process            import  GaussianProcessRegressor




# -------------------------------------------------------------------------------------------------
# Gaussian Process functions! 
# -------------------------------------------------------------------------------------------------

def fit_gps(X : np.ndarray, Y : np.ndarray) -> list[GaussianProcessRegressor]:
    """
    Trains a GP for each column of Y. If Y has shape N x k, then we train k GP regressors. In this 
    case, we assume that X has shape N x M. Thus, the Input to the GP is in \mathbb{R}^M. For each
    k, we train a GP where the i'th row of X is the input and the i,k component of Y is the
    corresponding target. Thus, we return a list of k GP Regressor objects, the k'th one of which 
    makes predictions for the k'th coefficient in the latent dynamics. 

    We assume each target coefficient is independent with each other.


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    X: A 2d numpy array of shape (n_train, input_dim), where n_train is the number of training 
    examples and input_dim is the number of components in each input (e.g., the number of 
    parameters)

    Y: A 2d numpy array of shape (n_train, n_coef), where n_train is the number of training 
    examples and n_coef is the number of coefficients in the latent dynamics. 

    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    A list of trained GP regressor objects. If Y has k columns, then the returned list has k 
    elements. It's i'th element holds a trained GP regressor object whose training inputs are the 
    columns of X and whose corresponding target values are the elements of the i'th column of Y.
    """

    # Determine the number of components (columns) of Y. Since this is a regression task, we will
    # perform a GP regression fit on each component (column) of Y.
    n_coef : int = 1 if (Y.ndim == 1) else Y.shape[1]

    # Transpose Y so that each row corresponds to a particular coefficient. This allows us to 
    # iterate over the coefficients by iterating through the rows of Y.
    if (n_coef > 1):
        Y = Y.T

    # Sklearn requires X to be a 2D array... so make sure this holds.
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Initialize a list to hold the trained GP objects.
    gp_list : list[GaussianProcessRegressor] = []

    # Cycle through the rows of Y (which we transposed... so this just cycles through the 
    # coefficients)
    for yk in Y:
        # Make the kernel.
        # kernel = ConstantKernel() * Matern(length_scale_bounds = (0.01, 1e5), nu = 1.5)
        kernel  = ConstantKernel() * RBF(length_scale_bounds = (0.1, 1e5))

        # Initialize the GP object.
        gp      = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 10, random_state = 1)

        # Fit it to the data (train), then add it to the list of trained GPs
        gp.fit(X, yk)
        gp_list += [gp]

    # All done!
    return gp_list



def eval_gp(gp_list : list[GaussianProcessRegressor], param_grid : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the GPs predictive mean and standard deviation for points of the parameter space grid


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    gp_list: a list of trained GP regressor objects. The number of elements in this list should 
    match the number of columns in param_grid. The i'th element of this list is a GP regressor 
    object that predicts the i'th coefficient. 
    
    param_grid: A 2d numpy.ndarray object of shape (number of parameter combination, number of 
    parameters). The i,j element of this array specifies the value of the j'th parameter in the 
    i'th combination of parameters. We use this as the testing set for the GP evaluation.


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------  

    A two element tuple. Both are 2d numpy arrays of shape (number of parameter combinations, 
    number of coefficients). The two arrays hold the predicted means and std's for each parameter
    at each training example, respectively. 
    
    Thus, the i,j element of the first return variable holds the predicted mean of the j'th 
    coefficient in the latent dynamics at the i'th training example. Likewise, the i,j element of 
    the second return variable holds the standard deviation in the predicted distribution for the 
    j'th coefficient in the latent dynamics at the i'th combination of parameter values.
    """

    # Fetch the numbers coefficients. Since there is one GP Regressor per SINDy coefficient, this 
    # just the length of the gp_list.
    n_coef : int = len(gp_list)

    # Fetch the number of parameters, make sure the grid is 2D. 
    if (param_grid.ndim == 1):
        param_grid = param_grid.reshape(1, -1)
    n_points = param_grid.shape[0]

    # Initialize arrays to hold the mean, STD.
    pred_mean, pred_std = np.zeros([n_points, n_coef]), np.zeros([n_points, n_coef])

    # Cycle through the GPs (one for each coefficient in the SINDy coefficients!).
    for k, gp in enumerate(gp_list):
        # Make predictions using the parameters in the param_grid.
        pred_mean[:, k], pred_std[:, k] = gp.predict(param_grid, return_std = True)

    # All done!
    return pred_mean, pred_std



def sample_coefs(   gp_list     : list[GaussianProcessRegressor], 
                    param       : np.ndarray, 
                    n_samples   : int):
    """
    Generates sets of ODE (SINDy) coefficients sampled from the predictive distribution for those 
    coefficients at the specified parameter value (parma). Specifically, for the k'th SINDy 
    coefficient, we draw n_samples samples of the predictive distribution for the k'th coefficient
    when param is the parameter. 
    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    gp_list: a list of trained GP regressor objects. The number of elements in this list should 
    match the number of columns in param_grid. The i'th element of this list is a GP regressor 
    object that predicts the i'th coefficient. 

    param: A combination of parameter values. i.e., a single test example. We evaluate each GP in 
    the gp_list at this parameter value (getting a prediction for each coefficient).

    n_samples: Number of samples of the predicted latent dynamics used to build ensemble of fom 
    predictions. N_s in the paper. 
    

    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    A 2d numpy ndarray object called coef_samples. It has shape (n_samples, n_coef), where n_coef 
    is the number of coefficients (length of gp_list). The i,j element of this list is the i'th 
    sample of the j'th SINDy coefficient.
    """

    # Fetch the number of coefficients (since there is one GP Regressor per coefficient, this is
    # just the length of the gp_list).
    n_coef          : int           = len(gp_list)

    # Initialize an array to hold the coefficient samples.
    coef_samples    : np.ndarray    = np.zeros([n_samples, n_coef])

    # Make sure param is a 2d array with one row, we need this when evaluating the GP Regressor
    # object.
    if param.ndim == 1:
        param = param.reshape(1, -1)
    
    # Make sure we only have a single sample.
    n_points : int = param.shape[0]
    assert(n_points == 1)

    # Evaluate the predicted mean and std at the parameter value.
    pred_mean, pred_std = eval_gp(gp_list, param)
    pred_mean, pred_std = pred_mean[0], pred_std[0]

    # Cycle through the samples and coefficients. For each sample of the k'th coefficient, we draw
    # a sample from the normal distribution with mean pred_mean[k] and std pred_std[k].
    for s in range(n_samples):
        for k in range(n_coef):
            coef_samples[s, k] = np.random.normal(pred_mean[k], pred_std[k])

    # All done!
    return coef_samples