# -------------------------------------------------------------------------------------------------
# Imports and setup
# -------------------------------------------------------------------------------------------------

import  torch
import  numpy       as      np

from   .physics     import  Physics

# activation dict
act_dict = {'ELU'           : torch.nn.ELU,
            'hardshrink'    : torch.nn.Hardshrink,
            'hardsigmoid'   : torch.nn.Hardsigmoid,
            'hardtanh'      : torch.nn.Hardtanh,
            'hardswish'     : torch.nn.Hardswish,
            'leakyReLU'     : torch.nn.LeakyReLU,
            'logsigmoid'    : torch.nn.LogSigmoid,
            'PReLU'         : torch.nn.PReLU,
            'ReLU'          : torch.nn.ReLU,
            'ReLU6'         : torch.nn.ReLU6,
            'RReLU'         : torch.nn.RReLU,
            'SELU'          : torch.nn.SELU,
            'CELU'          : torch.nn.CELU,
            'GELU'          : torch.nn.GELU,
            'sigmoid'       : torch.nn.Sigmoid,
            'SiLU'          : torch.nn.SiLU,
            'mish'          : torch.nn.Mish,
            'softplus'      : torch.nn.Softplus,
            'softshrink'    : torch.nn.Softshrink,
            'tanh'          : torch.nn.Tanh,
            'tanhshrink'    : torch.nn.Tanhshrink,
            'threshold'     : torch.nn.Threshold,}



# -------------------------------------------------------------------------------------------------
# initial_conditions_latent function
# -------------------------------------------------------------------------------------------------

def initial_condition_latent(param_grid     : np.ndarray, 
                             physics        : Physics, 
                             autoencoder    : torch.nn.Module) -> list[np.ndarray]:
    """
    This function maps a set of initial conditions for the fom to initial conditions for the 
    latent space dynamics. Specifically, we take in a set of possible parameter values. For each 
    set of parameter values, we recover the fom IC (from physics), then map this fom IC to a 
    latent space IC (by encoding it using the autoencoder). We do this for each parameter 
    combination and then return a list housing the latent space ICs.

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    param_grid: A 2d numpy.ndarray object of shape (number of parameter combination) x (number of 
    parameters). The i,j element of this array holds the value of the j'th parameter in the i'th 
    combination of parameters.

    physics: A "Physics" object that, among other things, stores the IC for each combination of 
    parameters. 

    autoencoder: The actual autoencoder object that we use to map the ICs into the latent space.


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------
    
    A list of numpy ndarray objects whose i'th element holds the latent space initial condition 
    for the i'th set of parameters in the param_grid. That is, if we let U0_i denote the fom IC for 
    the i'th set of parameters, then the i'th element of the returned list is Z0_i = encoder(U0_i).
    """

    # Figure out how many parameters we are testing at. 
    n_param     : int               = param_grid.shape[0]
    Z0          : list[np.ndarray]  = []
    sol_shape   : list[int]         = [1, 1] + physics.qgrid_size
    
    # Cycle through the parameters.
    for i in range(n_param):
        # TODO(kevin): generalize parameter class.

        # Fetch the IC for the i'th set of parameters. Then map it to a tensor.
        u0 : np.ndarray = physics.initial_condition(param_grid[i])
        u0              = u0.reshape(sol_shape)
        u0              = torch.Tensor(u0)

        # Encode the IC, then map the encoding to a numpy array.
        z0 : np.ndarray = autoencoder.encoder(u0)
        z0              = z0[0, 0, :].detach().numpy()

        # Append the new IC to the list of latent ICs
        Z0.append(z0)

    # Return the list of latent ICs.
    return Z0



# -------------------------------------------------------------------------------------------------
# MLP class
# -------------------------------------------------------------------------------------------------

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(   self, 
                    layer_sizes     : list[int],
                    act_type        : str           = 'sigmoid',
                    reshape_index   : int           = None, 
                    reshape_shape   : tuple[int]    = None,
                    threshold       : float         = 0.1, 
                    value           : float         = 0.0) -> None:
        """
        This class defines a standard multi-layer network network.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        layer_sizes: A list of integers specifying the widths of the layers (including the 
        dimensionality of the domain of each layer, as well as the co-domain of the final layer).
        Suppose this list has N elements. Then the network will have N - 1 layers. The i'th layer 
        maps from \mathbb{R}^{layer_sizes[i]} to \mathbb{R}^{layers_sizes[i]}. Thus, the i'th 
        element of this list represents the domain of the i'th layer AND the co-domain of the 
        i-1'th layer.

        act_type: A string specifying which activation function we want to use at the end of each 
        layer (except the final one). We use the same activation for each layer. 

        reshape_index: This argument specifies if we should reshape the network's input or output 
        (or neither). If the user specifies reshape_index, then it must be either 0 or -1. Further, 
        in this case, they must also specify reshape_shape (you need to specify both together). If
        it is 0, then reshape_shape specifies how we reshape the input before passing it through 
        the network (the input to the first layer). If reshape_index is -1, then reshape_shape 
        specifies how we reshape the network output before returning it (the output to the last 
        layer). 

        reshape_shape: These are lists of k integers specifying the final k dimensions of the shape
        of the input to the first layer (if reshape_index == 0) or the output of the last layer 
        (if reshape_index == -1). You must specify this argument if and only if you specify 
        reshape_index. 

        threshold: You should only specify this argument if you are using the "Threshold" 
        activation type. In this case, it specifies the threshold value for that activation (if x
        is the input and x > threshold, then the function returns x. Otherwise, it returns the 
        value).

        value: You should only specify this argument if you are using the "Threshold" activation
        type. In this case, "value" specifies the value that the function returns if the input is
        below the threshold. 


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        Nothing!
        """

        # Run checks.
        # Make sure the reshape index is either 0 (input to 1st layer) or -1 (output of last 
        # layer). Also make sure that that product of the dimensions in the reshape_shape match
        # the input dimension for the 1st layer (if reshape_index == 0) or the output dimension of
        # the last layer (if reshape_index == 1). 
        # 
        # Why do we need the later condition? Well, suppose that reshape_shape has a length of k. 
        # If reshape_index == 0, then we squeeze the final k dimensions of the input and feed that 
        # into the first layer. Thus, in this case, we need the last dimension of the squeezed 
        # array to match the input domain of the first layer. On the other hand, reshape_index == -1 
        # then we reshape the final dimension of the output to match the reshape_shape. Thus, in 
        # both cases, we need the product of the components of reshape_shape to match a 
        # corresponding element of layer_sizes.
        assert((reshape_index is None) or (reshape_index in [0, -1]))
        assert((reshape_shape is None) or (np.prod(reshape_shape) == layer_sizes[reshape_index]))

        super(MultiLayerPerceptron, self).__init__()

        # Note that layer_sizes specifies the dimensionality of the domains and co-domains of each
        # layer. Specifically, the i'th element specifies the input dimension of the i'th layer,
        # while the final element specifies the dimensionality of the co-domain of the final layer.
        # Thus, the number of layers is one less than the length of layer_sizes.
        self.n_layers       : int                   = len(layer_sizes) - 1;
        self.layer_sizes    : list[int]             = layer_sizes

        # Set up the affine parts of the layers.
        self.layers            : list[torch.nn.Module] = [];
        for k in range(self.n_layers):
            self.layers += [torch.nn.Linear(layer_sizes[k], layer_sizes[k + 1])]
        self.layers = torch.nn.ModuleList(self.layers)

        # Now, initialize the weight matrices and bias vectors in the affine portion of each layer.
        self.init_weight()

        # Reshape input to the 1st layer or output of the last layer.
        self.reshape_index : int        = reshape_index
        self.reshape_shape : list[int]  = reshape_shape

        # Set up the activation function. Note that we need to handle the threshold activation type
        # separately because it requires an extra set of parameters to initialize.
        self.act_type   : str               = act_type
        if act_type == "threshold":
            self.act    : torch.nn.Module   = act_dict[act_type](threshold, value)
        else:
            self.act    : torch.nn.Module   = act_dict[act_type]()

        # All done!
        return
    


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        This function defines the forward pass through self.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        x: A tensor holding a batch of inputs. We pass this tensor through the network's layers 
        and then return the result. If self.reshape_index == 0 and self.reshape_shape has k
        elements, then the final k elements of x's shape must match self.reshape_shape. 


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        The image of x under the network's layers. If self.reshape_index == -1 and 
        self.reshape_shape has k elements, then we reshape the output so that the final k elements
        of its shape match those of self.reshape_shape.
        """

        # If the reshape_index is 0, we need to reshape x before passing it through the first 
        # layer.
        if (self.reshape_index == 0):
            # Make sure the input has a proper shape. There is a lot going on in this line; let's
            # break it down. If reshape_index == 0, then we need to reshape the input, x, before
            # passing it through the layers. Let's assume that reshape_shape has k elements. Then,
            # we need to squeeze the final k dimensions of the input, x, so that the resulting 
            # tensor has a final dimension size that matches the input dimension size for the first
            # layer. The check below makes sure that the final k dimensions of the input, x, match
            # the stored reshape_shape.
            assert(list(x.shape[-len(self.reshape_shape):]) == self.reshape_shape)
            
            # Now that we know the final k dimensions of x have the correct shape, let's squeeze 
            # them into 1 dimension (so that we can pass the squeezed tensor through the first 
            # layer). To do this, we reshape x by keeping all but the last k dimensions of x, and 
            # replacing the last k with a single dimension whose size matches the dimensionality of
            # the domain of the first layer. Note that we use torch.Tensor.view instead of 
            # torch.Tensor.reshape in order to avoid data copying.
            x = x.view(list(x.shape[:-len(self.reshape_shape)]) + [self.layer_sizes[self.reshape_index]])

        # Pass x through the network layers (except for the final one, which has no activation 
        # function).
        for i in range(self.n_layers - 1):
            x : torch.Tensor = self.layers[i](x)   # apply linear layer
            x : torch.Tensor = self.act(x)      # apply activation

        # Apply the final (output) layer.
        x = self.layers[-1](x)

        # If the reshape_index is -1, then we need to reshape the output before returning. 
        if (self.reshape_index == -1):
            # In this case, we need to split the last dimension of x, the output of the final
            # layer, to match the reshape_shape. This is precisely what the line below does. Note
            # that we use torch.Tensor.view instead of torch.Tensor.reshape in order to avoid data 
            # copying. 
            x = x.view(list(x.shape[:-1]) + self.reshape_shape)

        # All done!
        return x
    


    def init_weight(self) -> None:
        """
        This function initializes the weight matrices and bias vectors in self's layers. It takes 
        no arguments and returns nothing!
        """

        # TODO(kevin): support other initializations?
        for layer in self.layers:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        
        # All done!
        return



# -------------------------------------------------------------------------------------------------
# Autoencoder class
# -------------------------------------------------------------------------------------------------

class Autoencoder(torch.nn.Module):
    def __init__(self, physics : Physics, config : dict) -> None:
        """
        Initializes an Autoencoder object. An Autoencoder consists of two networks, an encoder, 
        E : \mathbb{R}^F -> \mathbb{R}^L, and a decoder, D : \mathbb{R}^L -> \marthbb{R}^F. We 
        assume that the dataset consists of samples of a parameterized L-manifold in 
        \mathbb{R}^F. The idea then is that E and D act like the inverse coordinate patch and 
        coordinate patch, respectively. In our case, E and D are trainable neural networks. We 
        try to train E and map data in \mathbb{R}^F to elements of a low dimensional latent 
        space (\mathbb{R}^L) which D can send back to the original data. (thus, E, and D should
        act like inverses of one another).

        The Autoencoder class implements this model as a trainable torch.nn.Module object. 


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        physics: A "Physics" object that holds the fom solution frames. We use this object to 
        determine the shape of each fom solution frame. Recall that each Physics object has a 
        corresponding PDE. We 

        config: A dictionary representing the loaded .yml configuration file. We expect it to have 
        the following keys/:
            hidden_units: A list of integers specifying the dimension of the co-domain of each 
            encoder layer except for the final one. Thus, if the k'th layer maps from 
            \mathbb{R}^{n(k)} to \mathbb{R}^{n(k + 1)} and there are K layers (indexed 0, 1, ... , 
            K - 1), then hidden_units should specify n(1), ... , n(K - 1). 

            latent_dimension: The dimensionality of the Autoencoder's latent space. Equivalently, 
            the dimensionality of the co-domain of the encoder (i.e., the dimensionality of the 
            co-domain of the last layer of the encoder) and the domain of the decoder (i.e., the 
            dimensionality of the domain of the first layer of the decoder).
            

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        super(Autoencoder, self).__init__()

        # A Physics object's qgrid_size is a list of integers specifying the shape of each frame of 
        # the fom solution. If the solution is scalar valued, then this is just a list whose i'th 
        # element specifies the number of grid points along the i'th spatial axis. If the solution 
        # is vector valued, however, we prepend the dimensionality of the vector field to the list 
        # from the scalar list (so the 0 element represents the dimension of the vector field at 
        # each point).
        self.qgrid_size : list[int]     = physics.qgrid_size;

        # The product of the elements of qgrid_size is the number of dimensions in each fom 
        # solution frame. This number represents the dimensionality of the input to the encoder 
        # (since we pass a flattened fom frame as input).
        self.space_dim  : np.ndarray    = np.prod(self.qgrid_size);

        # Fetch information about the domain/co-domain of each encoder layer.
        hidden_units    : list[int]     = config['hidden_units'];
        n_z             : int           = config['latent_dimension'];
        self.n_z        : int           = n_z;

        # Build the "layer_sizes" argument for the MLP class. This consists of the dimensions of 
        # each layers' domain + the dimension of the co-domain of the final layer.
        layer_sizes = [self.space_dim] + hidden_units + [n_z];

        # Use the settings to set up the activation information for the encoder.
        act_type    = config['activation']  if 'activation' in config else 'sigmoid'
        threshold   = config["threshold"]   if "threshold"  in config else 0.1
        value       = config["value"]       if "value"      in config else 0.0

        # Now, build the encoder.
        self.encoder = MultiLayerPerceptron(
                            layer_sizes         = layer_sizes, 
                            act_type            = act_type,
                            reshape_index       = 0,                    # We need to flatten the spatial dimensions of each fom frame.
                            reshape_shape       = self.qgrid_size,
                            threshold           = threshold, 
                            value               = value);

        self.decoder = MultiLayerPerceptron(
                            layer_sizes         = layer_sizes[::-1],    # Reverses the order of the the list.
                            act_type            = act_type,
                            reshape_index       = -1,               
                            reshape_shape       = self.qgrid_size,      # We need to reshape the network output to a fom frame.
                            threshold           = threshold, 
                            value               = value)

        # All done!
        return



    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        This function defines the forward pass through self.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        x: A tensor holding a batch of inputs. We pass this tensor through the encoder + decoder 
        and then return the result.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        The image of x under the encoder + decoder. 
        """

        # Encoder the input
        z : torch.Tensor    = self.encoder(x)

        # Now decode z.
        y : torch.Tensor    = self.decoder(z)

        # All done! Hopefully y \approx x.
        return y
    


    def export(self) -> dict:
        """
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        This function extracts self's parameters and returns them in a dictionary. You can pass 
        the dictionary returned by this function to the load method of another Autoencoder object 
        (that you initialized to have the same architecture as self) to make the other autoencoder
        identical to self.
        """

        # TO DO: deep export which includes all information needed to re-initialize self from 
        # scratch. This would probably require changing the initializer.

        dict_ = {   'autoencoder_param' : self.cpu().state_dict()}
        return dict_
    


    def load(self, dict_ : dict) -> None:
        """
        This function loads self's state dictionary.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        dict_: This should be a dictionary with the key "autoencoder_param" whose corresponding 
        value is the state dictionary of an autoencoder which has the same architecture (i.e., 
        layer sizes) as self.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        self.load_state_dict(dict_['autoencoder_param'])
        return