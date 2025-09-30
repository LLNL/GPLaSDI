import torch
import numpy as np

# activation dict
act_dict = {'ELU': torch.nn.ELU,
            'hardshrink': torch.nn.Hardshrink,
            'hardsigmoid': torch.nn.Hardsigmoid,
            'hardtanh': torch.nn.Hardtanh,
            'hardswish': torch.nn.Hardswish,
            'leakyReLU': torch.nn.LeakyReLU,
            'logsigmoid': torch.nn.LogSigmoid,
            'multihead': torch.nn.MultiheadAttention,
            'PReLU': torch.nn.PReLU,
            'ReLU': torch.nn.ReLU,
            'ReLU6': torch.nn.ReLU6,
            'RReLU': torch.nn.RReLU,
            'SELU': torch.nn.SELU,
            'CELU': torch.nn.CELU,
            'GELU': torch.nn.GELU,
            'sigmoid': torch.nn.Sigmoid,
            'SiLU': torch.nn.SiLU,
            'mish': torch.nn.Mish,
            'softplus': torch.nn.Softplus,
            'softshrink': torch.nn.Softshrink,
            'tanh': torch.nn.Tanh,
            'tanhshrink': torch.nn.Tanhshrink,
            'threshold': torch.nn.Threshold,
            }

def initial_condition_latent(param_grid, physics, autoencoder):
    '''
        Outputs the initial condition in the latent space: Z0 = encoder(U0)

        Arguments
        ---------
        param_grid : :obj:`numpy.array`
            A 2d array of shape `(n_param, param_dim)` for parameter points to obtain initial condition.
        physics : :obj:`lasdi.physics.Physics`
            Physics class to generate initial condition.
        autoencoder : :obj:`lasdi.latent_space.Autoencoder`
            Autoencoder class to encode initial conditions into latent variables.

        Returns
        -------
        Z0 : :obj:`torch.Tensor`
            a torch tensor of size `(n_param, n_z)`, where `n_z` is the latent variable dimension
            defined by `autoencoder`.
    '''

    n_param = param_grid.shape[0]
    Z0 = []

    sol_shape = [1, 1] + physics.qgrid_size

    for i in range(n_param):
        u0 = physics.initial_condition(param_grid[i])
        u0 = u0.reshape(sol_shape)
        u0 = torch.Tensor(u0)
        z0 = autoencoder.encoder(u0)
        z0 = z0[0, 0, :].detach().numpy()
        Z0.append(z0)

    return Z0

class MultiLayerPerceptron(torch.nn.Module):
    """A standard multi-layer perceptron (MLP) module."""

    def __init__(self, layer_sizes,
                 act_type='sigmoid', reshape_index=None, reshape_shape=None,
                 threshold=0.1, value=0.0, num_heads=1):
        super(MultiLayerPerceptron, self).__init__()

        self.n_layers = len(layer_sizes)
        """:obj:`int`: Depth of MLP including input, hidden, and output layers."""
        self.layer_sizes = layer_sizes
        """:obj:`list(int)`: Widths of each MLP layer, including input, hidden and output layers."""

        self.fcs = []
        """:obj:`torch.nn.ModuleList`: torch module list of :math:`(self.n\_layers-1)` linear layers,
                                        connecting from input to output layers.
        """
        for k in range(self.n_layers-1):
            self.fcs += [torch.nn.Linear(layer_sizes[k], layer_sizes[k + 1])]
        self.fcs = torch.nn.ModuleList(self.fcs)
        self.init_weight()

        # Reshape input or output layer
        assert((reshape_index is None) or (reshape_index in [0, -1]))
        assert((reshape_shape is None) or (np.prod(reshape_shape) == layer_sizes[reshape_index]))

        self.reshape_index = reshape_index
        """:obj:`int`: Index of the layer to reshape.

        * 0: Input data is n-dimensional and will be squeezed into 1d tensor for MLP input.
        * 1: Output data should be n-dimensional and MLP output will be reshaped as such.
        """
        self.reshape_shape = reshape_shape
        """:obj:`list(int)`: Shape of the layer to be reshaped.

        * :math:`(self.reshape_index=0)`: Shape of the input data that will be squeezed into 1d tensor for MLP input.
        * :math:`(self.reshape_index=1)`: Shape of the output data into which MLP output shall be reshaped.
        """

        # Initalize activation function
        self.act_type = act_type
        """:obj:`str`: type of activation function"""
        self.use_multihead = False
        """:obj:`bool`: switch to use multihead attention.
        
        Warning:
            this attribute is obsolete and will be removed in future.
        """

        self.act = None
        """:obj:`torch.nn.Module`: activation function"""
        if act_type == "threshold":
            self.act = act_dict[act_type](threshold, value)
            

        elif act_type == "multihead":
            self.use_multihead = True
            if (self.n_layers > 3): # if you have more than one hidden layer
                self.act = []
                for i in range(self.n_layers-2):
                    self.act += [act_dict[act_type](layer_sizes[i+1], num_heads)]
            else:
                self.act = [torch.nn.Identity()]  # No additional activation
            self.act = torch.nn.ModuleList(self.fcs)

        #all other activation functions initialized here
        else:
            self.act = act_dict[act_type]()
        return
    
    def forward(self, x):
        """Pass the input through the MLP layers.

        Args:
            x (:obj:`torch.Tensor`): n-dimensional torch.Tensor for input data.

        Note:
            * If :obj:`self.reshape_index == 0`, then the last n dimensions of :obj:`x` must match :obj:`self.reshape_shape`. In other words, :obj:`list(x.shape[-len(self.reshape_shape):]) == self.reshape_shape`
            * If :obj:`self.reshape_index == -1`, then the last layer output :obj:`z` is reshaped into :obj:`self.reshape_shape`. In other words, :obj:`list(z.shape[-len(self.reshape_shape):]) == self.reshape_shape`

        Returns:
            n-dimensional torch.Tensor for output data.

        """

        if (self.reshape_index == 0):
            # make sure the input has a proper shape
            assert(list(x.shape[-len(self.reshape_shape):]) == self.reshape_shape)
            # we use torch.Tensor.view instead of torch.Tensor.reshape,
            # in order to avoid data copying.
            x = x.view(list(x.shape[:-len(self.reshape_shape)]) + [self.layer_sizes[self.reshape_index]])

        for i in range(self.n_layers-2):
            x = self.fcs[i](x) # apply linear layer
            if (self.use_multihead):
                x = self.apply_attention(self, x, i)
            else:
                x = self.act(x)

        x = self.fcs[-1](x)

        if (self.reshape_index == -1):
            # we use torch.Tensor.view instead of torch.Tensor.reshape,
            # in order to avoid data copying.
            x = x.view(list(x.shape[:-1]) + self.reshape_shape)

        return x
    
    def apply_attention(self, x, act_idx):
        x = x.unsqueeze(1)  # Add sequence dimension for attention
        x, _ = self.act[act_idx](x, x, x) # apply attention
        x = x.squeeze(1)  # Remove sequence dimension
        return x
    
    def init_weight(self):
        """Initialize the weights and biases of the linear layers.

        Returns:
            Does not return a value.

        """
        # TODO(kevin): support other initializations?
        for fc in self.fcs:
            torch.nn.init.xavier_uniform_(fc.weight)
        return

class Autoencoder(torch.nn.Module):
    """A standard autoencoder using MLP.

    Args:
        physics (:obj:`lasdi.physics.Physics`): Physics class that specifies full-order model solution dimensions.

        config: (:obj:`dict`): options for autoencoder. It must include the following keys and values.
            * :obj:`'hidden_units'`: a list of integers for the widths of hidden layers.
            * :obj:`'latent_dimension'`: integer for the latent space dimension.
            * :obj:`'activation'`: string for type of activation function.
    """

    def __init__(self, physics, config):
        super(Autoencoder, self).__init__()

        self.qgrid_size = physics.qgrid_size
        self.space_dim = np.prod(self.qgrid_size)
        hidden_units = config['hidden_units']
        n_z = config['latent_dimension']
        self.n_z = n_z

        layer_sizes = [self.space_dim] + hidden_units + [n_z]
        #grab relevant initialization values from config
        act_type = config['activation'] if 'activation' in config else 'sigmoid'
        threshold = config["threshold"] if "threshold" in config else 0.1
        value = config["value"] if "value" in config else 0.0
        num_heads = config['num_heads'] if 'num_heads' in config else 1

        self.encoder = MultiLayerPerceptron(layer_sizes, act_type,
                                            reshape_index=0, reshape_shape=self.qgrid_size,
                                            threshold=threshold, value=value, num_heads=num_heads)
        
        self.decoder = MultiLayerPerceptron(layer_sizes[::-1], act_type,
                                            reshape_index=-1, reshape_shape=self.qgrid_size,
                                            threshold=threshold, value=value, num_heads=num_heads)

        return

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x
    
    def export(self):
        dict_ = {'autoencoder_param': self.cpu().state_dict()}
        return dict_
    
    def load(self, dict_):
        self.load_state_dict(dict_['autoencoder_param'])
        return
    
class MLPWithMask(MultiLayerPerceptron):
    """Multi-layer perceptron with additional mask output.
    
    Args:
        mlp (:obj:`lasdi.latent_space.MultiLayerPerceptron`): MultiLayerPerceptron class to copy.
        The same architecture, activation function, reshaping will be used.

    """

    def __init__(self, mlp):
        assert(isinstance(mlp, MultiLayerPerceptron))
        from copy import deepcopy
        torch.nn.Module.__init__(self)

        # including input, hidden, output layers
        self.n_layers = mlp.n_layers
        self.layer_sizes = deepcopy(mlp.layer_sizes)

        # Linear features between layers
        self.fcs = deepcopy(mlp.fcs)

        # Reshape input or output layer
        self.reshape_index = deepcopy(mlp.reshape_index)
        self.reshape_shape = deepcopy(mlp.reshape_shape)

        # Initalize activation function
        self.act_type = mlp.act_type
        self.use_multihead = mlp.use_multihead
        self.act = deepcopy(mlp.act)

        self.bool_d = torch.nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1])
        """:obj:`torch.nn.Linear`: additional linear layer to output a mask variable."""
        torch.nn.init.xavier_uniform_(self.bool_d.weight)

        self.sigmoid = torch.nn.Sigmoid()
        """:obj:`torch.nn.Sigmoid`: mask output passes through the sigmoid activation function to ensure :math:`[0, 1]`."""
        return

    def forward(self, x):
        """Pass the input through the MLP layers.

        Args:
            x (:obj:`torch.Tensor`): n-dimensional torch.Tensor for input data.

        Note:
            * If :obj:`self.reshape_index == 0`, then the last n dimensions of :obj:`x` must match :obj:`self.reshape_shape`. In other words, :obj:`list(x.shape[-len(self.reshape_shape):]) == self.reshape_shape`
            * If :obj:`self.reshape_index == -1`, then the last layer outputs :obj:`xval` and :obj:`xbool` are reshaped into :obj:`self.reshape_shape`. In other words, :obj:`list(xval.shape[-len(self.reshape_shape):]) == self.reshape_shape`

        Returns:
            xval (:obj:`torch.Tensor`): n-dimensional torch.Tensor for output data.
            xbool (:obj:`torch.Tensor`): n-dimensional torch.Tensor for output mask.

        """
        if (self.reshape_index == 0):
            # make sure the input has a proper shape
            assert(list(x.shape[-len(self.reshape_shape):]) == self.reshape_shape)
            # we use torch.Tensor.view instead of torch.Tensor.reshape,
            # in order to avoid data copying.
            x = x.view(list(x.shape[:-len(self.reshape_shape)]) + [self.layer_sizes[self.reshape_index]])

        for i in range(self.n_layers-2):
            x = self.fcs[i](x) # apply linear layer
            if (self.use_multihead):
                x = self.apply_attention(self, x, i)
            else:
                x = self.act(x)

        xval, xbool = self.fcs[-1](x), self.bool_d(x)
        xbool = self.sigmoid(xbool)

        if (self.reshape_index == -1):
            # we use torch.Tensor.view instead of torch.Tensor.reshape,
            # in order to avoid data copying.
            xval = xval.view(list(x.shape[:-1]) + self.reshape_shape)
            xbool = xbool.view(list(x.shape[:-1]) + self.reshape_shape)

        return xval, xbool

class AutoEncoderWithMask(Autoencoder):
    """Autoencoder class with additional mask output.

    Its decoder is :obj:`lasdi.latent_space.MLPWithMask`,
    which has an additional mask output.

    Note:
        Unlike the standard autoencoder, the decoder output will have two outputs (with the same shape of the input of the encoder).
    """

    def __init__(self, physics, config):
        Autoencoder.__init__(self, physics, config)

        self.decoder = MLPWithMask(self.decoder)
        return