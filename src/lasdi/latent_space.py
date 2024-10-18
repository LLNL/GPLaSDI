# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import  torch
import  numpy       as      np

from   .physics     import  Physics



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
    parameters).

    physics: A "Physics" object that stores the datasets for each parameter combination. 

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
        u0 = u0.reshape(sol_shape)
        u0 = torch.Tensor(u0)

        # Encode the IC, then map the encoding to a numpy array.
        z0 = autoencoder.encoder(u0)
        z0 = z0[0, 0, :].detach().numpy()

        # Append the new IC to the list of latent ICs
        Z0.append(z0)

    # Return the list of latent ICs.
    return Z0



# -------------------------------------------------------------------------------------------------
# Autoencoder class
# -------------------------------------------------------------------------------------------------

class Autoencoder(torch.nn.Module):
    # set by physics.qgrid_size
    qgrid_size = []
    # prod(qgrid_size)
    space_dim = -1
    n_z = -1

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

    def __init__(self, physics, config):
        super(Autoencoder, self).__init__()

        self.qgrid_size = physics.qgrid_size
        self.space_dim = np.prod(self.qgrid_size)
        hidden_units = config['hidden_units']
        n_z = config['latent_dimension']
        self.n_z = n_z

        n_layers = len(hidden_units)
        self.n_layers = n_layers

        fc1_e = torch.nn.Linear(self.space_dim, hidden_units[0])
        torch.nn.init.xavier_uniform_(fc1_e.weight)
        self.fc1_e = fc1_e

        if n_layers > 1:
            for i in range(n_layers - 1):
                fc_e = torch.nn.Linear(hidden_units[i], hidden_units[i + 1])
                torch.nn.init.xavier_uniform_(fc_e.weight)
                setattr(self, 'fc' + str(i + 2) + '_e', fc_e)

        fc_e = torch.nn.Linear(hidden_units[-1], n_z)
        torch.nn.init.xavier_uniform_(fc_e.weight)
        setattr(self, 'fc' + str(n_layers + 1) + '_e', fc_e)

        act_type = config['activation'] if 'activation' in config else 'sigmoid'
        if act_type == "threshold":
            #grab relevant initialization values from config
            threshold = config["threshold"] if "threshold" in config else 0.1
            value = config["value"] if "value" in config else 0.0
            self.g_e = self.act_dict[act_type](threshold, value)

        elif act_type == "multihead":
            #grab relevant initialization values from config
            num_heads = config['num_heads'] if 'num_heads' in config else 1
            if n_layers > 1:
                for i in range(n_layers):
                    setattr(self, 'a' + str(i + 1), self.act_dict[act_type](hidden_units[i], num_heads))
            self.g_e = torch.nn.Identity()  # No additional activation

        #all other activation functions initialized here
        else:
            self.g_e = self.act_dict[act_type]()

        fc1_d = torch.nn.Linear(n_z, hidden_units[-1])
        torch.nn.init.xavier_uniform_(fc1_d.weight)
        self.fc1_d = fc1_d

        if n_layers > 1:
            for i in range(n_layers - 1, 0, -1):
                fc_d = torch.nn.Linear(hidden_units[i], hidden_units[i - 1])
                torch.nn.init.xavier_uniform_(fc_d.weight)
                setattr(self, 'fc' + str(n_layers - i + 1) + '_d', fc_d)

        fc_d = torch.nn.Linear(hidden_units[0], self.space_dim)
        torch.nn.init.xavier_uniform_(fc_d.weight)
        setattr(self, 'fc' + str(n_layers + 1) + '_d', fc_d)



    def encoder(self, x):
        # make sure the input has a proper shape
        assert(list(x.shape[-len(self.qgrid_size):]) == self.qgrid_size)
        # we use torch.Tensor.view instead of torch.Tensor.reshape,
        # in order to avoid data copying.
        x = x.view(list(x.shape[:-len(self.qgrid_size)]) + [self.space_dim])

        for i in range(1, self.n_layers + 1):
            fc = getattr(self, 'fc' + str(i) + '_e')
            x = fc(x) # apply linear layer
            if hasattr(self, 'a1'): # test if there is at least one attention layer
                x = self.apply_attention(self, x, i)
                
            x = self.g_e(x) # apply activation function

        fc = getattr(self, 'fc' + str(self.n_layers + 1) + '_e')
        x = fc(x)

        return x


    def decoder(self, x):

        for i in range(1, self.n_layers + 1):
            fc = getattr(self, 'fc' + str(i) + '_d')
            x = fc(x) # apply linear layer
            if hasattr(self, 'a1'): # test if there is at least one attention layer
                x = self.apply_attention(self, x, self.n_layers - i)

            x = self.g_e(x) # apply activation function

        fc = getattr(self, 'fc' + str(self.n_layers + 1) + '_d')
        x = fc(x)

        # we use torch.Tensor.view instead of torch.Tensor.reshape,
        # in order to avoid data copying.
        x = x.view(list(x.shape[:-1]) + self.qgrid_size)

        return x


    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x


    def apply_attention(self, x, layer):
        x = x.unsqueeze(1)  # Add sequence dimension for attention
        a = getattr(self, 'a' + str(layer))
        x, _ = a(x, x, x) # apply attention
        x = x.squeeze(1)  # Remove sequence dimension
        
        return x