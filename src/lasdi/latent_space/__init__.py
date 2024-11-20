import torch
import numpy as np
from ..networks import MultiLayerPerceptron

def initial_condition_latent(param_grid, physics, autoencoder):

    '''

    Outputs the initial condition in the latent space: Z0 = encoder(U0)

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

class LatentSpace(torch.nn.Module):

    def __init__(self, physics, config):
        super(LatentSpace, self).__init__()

        self.qgrid_size = physics.qgrid_size
        self.n_z = config['latent_dimension']

        return

    def forward(self, x):
        raise RuntimeError("LatentSpace.forward: abstract method!")

    def export(self):
        dict_ = {'qgrid_size': self.qgrid_size,
                 'n_z': self.n_z}
        return dict_

    def load(self, dict_):
        """
        Notes
        -----
        This abstract class only checks if the variables in restart file are the same as the instance attributes.
        """

        assert(dict_['qgrid_size'] == self.qgrid_size)
        assert(dict_['n_z'] == self.n_z)
        return


class Autoencoder(LatentSpace):

    def __init__(self, physics, config):
        super().__init__(physics, config)

        self.space_dim = np.prod(self.qgrid_size)
        hidden_units = config['hidden_units']

        layer_sizes = [self.space_dim] + hidden_units + [self.n_z]
        #grab relevant initialization values from config
        act_type = config['activation'] if 'activation' in config else 'sigmoid'
        threshold = config["threshold"] if "threshold" in config else 0.1
        value = config["value"] if "value" in config else 0.0

        self.encoder = MultiLayerPerceptron(layer_sizes, act_type,
                                            reshape_index=0, reshape_shape=self.qgrid_size,
                                            threshold=threshold, value=value)
        
        self.decoder = MultiLayerPerceptron(layer_sizes[::-1], act_type,
                                            reshape_index=-1, reshape_shape=self.qgrid_size,
                                            threshold=threshold, value=value)

        return

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x
    
    def export(self):
        dict_ = super().export()
        dict_['autoencoder_param'] = self.cpu().state_dict()
        return dict_
    
    def load(self, dict_):
        super().load(dict_)
        self.load_state_dict(dict_['autoencoder_param'])
        return