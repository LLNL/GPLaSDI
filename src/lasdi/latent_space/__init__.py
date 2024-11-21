import torch
import numpy as np
from ..networks import MultiLayerPerceptron, CNN2D

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
    
class Conv2DAutoencoder(LatentSpace):
    def __init__(self, physics, config):
        super().__init__(physics, config)
        from lasdi.inputs import InputParser
        parser = InputParser(config)

        assert(physics.dim == 2)

        if (len(self.qgrid_size) == 2):
            cnn_layers = [[1] + self.qgrid_size]
        cnn_layers += parser.getInput(['cnn_layers'], datatype=list)

        strides = parser.getInput(['strides'], fallback=[1] * (len(cnn_layers) - 1))
        paddings = parser.getInput(['paddings'], fallback=[0] * (len(cnn_layers) - 1))
        dilations = parser.getInput(['dilations'], fallback=[1] * (len(cnn_layers) - 1))

        cnn_act_type = parser.getInput(['cnn_activation'], fallback='ReLU')

        batch_shape = parser.getInput(['batch_shape'], datatype=list)
        data_shape = batch_shape + self.qgrid_size

        cnn_f = CNN2D(cnn_layers, 'forward', strides, paddings,
                      dilations, act_type=cnn_act_type, data_shape=data_shape)
        cnn_b = CNN2D(cnn_layers[::-1], 'backward', strides[::-1], paddings[::-1],
                      dilations[::-1], act_type=cnn_act_type, data_shape=data_shape)

        mlp_layers = [np.prod(cnn_f.layer_sizes[-1])]
        mlp_layers += parser.getInput(['mlp_layers'], datatype=list)
        mlp_layers += [self.n_z]

        act_type = parser.getInput(['mlp_activation'], fallback='sigmoid')
        threshold = parser.getInput(['threshold'], fallback=0.1)
        value = parser.getInput(['value'], fallback=0.0)

        mlp_f = MultiLayerPerceptron(mlp_layers, act_type=act_type,
                                     reshape_index=0, reshape_shape=cnn_f.layer_sizes[-1],
                                     threshold=threshold, value=value)
        mlp_b = MultiLayerPerceptron(mlp_layers[::-1], act_type=act_type,
                                     reshape_index=-1, reshape_shape=cnn_b.layer_sizes[0],
                                     threshold=threshold, value=value),
        
        self.encoder = torch.nn.Sequential(cnn_f, mlp_f)
        self.decoder = torch.nn.Sequential(mlp_b, cnn_b)
        
        self.print_architecture()
        
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
    
    def set_batch_shape(self, batch_shape):
        data_shape = batch_shape + self.qgrid_size

        self.encoder[0].set_data_shape(data_shape)
        self.decoder[1].set_data_shape(data_shape)

        self.print_architecture()
        return
    
    def print_architecture(self):
        self.encoder[0].print_data_shape()
        self.encoder[1].print_architecture()
        self.decoder[0].print_architecture()
        self.decoder[1].print_data_shape()
        return