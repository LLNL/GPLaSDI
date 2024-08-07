import torch
import numpy as np

def initial_condition_latent(param_grid, physics, autoencoder):

    '''

    Outputs the initial condition in the latent space: Z0 = encoder(U0)

    '''

    n_param = param_grid.shape[0]
    Z0 = []

    sol_shape = [1, 1] + physics.qgrid_size

    for i in range(n_param):
        # TODO(kevin): generalize parameter class.
        u0 = physics.initial_condition(param_grid[i])
        u0 = u0.reshape(sol_shape)
        u0 = torch.Tensor(u0)
        z0 = autoencoder.encoder(u0)
        z0 = z0[0, 0, :].detach().numpy()
        Z0.append(z0)

    return Z0

def get_max_std(autoencoder, Zis):

    '''

    Computes the maximum standard deviation accross the parameter space grid and finds the corresponding parameter location

    '''

    max_std = 0

    for m, Zi in enumerate(Zis):
        Z_m = torch.Tensor(Zi)
        X_pred_m = autoencoder.decoder(Z_m).detach().numpy()
        X_pred_m_std = X_pred_m.std(0)
        max_std_m = X_pred_m_std.max()

        if max_std_m > max_std:
            m_index = m
            max_std = max_std_m

    return m_index
    
class Autoencoder(torch.nn.Module):
    # set by physics.qgrid_size
    qgrid_size = []
    # prod(qgrid_size)
    space_dim = -1

    # activation dict
    act_dict = {'sigmoid': torch.nn.Sigmoid,
                'softplus': torch.nn.Softplus}

    def __init__(self, physics, config):
        super(Autoencoder, self).__init__()

        self.qgrid_size = physics.qgrid_size
        self.space_dim = np.prod(self.qgrid_size)
        hidden_units = config['hidden_units']
        n_z = config['latent_dimension']

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
            x = self.g_e(fc(x))

        fc = getattr(self, 'fc' + str(self.n_layers + 1) + '_e')
        x = fc(x)

        return x


    def decoder(self, x):

        for i in range(1, self.n_layers + 1):
            fc = getattr(self, 'fc' + str(i) + '_d')
            x = self.g_e(fc(x))

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