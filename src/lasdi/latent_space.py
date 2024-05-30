import torch

def initial_condition_latent(param_grid, initial_condition, x_grid, autoencoder):

    '''

    Outputs the initial condition in the latent space: Z0 = encoder(U0)

    '''

    n_param = param_grid.shape[0]
    Z0 = []

    for i in range(n_param):
        u0 = initial_condition(param_grid[i, 0], param_grid[i, 1], x_grid)
        u0 = u0.reshape(1, 1, -1)
        u0 = torch.Tensor(u0)
        z0 = autoencoder.encoder(u0)
        z0 = z0[0, 0, :].detach().numpy()
        Z0.append(z0)

    return Z0

def get_max_std(autoencoder, Zis, n_a_grid, n_b_grid):

    '''

    Computes the maximum standard deviation accross the parameter space grid and finds the corresponding parameter location

    '''

    max_std = 0
    m = 0

    for j in range(n_b_grid):
        for i in range(n_a_grid):

            Z_m = torch.Tensor(Zis[m])
            X_pred_m = autoencoder.decoder(Z_m).detach().numpy()
            X_pred_m_std = X_pred_m.std(0)
            max_std_m = X_pred_m_std.max()


            if max_std_m > max_std:
                    a_index = i
                    b_index = j
                    m_index = m
                    max_std = max_std_m

            m += 1

    return a_index, b_index, m_index

class Autoencoder(torch.nn.Module):
    def __init__(self, space_dim, hidden_units, n_z):
        super(Autoencoder, self).__init__()

        fc1_e = torch.nn.Linear(space_dim, hidden_units)
        fc2_e = torch.nn.Linear(hidden_units, n_z)

        g_e = torch.nn.Sigmoid()

        self.fc1_e = fc1_e
        self.fc2_e = fc2_e
        self.g_e = g_e

        fc1_d = torch.nn.Linear(n_z, hidden_units)
        fc2_d = torch.nn.Linear(hidden_units, space_dim)

        self.fc1_d = fc1_d
        self.fc2_d = fc2_d


    def encoder(self, x):

        x = self.g_e(self.fc1_e(x))
        x = self.fc2_e(x)

        return x


    def decoder(self, x):

        x = self.g_e(self.fc1_d(x))
        x = self.fc2_d(x)

        return x


    def forward(self, x):

        x = Autoencoder.encoder(self, x)
        x = Autoencoder.decoder(self, x)

        return x