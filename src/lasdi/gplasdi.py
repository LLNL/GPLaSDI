from .sindy import *
from .interp import *
from .latent_space import *
from .enums import *
import torch
import time
import numpy as np

def find_sindy_coef(Z, Dt, n_train, time_dim, loss_function):

    '''

    Computes the SINDy loss, reconstruction loss, and sindy coefficients

    '''

    loss_sindy = 0
    loss_coef = 0

    dZdt, Z = compute_sindy_data(Z, Dt)
    sindy_coef = []

    for i in range(n_train):

        dZdt_i = dZdt[i, :, :]
        Z_i = torch.cat([torch.ones(time_dim - 1, 1), Z[i, :, :]], dim = 1)
        # coef_matrix_i = Z_i.pinverse() @ dZdt_i
        coef_matrix_i = torch.linalg.lstsq(Z_i, dZdt_i).solution

        loss_sindy += loss_function(dZdt_i, Z_i @ coef_matrix_i)
        loss_coef += torch.norm(coef_matrix_i)

        sindy_coef.append(coef_matrix_i.detach().numpy())

    return loss_sindy, loss_coef, sindy_coef

class BayesianGLaSDI:
    def __init__(self, autoencoder, model_parameters):

        '''

        This class runs a full GPLaSDI training. It takes into input the autoencoder defined as a PyTorch object and the
        dictionnary containing all the training parameters.
        The "train" method with run the active learning training loop, compute the reconstruction and SINDy loss, train the GPs,
        and sample a new FOM data point.

        '''

        self.autoencoder = autoencoder

        # TODO(kevin): we need to simply possess physics object.
        self.time_dim = model_parameters['time_dim']
        self.space_dim = model_parameters['space_dim']
        # self.n_z = model_parameters['n_z']

        # TODO(kevin): we need to simply possess physics object.
        xmin = model_parameters['xmin']
        xmax = model_parameters['xmax']
        self.Dx = (xmax - xmin) / (self.space_dim - 1)
        assert(self.Dx > 0.)

        tmax = model_parameters['tmax']
        self.Dt = tmax / (self.time_dim - 1)

        self.x_grid = np.linspace(xmin, xmax, self.space_dim)
        self.t_grid = np.linspace(0, tmax, self.time_dim)

        # TODO(kevin): generalize physics
        # self.initial_condition = model_parameters['initial_condition']
        from .physics.burgers1d import initial_condition
        self.initial_condition = initial_condition

        # TODO(kevin): generalize parameters
        self.a_min = model_parameters['a_min']
        self.a_max = model_parameters['a_max']
        self.w_min = model_parameters['w_min']
        self.w_max = model_parameters['w_max']

        self.n_a_grid = model_parameters['n_a_grid']
        self.n_w_grid = model_parameters['n_w_grid']
        a_grid = np.linspace(self.a_min, self.a_max, self.n_a_grid)
        w_grid = np.linspace(self.w_min, self.w_max, self.n_w_grid)
        self.a_grid, self.w_grid = np.meshgrid(a_grid, w_grid)

        self.param_grid = np.hstack((self.a_grid.reshape(-1, 1), self.w_grid.reshape(-1, 1)))
        self.n_test = self.param_grid.shape[0]

        # data_train = model_parameters['data_train']
        # data_test = model_parameters['data_test']

        # self.param_train = data_train['param_train']
        # X_train = data_train['X_train']
        # self.X_train = torch.Tensor(X_train)

        # self.n_init = data_train['n_train']
        # self.n_train = data_train['n_train']

        # self.param_grid = data_test['param_grid']
        # self.n_test = data_test['n_test']

        self.n_samples = model_parameters['n_samples']
        self.lr = model_parameters['lr']
        self.n_iter = model_parameters['n_iter']
        self.n_greedy = model_parameters['n_greedy']
        self.max_greedy_iter = model_parameters['max_greedy_iter']
        self.sindy_weight = model_parameters['sindy_weight']
        self.coef_weight = model_parameters['coef_weight']

        self.optimizer = torch.optim.Adam(autoencoder.parameters(), lr = self.lr)
        self.MSE = torch.nn.MSELoss()

        self.path_checkpoint = model_parameters['path_checkpoint']
        self.path_results = model_parameters['path_results']

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.best_loss = np.Inf
        self.restart_iter = 0

        return

    def train(self):

        # n_iter = self.n_iter
        # n_train = self.n_train
        # n_test = self.n_test
        # n_greedy = self.n_greedy
        # max_greedy_iter = self.max_greedy_iter
        # time_dim = self.time_dim
        # space_dim = self.space_dim
        # n_samples = self.n_samples

        # n_a_grid = self.n_a_grid
        # n_w_grid = self.n_w_grid

        # Dt = self.Dt
        # Dx = self.Dx

        # x_grid = self.x_grid
        # t_grid = self.t_grid

        # autoencoder = self.autoencoder
        # optimizer = self.optimizer
        # MSE = self.MSE
        # sindy_weight = self.sindy_weight
        # coef_weight = self.coef_weight

        # param_train = self.param_train
        # param_grid = self.param_grid
        # X_train = self.X_train

        # path_checkpoint = self.path_checkpoint
        # path_results = self.path_results

        # initial_condition = self.initial_condition

        device = self.device
        autoencoder_device = self.autoencoder.to(device)
        X_train_device = self.X_train.to(device)

        tic_start = time.time()
        start_train_phase = []
        start_fom_phase = []
        end_train_phase = []
        end_fom_phase = []


        start_train_phase.append(tic_start)

        for iter in range(self.restart_iter, self.n_iter):

            self.optimizer.zero_grad()
            Z = autoencoder_device.encoder(X_train_device)
            X_pred = autoencoder_device.decoder(Z)
            Z = Z.cpu()

            loss_ae = self.MSE(X_train_device, X_pred)
            loss_sindy, loss_coef, sindy_coef = find_sindy_coef(Z, self.Dt, self.n_train, self.time_dim, self.MSE)

            max_coef = np.abs(np.array(sindy_coef)).max()

            loss = loss_ae + self.sindy_weight * loss_sindy / self.n_train + self.coef_weight * loss_coef / self.n_train

            loss.backward()
            self.optimizer.step()

            if loss.item() < self.best_loss:
                torch.save(autoencoder_device.state_dict(), self.path_checkpoint + 'checkpoint.pt')
                self.best_sindy_coef = sindy_coef
                self.best_loss = loss.item()

            print("Iter: %05d/%d, Loss: %3.10f, Loss AE: %3.10f, Loss SI: %3.10f, Loss COEF: %3.10f, max|c|: %04.1f, "
                  % (iter + 1, self.n_iter, loss.item(), loss_ae.item(), loss_sindy.item(), loss_coef.item(), max_coef),
                  end = '')

            if self.n_train < 6:
                print('Param: ' + str(np.round(self.param_train[0, :], 4)), end = '')

                for i in range(1, self.n_train - 1):
                    print(', ' + str(np.round(self.param_train[i, :], 4)), end = '')
                print(', ' + str(np.round(self.param_train[-1, :], 4)))

            else:
                print('Param: ...', end = '')
                for i in range(5):
                    print(', ' + str(np.round(self.param_train[-6 + i, :], 4)), end = '')
                print(', ' + str(np.round(self.param_train[-1, :], 4)))


            if ((iter > self.restart_iter) and (iter < self.max_greedy_iter) and (iter % self.n_greedy == 0)):

                end_train_phase.append(time.time())

                print('\n~~~~~~~ Finding New Point ~~~~~~~')
                # TODO(kevin): need to re-write this part.

                start_fom_phase.append(time.time())
                # X_train = X_train_device.cpu()
                autoencoder = autoencoder_device.cpu()
                autoencoder.load_state_dict(torch.load(self.path_checkpoint + 'checkpoint.pt'))

                if len(self.best_sindy_coef) == self.n_train:
                    sindy_coef = self.best_sindy_coef

                Z0 = initial_condition_latent(self.param_grid, self.initial_condition, self.x_grid, autoencoder)

                interpolation_data = build_interpolation_data(sindy_coef, self.param_train)
                gp_dictionnary = fit_gps(interpolation_data)
                n_coef = interpolation_data['n_coef']

                coef_samples = [interpolate_coef_matrix(gp_dictionnary, self.param_grid[i, :], self.n_samples, n_coef, sindy_coef) for i in range(self.n_test)]
                Zis = [simulate_uncertain_sindy(gp_dictionnary, self.param_grid[i, 0], self.n_samples, Z0[i], self.t_grid, sindy_coef, n_coef, coef_samples[i]) for i in range(self.n_test)]

                a_index, w_index, m_index = get_max_std(autoencoder, Zis, self.n_a_grid, self.n_w_grid)

                # TODO(kevin): implement save/load the new parameter
                self.new_param = self.param_grid[m_index, :]
                self.restart_iter = iter
                next_step, result = NextStep.RunSample, Result.Success
                end_fom_phase.append(time.time())
                print('New param: ' + str(np.round(self.new_param, 4)) + '\n') 
                return result, next_step
        
        if len(self.best_sindy_coef) == self.n_train:
            sindy_coef = self.best_sindy_coef
        interpolation_data = build_interpolation_data(sindy_coef, self.param_train)
        gp_dictionnary = fit_gps(interpolation_data)

        tic_end = time.time()
        total_time = tic_end - tic_start

        # autoencoder = autoencoder_device.cpu()
        # X_train = X_train_device.cpu()

        bglasdi_results = {'autoencoder_param': self.autoencoder.state_dict(), 'final_param_train': self.param_train,
                           'final_X_train': self.X_train, 'param_grid': self.param_grid,
                           'sindy_coef': sindy_coef, 'gp_dictionnary': gp_dictionnary, 'lr': self.lr, 'n_iter': self.n_iter,
                           'n_greedy': self.n_greedy, 'sindy_weight': self.sindy_weight, 'coef_weight': self.coef_weight,
                           'n_init': self.n_init, 'n_samples' : self.n_samples, 'n_a_grid' : self.n_a_grid, 'n_w_grid' : self.n_w_grid,
                           'a_grid' : self.a_grid, 'w_grid' : self.w_grid,
                           't_grid' : self.t_grid, 'x_grid' : self.x_grid, 'Dt' : self.Dt, 'Dx' : self.Dx,
                           'total_time' : total_time, 'start_train_phase' : start_train_phase,
                           'start_fom_phase' : start_fom_phase, 'end_train_phase' : end_train_phase, 'end_fom_phase' : end_fom_phase}

        date = time.localtime()
        date_str = "{month:02d}_{day:02d}_{year:04d}_{hour:02d}_{minute:02d}"
        date_str = date_str.format(month = date.tm_mon, day = date.tm_mday, year = date.tm_year, hour = date.tm_hour + 3, minute = date.tm_min)
        np.save(self.path_results + 'bglasdi_' + date_str + '.npy', bglasdi_results)

        next_step, result = None, Result.Complete
        return result, next_step
    
    def sample_fom(self):

        # TODO(kevin): generalize this physics part.
        from .physics.burgers1d import solver

        new_a, new_b = self.new_param[0], self.new_param[1]
        # TODO(kevin): generalize this physics part.
        u0 = self.initial_condition(new_a, new_b, self.x_grid)

        maxk = 10
        convergence_threshold = 1e-8
        new_X = solver(u0, maxk, convergence_threshold, self.time_dim - 1, self.space_dim, self.Dt, self.Dx)
        new_X = new_X.reshape(1, self.time_dim, self.space_dim)
        new_X = torch.Tensor(new_X)

        self.X_train = torch.cat([self.X_train, new_X], dim = 0)
        self.param_train = np.vstack((self.param_train, np.array([[new_a, new_b]])))
        self.n_train += 1