from utils import *
import torch
import time
import numpy as np

class BayesianGLaSDI:
    def __init__(self, autoencoder, model_parameters):

        '''

        This class runs a full GPLaSDI training. It takes into input the autoencoder defined as a PyTorch object and the
        dictionnary containing all the training parameters.
        The "train" method with run the active learning training loop, compute the reconstruction and SINDy loss, train the GPs,
        and sample a new FOM data point.

        '''

        self.autoencoder = autoencoder

        time_dim, space_dim = model_parameters['time_dim'], model_parameters['space_dim']
        n_z = model_parameters['n_z']

        self.time_dim = time_dim
        self.space_dim = space_dim
        self.n_z = n_z

        Dt, Dx = model_parameters['Dt'], model_parameters['Dx']
        t_grid, x_grid = model_parameters['t_grid'], model_parameters['x_grid']
        initial_condition = model_parameters['initial_condition']

        self.Dt = Dt
        self.Dx = Dx
        self.t_grid = t_grid
        self.x_grid = x_grid
        self.initial_condition = initial_condition

        a_min, a_max = model_parameters['a_min'], model_parameters['a_max']
        w_min, w_max = model_parameters['w_min'], model_parameters['w_max']

        self.a_min = a_min
        self.a_max = a_max
        self.w_min = w_min
        self.w_max = w_max

        n_a_grid, n_w_grid = model_parameters['n_a_grid'], model_parameters['n_w_grid']
        a_grid = np.linspace(a_min, a_max, n_a_grid)
        w_grid = np.linspace(w_min, w_max, n_w_grid)
        a_grid, w_grid = np.meshgrid(a_grid, w_grid)

        self.n_a_grid = n_a_grid
        self.n_w_grid = n_w_grid
        self.a_grid = a_grid
        self.w_grid = w_grid

        data_train = model_parameters['data_train']
        data_test = model_parameters['data_test']

        param_train = data_train['param_train']
        X_train = data_train['X_train']
        X_train = torch.Tensor(X_train)

        self.param_train = param_train
        self.X_train = X_train

        n_init = data_train['n_train']
        n_train = data_train['n_train']

        self.n_init = n_init
        self.n_train = n_train

        param_grid = data_test['param_grid']
        n_test = data_test['n_test']

        self.param_grid = param_grid
        self.n_test = n_test

        n_samples = model_parameters['n_samples']
        lr = model_parameters['lr']
        n_iter = model_parameters['n_iter']
        n_greedy = model_parameters['n_greedy']
        max_greedy_iter = model_parameters['max_greedy_iter']
        sindy_weight = model_parameters['sindy_weight']
        coef_weight = model_parameters['coef_weight']

        self.n_samples = n_samples
        self.lr = lr
        self.n_iter = n_iter
        self.n_greedy = n_greedy
        self.max_greedy_iter = max_greedy_iter
        self.sindy_weight = sindy_weight
        self.coef_weight = coef_weight

        optimizer = torch.optim.Adam(autoencoder.parameters(), lr = lr)
        MSE = torch.nn.MSELoss()

        self.optimizer = optimizer
        self.MSE = MSE

        path_checkpoint = model_parameters['path_checkpoint']
        path_results = model_parameters['path_results']

        self.path_checkpoint = path_checkpoint
        self.path_results = path_results

        if model_parameters['cuda']:
            device = 'cuda'
        else:
            device = 'cpu'

        self.device = device


    def train(self):

        best_loss = np.Inf

        n_iter = self.n_iter
        n_train = self.n_train
        n_test = self.n_test
        n_greedy = self.n_greedy
        max_greedy_iter = self.max_greedy_iter
        time_dim = self.time_dim
        space_dim = self.space_dim
        n_samples = self.n_samples

        n_a_grid = self.n_a_grid
        n_w_grid = self.n_w_grid

        Dt = self.Dt
        Dx = self.Dx

        x_grid = self.x_grid
        t_grid = self.t_grid

        autoencoder = self.autoencoder
        optimizer = self.optimizer
        MSE = self.MSE
        sindy_weight = self.sindy_weight
        coef_weight = self.coef_weight

        param_train = self.param_train
        param_grid = self.param_grid
        X_train = self.X_train

        path_checkpoint = self.path_checkpoint
        path_results = self.path_results

        initial_condition = self.initial_condition

        device = self.device
        autoencoder = autoencoder.to(device)
        X_train = X_train.to(device)

        tic_start = time.time()
        start_train_phase = []
        start_fom_phase = []
        end_train_phase = []
        end_fom_phase = []


        start_train_phase.append(tic_start)

        for iter in range(n_iter):

            optimizer.zero_grad()
            Z = autoencoder.encoder(X_train)
            X_pred = autoencoder.decoder(Z)
            Z = Z.cpu()

            loss_ae = MSE(X_train, X_pred)
            loss_sindy, loss_coef, sindy_coef = find_sindy_coef(Z, Dt, n_train, time_dim, MSE)

            max_coef = np.abs(np.array(sindy_coef)).max()

            loss = loss_ae + sindy_weight * loss_sindy / n_train + coef_weight * loss_coef / n_train

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                torch.save(autoencoder.state_dict(), path_checkpoint + 'checkpoint.pt')
                best_sindy_coef = sindy_coef
                best_loss = loss.item()



            print("Iter: %05d/%d, Loss: %3.10f, Loss AE: %3.10f, Loss SI: %3.10f, Loss COEF: %3.10f, max|c|: %04.1f, "
                  % (iter + 1,
                     n_iter,
                     loss.item(),
                     loss_ae.item(),
                     loss_sindy.item(),
                     loss_coef.item(),
                     max_coef),
                  end = '')

            if n_train < 6:
                print('Param: ' + str(np.round(param_train[0, :], 4)), end = '')

                for i in range(1, n_train - 1):
                    print(', ' + str(np.round(param_train[i, :], 4)), end = '')
                print(', ' + str(np.round(param_train[-1, :], 4)))

            else:
                print('Param: ...', end = '')
                for i in range(5):
                    print(', ' + str(np.round(param_train[-6 + i, :], 4)), end = '')
                print(', ' + str(np.round(param_train[-1, :], 4)))




            if iter > 0 and iter < max_greedy_iter and iter % n_greedy == 0:

                end_train_phase.append(time.time())

                print('\n~~~~~~~ Finding New Point ~~~~~~~')

                start_fom_phase.append(time.time())
                X_train = X_train.cpu()
                autoencoder = autoencoder.cpu()
                autoencoder.load_state_dict(torch.load(path_checkpoint + 'checkpoint.pt'))

                if len(best_sindy_coef) == n_train:
                    sindy_coef = best_sindy_coef

                Z0 = initial_condition_latent(param_grid, initial_condition, x_grid, autoencoder)

                interpolation_data = build_interpolation_data(sindy_coef, param_train)
                gp_dictionnary = fit_gps(interpolation_data)
                n_coef = interpolation_data['n_coef']

                coef_samples = [interpolate_coef_matrix(gp_dictionnary, param_grid[i, :], n_samples, n_coef, sindy_coef) for i in range(n_test)]
                Zis = [simulate_uncertain_sindy(gp_dictionnary, param_grid[i, 0], n_samples, Z0[i], t_grid, sindy_coef, n_coef, coef_samples[i]) for i in range(n_test)]

                a_index, w_index, m_index = get_max_std(autoencoder, Zis, n_a_grid, n_w_grid)

                X_train, param_train = get_new_param(m_index,
                                                     X_train,
                                                     param_train,
                                                     param_grid,
                                                     x_grid,
                                                     initial_condition,
                                                     time_dim,
                                                     space_dim,
                                                     Dt,
                                                     Dx)

                X_train = X_train.to(device)
                n_train += 1
                autoencoder = autoencoder.to(device)

                end_fom_phase.append(time.time())

                print('New param: ' + str(np.round(param_train[-1, :], 4)) + '\n')

                start_train_phase.append(time.time())

        tic_end = time.time()
        total_time = tic_end - tic_start


        autoencoder = autoencoder.cpu()
        X_train = X_train.cpu()

        bglasdi_results = {'autoencoder_param': autoencoder.state_dict(), 'final_param_train': param_train,
                           'final_X_train': X_train, 'param_grid': param_grid,
                           'sindy_coef': sindy_coef, 'gp_dictionnary': gp_dictionnary, 'lr': self.lr, 'n_iter': n_iter,
                           'n_greedy': n_greedy, 'sindy_weight': sindy_weight, 'coef_weight': coef_weight,
                           'n_init': self.n_init, 'n_samples' : n_samples, 'n_a_grid' : n_a_grid, 'n_w_grid' : n_w_grid,
                           'a_grid' : self.a_grid, 'w_grid' : self.w_grid,
                           't_grid' : t_grid, 'x_grid' : x_grid, 'Dt' : Dt, 'Dx' : Dx,
                           'total_time' : total_time, 'start_train_phase' : start_train_phase,
                           'start_fom_phase' : start_fom_phase, 'end_train_phase' : end_train_phase, 'end_fom_phase' : end_fom_phase}

        date = time.localtime()
        date_str = "{month:02d}_{day:02d}_{year:04d}_{hour:02d}_{minute:02d}"
        date_str = date_str.format(month = date.tm_mon, day = date.tm_mday, year = date.tm_year, hour = date.tm_hour + 3, minute = date.tm_min)
        np.save(path_results + 'bglasdi_' + date_str + '.npy', bglasdi_results)
