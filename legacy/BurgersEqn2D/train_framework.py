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

        t_grid = model_parameters['t_grid']
        self.t_grid = t_grid

        Dt, Dx, Dy = model_parameters['Dt'], model_parameters['Dx'], model_parameters['Dy']
        ic = model_parameters['ic']
        Re = model_parameters['Re']
        nt, nx, ny = model_parameters['nt'], model_parameters['nx'], model_parameters['ny']
        nxy = model_parameters['nxy']
        maxitr = model_parameters['maxitr']
        tol = model_parameters['tol']

        self.Dt = Dt
        self.Dx = Dx
        self.Dy = Dy
        self.ic = ic
        self.Re = Re
        self.nt = nt
        self.nx = nx
        self.ny = ny
        self.nxy = nxy
        self.maxitr = maxitr
        self.tol = tol

        initial_condition = model_parameters['initial_condition']
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
        param_grid = np.hstack((a_grid.reshape(-1, 1), w_grid.reshape(-1, 1)))
        n_test = param_grid.shape[0]

        self.n_a_grid = n_a_grid
        self.n_w_grid = n_w_grid
        self.a_grid = a_grid
        self.w_grid = w_grid
        self.param_grid = param_grid
        self.n_test = n_test

        data_train = model_parameters['data_train']

        param_train = data_train['param_train']
        X_train = data_train['X_train']
        X_train = torch.Tensor(X_train)

        self.param_train = param_train
        self.X_train = X_train

        n_init = data_train['n_train']
        n_train = data_train['n_train']

        self.n_init = n_init
        self.n_train = n_train

        n_samples = model_parameters['n_samples']
        lr = model_parameters['lr']
        n_iter = model_parameters['n_iter']
        n_greedy = model_parameters['n_greedy']
        sindy_weight = model_parameters['sindy_weight']
        coef_weight = model_parameters['coef_weight']

        self.n_samples = n_samples
        self.lr = lr
        self.n_iter = n_iter
        self.n_greedy = n_greedy
        self.sindy_weight = sindy_weight
        self.coef_weight = coef_weight

        optimizer = torch.optim.Adam(autoencoder.parameters(), lr = lr)
        MSE = torch.nn.MSELoss()

        self.optimizer = optimizer
        self.MSE = MSE

        path_checkpoint = model_parameters['path_checkpoint']
        path_results = model_parameters['path_results']
        keep_best_loss = model_parameters['keep_best_loss']

        self.path_checkpoint = path_checkpoint
        self.path_results = path_results
        self.keep_best_loss = keep_best_loss

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
        time_dim = self.time_dim
        space_dim = self.space_dim
        n_samples = self.n_samples

        n_a_grid = self.n_a_grid
        n_w_grid = self.n_w_grid

        t_grid = self.t_grid

        Dt = self.Dt
        Dx = self.Dx
        Dy = self.Dy
        ic = self.ic
        Re = self.Re
        nt = self.nt
        nx = self.nx
        ny = self.ny
        nxy = self.nxy
        maxitr = self.maxitr
        tol = self.tol

        initial_condition = self.initial_condition

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

        keep_best_loss = self.keep_best_loss

        device = self.device
        autoencoder = autoencoder.to(device)
        X_train = X_train.to(device)

        tic_start = time.time()
        start_train_phase = []
        start_fom_phase = []
        end_train_phase = []
        end_fom_phase = []

        checkpoint_history = []


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


            # if keep_best_loss is True and loss.item() < best_loss:
            if keep_best_loss is True and loss.item() < best_loss and iter > n_iter - 0.05 * n_greedy:
                torch.save(autoencoder.state_dict(), path_checkpoint + 'checkpoint.pt')
                best_sindy_coef = sindy_coef
                best_loss = loss.item()
                checkpoint_history.append(iter)



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




            if iter > 0 and iter % n_greedy == 0:

                end_train_phase.append(time.time())

                print('\n~~~~~~~ Finding New Point ~~~~~~~')

                start_fom_phase.append(time.time())
                X_train = X_train.cpu()
                autoencoder = autoencoder.cpu()
                '''
                if keep_best_loss is True:
                    autoencoder.load_state_dict(torch.load(path_checkpoint + 'checkpoint.pt'))
                    if len(best_sindy_coef) == n_train:
                        sindy_coef = best_sindy_coef
                '''
                Z0 = initial_condition_latent(param_grid, initial_condition, ic, nx, ny, autoencoder)

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
                                                     time_dim,
                                                     space_dim,
                                                     ic,
                                                     Re,
                                                     nx,
                                                     ny,
                                                     nt,
                                                     Dt,
                                                     nxy,
                                                     Dx,
                                                     Dy,
                                                     maxitr,
                                                     tol)


                X_train = X_train.to(device)
                n_train = X_train.shape[0]
                autoencoder = autoencoder.to(device)

                end_fom_phase.append(time.time())

                print('New param: ' + str(np.round(param_train[-1, :], 4)) + '\n')

                start_train_phase.append(time.time())

        tic_end = time.time()
        total_time = tic_end - tic_start


        autoencoder = autoencoder.cpu()

        if keep_best_loss is True:
            autoencoder.load_state_dict(torch.load(path_checkpoint + 'checkpoint.pt'))
            if len(best_sindy_coef) == n_train:
                sindy_coef = best_sindy_coef
        

        if n_greedy > n_iter:
            gp_dictionnary = None

        X_train = X_train.cpu()

        bglasdi_results = {'autoencoder_param': autoencoder.state_dict(), 'final_param_train': param_train,
                           'final_X_train': X_train, 'param_grid': param_grid,
                           'sindy_coef': sindy_coef, 'gp_dictionnary': gp_dictionnary, 'lr': self.lr, 'n_iter': n_iter,
                           'n_greedy': n_greedy, 'sindy_weight': sindy_weight, 'coef_weight': coef_weight,
                           'n_init': self.n_init, 'n_samples' : n_samples, 'n_a_grid' : n_a_grid, 'n_w_grid' : n_w_grid,
                           'a_grid' : self.a_grid, 'w_grid' : self.w_grid,
                           't_grid' : t_grid, 'Dt' : Dt, 'Dx' : Dx, 'Dy' : Dy, 'nx' : nx, 'ny' : ny,
                           'total_time' : total_time, 'start_train_phase' : start_train_phase,
                           'start_fom_phase' : start_fom_phase, 'end_train_phase' : end_train_phase, 'end_fom_phase' : end_fom_phase,
                           'checkpoint_history' : checkpoint_history, 'keep_best_loss' : keep_best_loss}

        date = time.localtime()
        date_str = "{month:02d}_{day:02d}_{year:04d}_{hour:02d}_{minute:02d}"
        date_str = date_str.format(month = date.tm_mon, day = date.tm_mday, year = date.tm_year, hour = date.tm_hour + 3, minute = date.tm_min)
        np.save(path_results + 'bglasdi_' + date_str + '.npy', bglasdi_results)
