from .sindy import *
from .interp import *
from .latent_space import *
from .enums import *
from .timing import Timer
import torch
import time
import numpy as np

def find_sindy_coef(Z, Dt, n_train, time_dim, loss_function, fd_type):

    '''

    Computes the SINDy loss, reconstruction loss, and sindy coefficients

    '''

    loss_sindy = 0
    loss_coef = 0

    dZdt = compute_time_derivative(Z, Dt, fd_type)
    sindy_coef = []

    for i in range(n_train):

        dZdt_i = dZdt[i, :, :]
        Z_i = torch.cat([torch.ones(time_dim, 1), Z[i, :, :]], dim = 1)
        # coef_matrix_i = Z_i.pinverse() @ dZdt_i
        coef_matrix_i = torch.linalg.lstsq(Z_i, dZdt_i).solution

        loss_sindy += loss_function(dZdt_i, Z_i @ coef_matrix_i)
        loss_coef += torch.norm(coef_matrix_i)

        sindy_coef.append(coef_matrix_i.detach().numpy())

    return loss_sindy, loss_coef, sindy_coef

class BayesianGLaSDI:
    X_train = None

    def __init__(self, physics, autoencoder, latent_dynamics, model_parameters):

        '''

        This class runs a full GPLaSDI training. It takes into input the autoencoder defined as a PyTorch object and the
        dictionnary containing all the training parameters.
        The "train" method with run the active learning training loop, compute the reconstruction and SINDy loss, train the GPs,
        and sample a new FOM data point.

        '''

        self.autoencoder = autoencoder
        self.latent_dynamics = latent_dynamics
        self.physics = physics
        self.param_space = physics.param_space
        self.timer = Timer()

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

        from os.path import dirname
        from pathlib import Path
        Path(dirname(self.path_checkpoint)).mkdir(parents=True, exist_ok=True)
        Path(dirname(self.path_results)).mkdir(parents=True, exist_ok=True)

        device = model_parameters['device'] if 'device' in model_parameters else 'cpu'
        if (device == 'cuda'):
            assert(torch.cuda.is_available())
            self.device = device
        elif (device == 'mps'):
            assert(torch.backends.mps.is_available())
            self.device = device
        else:
            self.device = 'cpu'

        self.best_loss = np.Inf
        self.restart_iter = 0

        return

    def train(self):
        assert(self.X_train is not None)
        assert(self.X_train.size(0) == self.param_space.n_train)

        device = self.device
        autoencoder_device = self.autoencoder.to(device)
        X_train_device = self.X_train.to(device)

        from pathlib import Path
        Path(self.path_checkpoint).mkdir(parents=True, exist_ok=True)
        Path(self.path_results).mkdir(parents=True, exist_ok=True)

        ps = self.param_space
        ld = self.latent_dynamics

        for iter in range(self.restart_iter, self.n_iter):
            self.timer.start("train_step")

            self.optimizer.zero_grad()
            Z = autoencoder_device.encoder(X_train_device)
            X_pred = autoencoder_device.decoder(Z)
            Z = Z.cpu()

            loss_ae = self.MSE(X_train_device, X_pred)
            sindy_coef, loss_sindy, loss_coef = ld.calibrate(Z, self.physics.dt, compute_loss=True, numpy=True)

            max_coef = np.abs(np.array(sindy_coef)).max()

            loss = loss_ae + self.sindy_weight * loss_sindy / ps.n_train + self.coef_weight * loss_coef / ps.n_train

            loss.backward()
            self.optimizer.step()

            if loss.item() < self.best_loss:
                torch.save(autoencoder_device.state_dict(), self.path_checkpoint + '/' + 'checkpoint.pt')
                self.best_sindy_coef = sindy_coef
                self.best_loss = loss.item()

            print("Iter: %05d/%d, Loss: %3.10f, Loss AE: %3.10f, Loss SI: %3.10f, Loss COEF: %3.10f, max|c|: %04.1f, "
                  % (iter + 1, self.n_iter, loss.item(), loss_ae.item(), loss_sindy.item(), loss_coef.item(), max_coef),
                  end = '')

            if ps.n_train < 6:
                print('Param: ' + str(np.round(ps.train_space[0, :], 4)), end = '')

                for i in range(1, ps.n_train - 1):
                    print(', ' + str(np.round(ps.train_space[i, :], 4)), end = '')
                print(', ' + str(np.round(ps.train_space[-1, :], 4)))

            else:
                print('Param: ...', end = '')
                for i in range(5):
                    print(', ' + str(np.round(ps.train_space[-6 + i, :], 4)), end = '')
                print(', ' + str(np.round(ps.train_space[-1, :], 4)))

            self.timer.end("train_step")

            if ((iter > self.restart_iter) and (iter < self.max_greedy_iter) and (iter % self.n_greedy == 0)):

                # self.timer.start("new_sample")

                # print('\n~~~~~~~ Finding New Point ~~~~~~~')
                # # TODO(kevin): need to re-write this part.

                # # X_train = X_train_device.cpu()
                # autoencoder = autoencoder_device.cpu()
                # autoencoder.load_state_dict(torch.load(self.path_checkpoint + '/' + 'checkpoint.pt'))

                if (self.best_sindy_coef.shape[0] == ps.n_train):
                    sindy_coef = self.best_sindy_coef

                # Z0 = initial_condition_latent(ps.test_space, self.physics, autoencoder)

                # interpolation_data = build_interpolation_data(sindy_coef, ps.train_space)
                # gp_dictionnary = fit_gps(interpolation_data)
                # n_coef = interpolation_data['n_coef']

                # coef_samples = [interpolate_coef_matrix(gp_dictionnary, ps.test_space[i, :], self.n_samples, n_coef, sindy_coef) for i in range(ps.n_test)]
                # Zis = [simulate_uncertain_sindy(gp_dictionnary, ps.test_space[i, 0], self.n_samples, Z0[i], self.physics.t_grid, sindy_coef, n_coef, coef_samples[i]) for i in range(ps.n_test)]

                # m_index = get_max_std(autoencoder, Zis)
                new_sample = self.get_new_sample_point(sindy_coef)

                # TODO(kevin): implement save/load the new parameter
                # ps.appendTrainSpace(ps.test_space[m_index, :])
                ps.appendTrainSpace(new_sample)
                self.restart_iter = iter
                next_step, result = NextStep.RunSample, Result.Success
                print('New param: ' + str(np.round(new_sample, 4)) + '\n')
                # self.timer.end("new_sample")
                return result, next_step
        
        self.timer.start("finalize")

        if (self.best_sindy_coef.shape[0] == ps.n_train):
            sindy_coef = self.best_sindy_coef
        sindy_coef = sindy_coef.reshape([ps.n_train, autoencoder_device.n_z+1, autoencoder_device.n_z])
        interpolation_data = build_interpolation_data(sindy_coef, ps.train_space)
        gp_dictionnary = fit_gps(interpolation_data)

        bglasdi_results = {'autoencoder_param': self.autoencoder.state_dict(), 'final_X_train': self.X_train,
                           'sindy_coef': sindy_coef, 'gp_dictionnary': gp_dictionnary, 'lr': self.lr, 'n_iter': self.n_iter,
                           'n_greedy': self.n_greedy, 'sindy_weight': self.sindy_weight, 'coef_weight': self.coef_weight,
                           'n_samples' : self.n_samples,
                           }
        bglasdi_results['physics'] = self.physics.export()
        bglasdi_results['parameters'] = self.param_space.export()
        # TODO(kevin): restart capability for timer.
        bglasdi_results['timer'] = self.timer.export()
        bglasdi_results['latent_dynamics'] = self.latent_dynamics.export()

        date = time.localtime()
        date_str = "{month:02d}_{day:02d}_{year:04d}_{hour:02d}_{minute:02d}"
        date_str = date_str.format(month = date.tm_mon, day = date.tm_mday, year = date.tm_year, hour = date.tm_hour + 3, minute = date.tm_min)
        np.save(self.path_results + '/' + 'bglasdi_' + date_str + '.npy', bglasdi_results)

        self.timer.end("finalize")
        self.timer.print()

        next_step, result = None, Result.Complete
        return result, next_step
    
    def get_new_sample_point(self, sindy_coef):
        self.timer.start("new_sample")

        print('\n~~~~~~~ Finding New Point ~~~~~~~')

        # X_train = X_train_device.cpu()
        ae = self.autoencoder.cpu()
        ps = self.param_space
        ae.load_state_dict(torch.load(self.path_checkpoint + '/' + 'checkpoint.pt'))

        # if (self.best_sindy_coef.shape[0] == ps.n_train):
        #     sindy_coef = self.best_sindy_coef
        # copy is inevitable for numpy==1.26. temporary placeholder.
        sindy_coef = sindy_coef.reshape([ps.n_train, ae.n_z+1, ae.n_z])

        Z0 = initial_condition_latent(ps.test_space, self.physics, ae)

        interpolation_data = build_interpolation_data(sindy_coef, ps.train_space)
        gp_dictionnary = fit_gps(interpolation_data)
        n_coef = interpolation_data['n_coef']

        coef_samples = [interpolate_coef_matrix(gp_dictionnary, ps.test_space[i, :], self.n_samples, n_coef, sindy_coef) for i in range(ps.n_test)]
        Zis = [simulate_uncertain_sindy(gp_dictionnary, ps.test_space[i, 0], self.n_samples, Z0[i], self.physics.t_grid, sindy_coef, n_coef, coef_samples[i]) for i in range(ps.n_test)]

        m_index = get_max_std(ae, Zis)

        self.timer.end("new_sample")
        return ps.test_space[m_index, :]
    
    def sample_fom(self):

        new_foms = self.param_space.n_train - self.X_train.size(0)
        assert(new_foms > 0)
        new_params = self.param_space.train_space[-new_foms:, :]

        if not self.physics.offline:
            new_X = self.physics.generate_solutions(new_params)

            self.X_train = torch.cat([self.X_train, new_X], dim = 0)
        else:
            # TODO(kevin): interface for offline FOM simulation
            raise RuntimeError("Offline FOM simulation is not supported yet!")