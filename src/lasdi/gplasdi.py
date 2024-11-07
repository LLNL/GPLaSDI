from .gp import *
from .latent_space import *
from .enums import *
from .timing import Timer
import torch
import time
import numpy as np

def average_rom(autoencoder, physics, latent_dynamics, gp_dictionary, param_grid):

    if (param_grid.ndim == 1):
        param_grid = param_grid.reshape(1, -1)
    n_test = param_grid.shape[0]

    Z0 = initial_condition_latent(param_grid, physics, autoencoder)

    pred_mean, _ = eval_gp(gp_dictionary, param_grid)

    Zis = np.zeros([n_test, physics.nt, autoencoder.n_z])
    for i in range(n_test):
        Zis[i] = latent_dynamics.simulate(pred_mean[i], Z0[i], physics.t_grid)

    return Zis

def sample_roms(autoencoder, physics, latent_dynamics, gp_dictionary, param_grid, n_samples):
    '''
        Collect n_samples of ROM trajectories on param_grid.
        gp_dictionary: list of Gaussian process regressors (size of n_test)
        param_grid: numpy 2d array
        n_samples: integer
        assert(len(gp_dictionnary) == param_grid.shape[0])

        output: np.array of size [n_test, n_samples, physics.nt, autoencoder.n_z]
    '''

    if (param_grid.ndim == 1):
        param_grid = param_grid.reshape(1, -1)
    n_test = param_grid.shape[0]

    Z0 = initial_condition_latent(param_grid, physics, autoencoder)

    coef_samples = [sample_coefs(gp_dictionary, param_grid[i], n_samples) for i in range(n_test)]

    Zis = np.zeros([n_test, n_samples, physics.nt, autoencoder.n_z])
    for i, Zi in enumerate(Zis):
        z_ic = Z0[i]
        for j, coef_sample in enumerate(coef_samples[i]):
            Zi[j] = latent_dynamics.simulate(coef_sample, z_ic, physics.t_grid)

    return Zis

def get_fom_max_std(autoencoder, Zis):

    '''

    Computes the maximum standard deviation accross the parameter space grid and finds the corresponding parameter location

    '''
    # TODO(kevin): currently this evaluate pointwise maximum standard deviation.
    #              is this a proper metric? we might want to consider an average, or L2 norm of std.

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

# move optimizer parameters to device
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

class BayesianGLaSDI:
    X_train = torch.Tensor([])
    X_test = torch.Tensor([])

    def __init__(self, physics, autoencoder, latent_dynamics, param_space, config):

        '''

        This class runs a full GPLaSDI training. It takes into input the autoencoder defined as a PyTorch object and the
        dictionnary containing all the training parameters.
        The "train" method with run the active learning training loop, compute the reconstruction and SINDy loss, train the GPs,
        and sample a new FOM data point.

        '''

        self.autoencoder = autoencoder
        self.latent_dynamics = latent_dynamics
        self.physics = physics
        self.param_space = param_space
        self.timer = Timer()

        self.n_samples = config['n_samples']
        self.lr = config['lr']
        self.n_iter = config['n_iter']      # number of iterations for one train and greedy sampling
        self.max_iter = config['max_iter']  # maximum iterations for overall training
        self.max_greedy_iter = config['max_greedy_iter'] # maximum iterations for greedy sampling
        self.ld_weight = config['ld_weight']
        self.coef_weight = config['coef_weight']

        self.optimizer = torch.optim.Adam(autoencoder.parameters(), lr = self.lr)
        self.MSE = torch.nn.MSELoss()

        self.path_checkpoint = config['path_checkpoint']
        self.path_results = config['path_results']

        from os.path import dirname
        from pathlib import Path
        Path(dirname(self.path_checkpoint)).mkdir(parents=True, exist_ok=True)
        Path(dirname(self.path_results)).mkdir(parents=True, exist_ok=True)

        device = config['device'] if 'device' in config else 'cpu'
        if (device == 'cuda'):
            assert(torch.cuda.is_available())
            self.device = device
        elif (device == 'mps'):
            assert(torch.backends.mps.is_available())
            self.device = device
        else:
            self.device = 'cpu'

        self.best_loss = np.Inf
        self.best_coefs = None
        self.restart_iter = 0

        self.X_train = torch.Tensor([])
        self.X_test = torch.Tensor([])

        return

    def train(self):
        assert(self.X_train.size(0) > 0)
        assert(self.X_train.size(0) == self.param_space.n_train())

        device = self.device
        autoencoder_device = self.autoencoder.to(device)
        X_train_device = self.X_train.to(device)

        from pathlib import Path
        Path(self.path_checkpoint).mkdir(parents=True, exist_ok=True)
        Path(self.path_results).mkdir(parents=True, exist_ok=True)

        ps = self.param_space
        n_train = ps.n_train()
        ld = self.latent_dynamics

        self.training_loss = []
        self.ae_loss = []
        self.ld_loss = []
        self.coef_loss = []

        '''
            determine number of iterations.
            Perform n_iter iterations until overall iterations hit max_iter.
        '''
        next_iter = min(self.restart_iter + self.n_iter, self.max_iter)

        for iter in range(self.restart_iter, next_iter):
            self.timer.start("train_step")

            self.optimizer.zero_grad()
            Z = autoencoder_device.encoder(X_train_device)
            X_pred = autoencoder_device.decoder(Z)
            Z = Z.cpu()

            loss_ae = self.MSE(X_train_device, X_pred)
            coefs, loss_ld, loss_coef = ld.calibrate(Z, self.physics.dt, compute_loss=True, numpy=True)

            max_coef = np.abs(coefs).max()

            loss = loss_ae + self.ld_weight * loss_ld / n_train + self.coef_weight * loss_coef / n_train

            self.training_loss.append(loss.item())
            self.ae_loss.append(loss_ae.item())
            self.ld_loss.append(loss_ld.item())
            self.coef_loss.append(loss_coef.item())

            loss.backward()
            self.optimizer.step()

            if loss.item() < self.best_loss:
                torch.save(autoencoder_device.cpu().state_dict(), self.path_checkpoint + '/' + 'checkpoint.pt')
                autoencoder_device = self.autoencoder.to(device)
                self.best_coefs = coefs
                self.best_loss = loss.item()

            print("Iter: %05d/%d, Loss: %3.10f, Loss AE: %3.10f, Loss LD: %3.10f, Loss COEF: %3.10f, max|c|: %04.1f, "
                  % (iter + 1, self.max_iter, loss.item(), loss_ae.item(), loss_ld.item(), loss_coef.item(), max_coef),
                  end = '')

            if n_train < 6:
                print('Param: ' + str(np.round(ps.train_space[0, :], 4)), end = '')

                for i in range(1, n_train - 1):
                    print(', ' + str(np.round(ps.train_space[i, :], 4)), end = '')
                print(', ' + str(np.round(ps.train_space[-1, :], 4)))

            else:
                print('Param: ...', end = '')
                for i in range(5):
                    print(', ' + str(np.round(ps.train_space[-6 + i, :], 4)), end = '')
                print(', ' + str(np.round(ps.train_space[-1, :], 4)))

            self.timer.end("train_step")
        
        self.timer.start("finalize")

        self.restart_iter += self.n_iter

        if ((self.best_coefs is not None) and (self.best_coefs.shape[0] == n_train)):
            state_dict = torch.load(self.path_checkpoint + '/' + 'checkpoint.pt')
            self.autoencoder.load_state_dict(state_dict)
        else:
            self.best_coefs = coefs
            torch.save(autoencoder_device.cpu().state_dict(), self.path_checkpoint + '/' + 'checkpoint.pt')

        self.timer.end("finalize")
        self.timer.print()

        return
    
    def get_new_sample_point(self):
        self.timer.start("new_sample")
        assert(self.X_test.size(0) > 0)
        assert(self.X_test.size(0) == self.param_space.n_test())
        assert(self.best_coefs.shape[0] == self.param_space.n_train())
        coefs = self.best_coefs

        print('\n~~~~~~~ Finding New Point ~~~~~~~')
        # TODO(kevin): william, this might be the place for new sampling routine.

        ae = self.autoencoder.cpu()
        ps = self.param_space
        n_test = ps.n_test()
        ae.load_state_dict(torch.load(self.path_checkpoint + '/' + 'checkpoint.pt'))

        Z0 = initial_condition_latent(ps.test_space, self.physics, ae)

        gp_dictionnary = fit_gps(ps.train_space, coefs)

        coef_samples = [sample_coefs(gp_dictionnary, ps.test_space[i], self.n_samples) for i in range(n_test)]

        Zis = np.zeros([n_test, self.n_samples, self.physics.nt, ae.n_z])
        for i, Zi in enumerate(Zis):
            z_ic = Z0[i]
            for j, coef_sample in enumerate(coef_samples[i]):
                Zi[j] = self.latent_dynamics.simulate(coef_sample, z_ic, self.physics.t_grid)

        m_index = get_fom_max_std(ae, Zis)

        new_sample = ps.test_space[m_index, :].reshape(1, -1)
        print('New param: ' + str(np.round(new_sample, 4)) + '\n')

        self.timer.end("new_sample")
        return new_sample
        
    def export(self):
        dict_ = {'X_train': self.X_train, 'X_test': self.X_test, 'lr': self.lr, 'n_iter': self.n_iter,
                 'n_samples' : self.n_samples, 'best_coefs': self.best_coefs, 'max_iter': self.max_iter,
                 'max_iter': self.max_iter, 'ld_weight': self.ld_weight, 'coef_weight': self.coef_weight,
                 'restart_iter': self.restart_iter, 'timer': self.timer.export(), 'optimizer': self.optimizer.state_dict(),
                 'training_loss' : self.training_loss, 'ae_loss' : self.ae_loss, 'ld_loss' : self.ld_loss, 'coeff_loss' : self.coef_loss
                 }
        return dict_
    
    def load(self, dict_):
        self.X_train = dict_['X_train']
        self.X_test = dict_['X_test']
        self.best_coefs = dict_['best_coefs']
        self.restart_iter = dict_['restart_iter']
        self.timer.load(dict_['timer'])
        self.optimizer.load_state_dict(dict_['optimizer'])
        if (self.device != 'cpu'):
            optimizer_to(self.optimizer, self.device)
        return
