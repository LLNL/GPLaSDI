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
        self.ld_weight = model_parameters['ld_weight']
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

        for iter in range(self.restart_iter, self.n_iter):
            self.timer.start("train_step")

            self.optimizer.zero_grad()
            Z = autoencoder_device.encoder(X_train_device)
            X_pred = autoencoder_device.decoder(Z)
            Z = Z.cpu()

            loss_ae = self.MSE(X_train_device, X_pred)
            coefs, loss_ld, loss_coef = ld.calibrate(Z, self.physics.dt, compute_loss=True, numpy=True)

            max_coef = np.abs(coefs).max()

            loss = loss_ae + self.ld_weight * loss_ld / n_train + self.coef_weight * loss_coef / n_train

            loss.backward()
            self.optimizer.step()

            if loss.item() < self.best_loss:
                torch.save(autoencoder_device.cpu().state_dict(), self.path_checkpoint + '/' + 'checkpoint.pt')
                autoencoder_device = self.autoencoder.to(device)
                self.best_coefs = coefs
                self.best_loss = loss.item()

            print("Iter: %05d/%d, Loss: %3.10f, Loss AE: %3.10f, Loss LD: %3.10f, Loss COEF: %3.10f, max|c|: %04.1f, "
                  % (iter + 1, self.n_iter, loss.item(), loss_ae.item(), loss_ld.item(), loss_coef.item(), max_coef),
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

            if ((iter > self.restart_iter) and (iter < self.max_greedy_iter) and (iter % self.n_greedy == 0)):

                if (self.best_coefs.shape[0] == n_train):
                    coefs = self.best_coefs

                new_sample = self.get_new_sample_point(coefs)

                # TODO(kevin): implement save/load the new parameter
                ps.appendTrainSpace(new_sample)
                self.restart_iter = iter
                next_step, result = NextStep.RunSample, Result.Success
                print('New param: ' + str(np.round(new_sample, 4)) + '\n')
                # self.timer.end("new_sample")
                return result, next_step
        
        self.timer.start("finalize")

        if (self.best_coefs.shape[0] == n_train):
            state_dict = torch.load(self.path_checkpoint + '/' + 'checkpoint.pt')
            self.autoencoder.load_state_dict(state_dict)
            coefs = self.best_coefs

        self.timer.end("finalize")
        self.timer.print()

        next_step, result = None, Result.Complete
        return result, next_step
    
    def get_new_sample_point(self, coefs):
        self.timer.start("new_sample")
        assert(self.X_test.size(0) > 0)
        assert(self.X_test.size(0) == self.param_space.n_test())

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

        self.timer.end("new_sample")
        return ps.test_space[m_index, :]
        
    def export(self):
        dict_ = {'X_train': self.X_train, 'X_test': self.X_test, 'lr': self.lr, 'n_iter': self.n_iter, 'n_samples' : self.n_samples,
                 'n_greedy': self.n_greedy, 'ld_weight': self.ld_weight, 'coef_weight': self.coef_weight,
                 'restart_iter': self.restart_iter, 'timer': self.timer.export(), 'optimizer': self.optimizer.state_dict()
                 }
        return dict_
    
    def load(self, dict_):
        self.X_train = dict_['X_train']
        self.X_test = dict_['X_test']
        self.restart_iter = dict_['restart_iter']
        self.timer.load(dict_['timer'])
        self.optimizer.load_state_dict(dict_['optimizer'])
        if (self.device != 'cpu'):
            optimizer_to(self.optimizer, self.device)
        return