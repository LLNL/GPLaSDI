import numpy as np
import torch

def compute_errors(n_a_grid, n_b_grid, Zis, autoencoder, X_test):

    '''

    Compute the maximum relative errors accross the parameter space grid

    '''
    
    max_e_relative = np.zeros([n_a_grid, n_b_grid])
    max_e_relative_mean = np.zeros([n_a_grid, n_b_grid])
    max_std = np.zeros([n_a_grid, n_b_grid])

    m = 0

    for j in range(n_b_grid):
        for i in range(n_a_grid):

            Z_m = torch.Tensor(Zis[m])
            X_pred_m = autoencoder.decoder(Z_m).detach().numpy()
            e_relative_m = np.linalg.norm((X_test[m:m + 1, :, :] - X_pred_m), axis = 2) / np.linalg.norm(X_test[m:m + 1, :, :], axis = 2)
            e_relative_m_mean = np.linalg.norm((X_test[m, :, :] - X_pred_m.mean(0)), axis = 1) / np.linalg.norm(X_test[m, :, :], axis = 1)
            max_e_relative_m = e_relative_m.max()
            max_e_relative_m_mean = e_relative_m_mean.max()
            max_std_m = X_pred_m.std(0).max()

            max_e_relative[j, i] = max_e_relative_m
            max_e_relative_mean[j, i] = max_e_relative_m_mean
            max_std[j, i] = max_std_m

            m += 1

    return max_e_relative, max_e_relative_mean, max_std
