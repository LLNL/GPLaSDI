from solver import *
import numpy as np
import time
import os

T_min, T_max = 1.4, 1.6
k_min, k_max = 1.0, 1.2

n_time_steps = 251

n_T_grid = 21
n_k_grid = 21

T_grid = np.linspace(T_min, T_max, n_T_grid)
k_grid = np.linspace(k_min, k_max, n_k_grid)
T_grid, k_grid = np.meshgrid(T_grid, k_grid)
param_grid = np.hstack((T_grid.reshape(-1, 1), k_grid.reshape(-1, 1)))
n_test = param_grid.shape[0]

batch_size = 20
batch_index = list(np.arange(0, n_test, batch_size))
if batch_index[-1] != n_test:
    batch_index.append(n_test)

n_batch = len(batch_index) - 1

data_dat = 'op_{t:05d}.dat'

for b in range(0, n_batch):
    for i in range(batch_index[b], batch_index[b + 1]):
                
        T, k = param_grid[i, 0], param_grid[i, 1]
        write_files(T, k, [4, 4])
        run_hypar(4)

        X_i = post_process_data(n_time_steps)
        X_i = X_i.reshape(1, X_i.shape[0], X_i.shape[1])

        if i == batch_index[b]:
            X_batch = X_i
        else:
            X_batch = np.concatenate((X_batch, X_i), axis = 0)

        for t in range(n_time_steps):
            os.system('rm ' + data_dat.format(t = t))

        time.sleep(10)
        
                
    param_batch = param_grid[batch_index[b]:batch_index[b + 1], :]
    data_batch = {'param_batch': param_batch, 'X_batch': X_batch}
    np.save('//p/gpfs1/cbonnevi/Vlasov1D1V/data/data_test_batch_' + str(b + 1) + '.npy', data_batch)

os.system('rm *.dat *.inp')
