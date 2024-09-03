from solver import *
import numpy as np
import time
import os

np.random.seed(0)

tc_min, tc_max = 0.5, 0.6
rc_min, rc_max = 150, 160

n_time_steps = 301

n_tc_grid = 21
n_rc_grid = 21

tc_grid = np.linspace(tc_min, tc_max, n_tc_grid)
rc_grid = np.linspace(rc_min, rc_max, n_rc_grid)
tc_grid, rc_grid = np.meshgrid(tc_grid, rc_grid)
param_grid = np.hstack((tc_grid.reshape(-1, 1), rc_grid.reshape(-1, 1)))
n_test = param_grid.shape[0]

batch_size = 25
batch_index = list(np.arange(0, n_test, batch_size))
if batch_index[-1] != n_test:
    batch_index.append(n_test)

n_batch = len(batch_index) - 1

data_bin = 'op_{t:05d}.bin'
data_dat = 'op_{t:05d}.dat'

for b in range(n_batch):
    if b < 11:
        continue

    for i in range(batch_index[b], batch_index[b + 1]):

        tc, rc = param_grid[i, 0], param_grid[i, 1]
        write_files(tc, rc, [4, 4])
        time.sleep(1)
        run_hypar(4)

        X_i = post_process_data(n_time_steps)
        X_i  = X_i.reshape(1, X_i.shape[0], X_i.shape[1])

        if i == batch_index[b]:
            X_batch = X_i
        else:
            X_batch = np.concatenate((X_batch, X_i), axis = 0)

        for t in range(n_time_steps):
            os.system('rm ' + data_bin.format(t = t))
            os.system('rm ' + data_dat.format(t = t)) 
   
        time.sleep(10)

    param_batch = param_grid[batch_index[b]:batch_index[b + 1], :]
    data_batch = {'param_batch': param_batch, 'X_batch': X_batch}
    np.save('//p/gpfs1/cbonnevi/RisingBubble/data/data_test_batch_' + str(b + 1) + '.npy', data_batch)


