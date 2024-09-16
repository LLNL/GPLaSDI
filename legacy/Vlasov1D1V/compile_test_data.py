import numpy as np

for b in range(0, 4):
    data = np.load('//p/gpfs1/cbonnevi/Vlasov1D1V/data/data_test_batch_' + str(b + 1) + '.npy', allow_pickle = True).item()
    X_batch_b = data['X_batch']
    param_batch_b = data['param_batch']

    if b == 0:
        X_batch = X_batch_b
        param_batch = param_batch_b
    else:
        X_batch = np.concatenate((X_batch, X_batch_b), axis = 0)
        param_batch = np.vstack((param_batch, param_batch_b))

data_batch = {'X_batch' : X_batch, 'param_batch' : param_batch, 'n_batch' : X_batch.shape[0]}
np.save('data_batch_compiled_1.npy', data_batch)


