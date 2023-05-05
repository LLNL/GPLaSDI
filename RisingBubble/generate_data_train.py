import numpy as np
from solver import *

path = '//p/gpfs1/cbonnevi/RisingBubble/data/'

data = np.load(path + 'data_test_batch_1.npy', allow_pickle = True).item()
param_train = data['param_batch']
X_train = data['X_batch']

for i in range(2, 19):
    data = np.load(path + 'data_test_batch_' + str(i) + '.npy', allow_pickle = True).item()
    param_train = np.vstack((param_train, data['param_batch']))
    X_train = np.vstack((X_train, data['X_batch']))

print(param_train)

index = [0, 20, -21, -1]
param_train = param_train[index, :]
print(param_train)
X_train = X_train[index, :, :]

print(param_train)

for i in range(4):
    X_i = X_train[i, :, :]
    X_i = normalize_add_noise(X_i, 0.0)
    X_train[i, :, :]  = X_i

data_train = {'param_train' : param_train, 'X_train' : X_train, 'n_train' : X_train.shape[0]}
np.save(path + 'data_train_noise_000.npy', data_train) 



                                                                                                                         
