import os
import time
import pandas as pd
import numpy as np

def write_files(T, k, iproc = [1, 1]):

    '''

    Writes all the relevant files to run HyPar with the Vlasov 1D1V equation
    HyPar Repo: http://hypar.github.io/

    '''

    f = open('solver.inp', 'w')

    f.write('begin\n')
    f.write('ndims 2\n')
    f.write('nvars 1\n')
    f.write('size 128 128\n')
    f.write('iproc ' + str(iproc[0]) + ' ' + str(iproc[1]) + '\n')
    f.write('ghost 3\n')
    f.write('n_iter 1000\n')
    f.write('restart_iter 0\n')
    f.write('time_scheme rk\n')
    f.write('time_scheme_type 44\n')
    f.write('hyp_space_scheme weno5\n')
    f.write('dt 0.005\n')
    f.write('conservation_check yes\n')
    f.write('screen_op_iter 4\n')
    f.write('file_op_iter 4\n')
    f.write('op_file_format tecplot2d\n')
    f.write('ip_file_type binary\n')
    f.write('input_mode serial\n')
    f.write('output_mode serial\n')
    f.write('op_overwrite no\n')
    f.write('model vlasov\n')
    f.write('end')

    f.close()

    f = open('boundary.inp', 'w')

    f.write('4\n')
    f.write('periodic   0       1    0.0000000000000000e+00 0.0000000000000000e+00 -7.0000000000000000e+00 7.0000000000000000e+00\n')
    f.write('periodic   0      -1    0.0000000000000000e+00 0.0000000000000000e+00 -7.0000000000000000e+00 7.0000000000000000e+00\n')
    f.write('dirichlet  1       1    0.0000000000000000e+00 6.28318530717959e+000   0.0000000000000000e+00 0.0000000000000000e+00\n')
    f.write('0\n')
    f.write('dirichlet  1      -1    0.0000000000000000e+00 6.28318530717959e+000   0.0000000000000000e+00 0.0000000000000000e+00\n')
    f.write('0\n')

    f.close()

    f = open('physics.inp', 'w')

    f.write('begin\n')
    f.write('self_consistent_electric_field 1')
    f.write('end')

    f.close()

    f = open('param_T.inp', 'w')

    f.write(str(T))

    f.close()

    f = open('param_k.inp', 'w')

    f.write(str(k))

    f.close()

    os.system('INIT')


def run_hypar(n_proc):

    '''

    HyPar is C++ parallel finite difference solver for PDEs, used to solve Vlasov equation. HyPar needs to be installed, and
    the init.c code in the Vlasov1D1V needs to be compiled first.
    This python function calls HyPar and needs to be run after write_files
    For more: http://hypar.github.io/

    '''

    os.system('jsrun -n1 -r1 -a' + str(n_proc * n_proc) + ' -g0 -c' + str(n_proc * n_proc) + ' //g/g92/cbonnevi/hypar/bin/HyPar')

def post_process_data(nt):

    '''

    Process the HyPar outputs into a numpy dataset

    '''

    data_dat = 'op_{t:05d}.dat'
    X = []

    for t in range(nt):
        while os.path.exists(data_dat.format(t = t)) is False:
            time.sleep(0.05)

    for t in range(nt):
        
        while os.path.exists(data_dat.format(t = t)) is False:
            time.sleep(0.5)
 
        U = pd.read_table(data_dat.format(t = t)).values
        U = U[1:, :]
        n = U.shape[0]

        u = []

        for i in range(n):

            _, _, _, _, u_i = U[i, 0].split()

            u.append(float(u_i))

        u = np.array(u)

        X.append(u)

    X = np.array(X)

    return X
