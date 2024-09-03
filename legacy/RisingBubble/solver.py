import os
import time
import pandas as pd
import numpy as np

np.random.seed(0)

def write_files(tc, rc, iproc = [1, 1]):

    '''

    Writes all the relevant files to run HyPar with the coupled Euler/AD equation
    HyPar Repo: http://hypar.github.io/

    '''

    f = open('solver.inp', 'w')

    f.write('begin\n')
    f.write('ndims 2\n')
    f.write('nvars 4\n')
    f.write('size 101 101\n')
    f.write('iproc ' + str(iproc[0]) + ' ' + str(iproc[1]) + '\n')
    f.write('ghost 3\n')
    f.write('n_iter 30000\n')
    f.write('restart_iter 0\n')
    f.write('time_scheme rk\n')
    f.write('time_scheme_type ssprk3\n')
    f.write('hyp_space_scheme weno5\n')
    f.write('hyp_flux_split no\n')
    f.write('hyp_interp_type components\n')
    f.write('par_space_type nonconservative-2stage\n')
    f.write('par_space_scheme 4\n')
    f.write('dt 0.01\n')
    f.write('conservation_check no\n')
    f.write('screen_op_iter 100\n')
    f.write('file_op_iter 100\n')
    f.write('input_mode serial\n')
    f.write('ip_file_type binary\n')
    f.write('output_mode serial\n')
    f.write('op_file_format binary\n')
    f.write('op_overwrite no\n')
    f.write('model navierstokes2d\n')
    f.write('end')

    f.close()

    f = open('boundary.inp', 'w')

    f.write('4\n')
    f.write('slip-wall   0 1 0 0 0 1000.0\n')
    f.write('0.0 0.0\n')
    f.write('slip-wall   0 -1 0 0 0 1000.0\n')
    f.write('0.0 0.0\n')
    f.write('slip-wall   1 1 0 1000.0 0 0\n')
    f.write('0.0 0.0\n')
    f.write('slip-wall   1 -1 0 1000.0 0 0\n')
    f.write('0.0 0.0\n')

    f.close()

    f = open('physics.inp', 'w')

    f.write('begin\n')
    f.write('gamma 1.4\n')
    f.write('upwinding rusanov\n')
    f.write('gravity 0.0 9.8\n')
    f.write('rho_ref 1.1612055171196529\n')
    f.write('p_ref 100000.0\n')
    f.write('R 287.058\n')
    f.write('HB 2\n')
    f.write('end')

    f.close()

    f = open('weno.inp', 'w')

    f.write('begin\n')
    f.write('mapped 0\n')
    f.write('borges 0\n')
    f.write('yc 1\n')
    f.write('no_limiting 0\n')
    f.write('epsilon 0.000001\n')
    f.write('p 2.0\n')
    f.write('rc 0.3\n')
    f.write('xi 0.001\n')
    f.write('end')

    f.close()

    f = open('param_rc.inp', 'w')

    f.write(str(rc))

    f.close()

    f = open('param_tc.inp', 'w')

    f.write(str(tc))

    f.close()

    os.system('INIT')


def run_hypar(n_proc):

    '''

    HyPar is C++ parallel finite difference solver for PDEs, used to solve Euler/AD equation. HyPar needs to be installed, and
    the init.c code in the RisingBubble needs to be compiled first.
    This python function calls HyPar and needs to be run after write_files
    For more: http://hypar.github.io/

    '''

    os.system('jsrun -n1 -r1 -a' + str(n_proc * n_proc) + ' -g0 -c' + str(n_proc * n_proc) + ' //g/g92/cbonnevi/hypar/bin/HyPar')
    

def post_process_data(nt):

    '''

    Process the HyPar outputs into a numpy dataset. PostProcess.c needs to be compiled in order to process the outputs from HyPar

    '''

    data_bin = 'op_{t:05d}.bin'
    data_dat = 'op_{t:05d}.dat'
    X = []

    for t in range(nt):
        while os.path.exists(data_bin.format(t = t)) is False:
            time.sleep(0.05)

    os.system('PP')

    for t in range(nt):
        while os.path.exists(data_dat.format(t = t)) is False:
            time.sleep(0.05)

    for t in range(nt):

        U = pd.read_table(data_dat.format(t = t)).values
        n = U.shape[0]

        theta = []

        for i in range(n - 100):

            _, _, _, _, _, _, _, _, theta_i, _, _, _, _ = U[i, 0].split()

            theta.append(float(theta_i))

        theta = np.array(theta)

        X.append(theta)

    X = np.array(X)

    return X

def normalize_add_noise(X, noise):

    '''

    Normalize the data and add Gaussian noise if needed

    '''

    time_dim, space_dim = X.shape
    X_norm = X - 300
    max_abs_norm = np.max(np.abs(X_norm))

    X_noisy = X_norm + np.random.normal(0, noise * max_abs_norm, [time_dim, space_dim])

    return X_noisy



