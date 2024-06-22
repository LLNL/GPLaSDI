import os
import pdb
import time
import pandas as pd
import numpy as np


def write_files(params, dir, iproc = [1, 1]):

    '''

    Writes all the relevant files to run HyPar with the Vlasov 1D1V equation
    HyPar Repo: http://hypar.github.io/

    '''

    f = open(f'{dir}/solver.inp', 'w')

    f.write('begin\n')
    f.write('ndims 2\n')
    f.write('nvars 1\n')
    f.write(f'size {params["n_x"]} {params["n_v"]}\n')
    f.write('iproc ' + str(iproc[0]) + ' ' + str(iproc[1]) + '\n')
    f.write('ghost 3\n')
    f.write(f'n_iter {params["n_iter"]}\n')
    f.write('restart_iter 0\n')
    f.write('time_scheme rk\n')
    f.write('time_scheme_type 44\n')
    f.write('hyp_space_scheme weno5\n')
    f.write(f'dt {params["dt"]}\n')
    f.write('conservation_check yes\n')
    f.write(f'screen_op_iter {params["output_iter"]}\n')
    f.write(f'file_op_iter {params["output_iter"]}\n')
    f.write('op_file_format tecplot2d\n')
    f.write('ip_file_type binary\n')
    f.write('input_mode serial\n')
    f.write('output_mode serial\n')
    f.write('op_overwrite no\n')
    f.write('model vlasov\n')

    for k, v in params.items():
        f.write(f'{k} {v}\n')

    f.write('end')

    f.close()

    f = open(f'{dir}/boundary.inp', 'w')

    L_box = 2 * np.pi / params['k_Dr']
    f.write('4\n')
    f.write('periodic   0       1    0.0000000000000000e+00 0.0000000000000000e+00 -7.0000000000000000e+00 7.0000000000000000e+00\n')
    f.write('periodic   0      -1    0.0000000000000000e+00 0.0000000000000000e+00 -7.0000000000000000e+00 7.0000000000000000e+00\n')
    f.write(f'dirichlet  1       1    0.0000000000000000e+00 {L_box}   0.0000000000000000e+00 0.0000000000000000e+00\n')
    f.write('0\n')
    f.write(f'dirichlet  1      -1    0.0000000000000000e+00 {L_box}   0.0000000000000000e+00 0.0000000000000000e+00\n')
    f.write('0\n')

    f.close()

    f = open(f'{dir}/physics.inp', 'w')

    f.write('begin\n')
    f.write('self_consistent_electric_field 1')
    f.write('end')

    f.close()

    # f = open('param_T.inp', 'w')
    #
    # f.write(str(T))
    #
    # f.close()
    #
    # f = open('param_k.inp', 'w')
    #
    # f.write(str(k))
    #
    # f.close()

    print(os.getcwd())
    os.system(f'cp ./INIT {dir}/INIT')
    os.chdir(dir)
    print(os.getcwd())
    os.system('./INIT')


def run_hypar(n_proc, dir):

    '''

    HyPar is C++ parallel finite difference solver for PDEs, used to solve Vlasov equation. HyPar needs to be installed, and
    the init.c code in the Vlasov1D1V needs to be compiled first.
    This python function calls HyPar and needs to be run after write_files
    For more: http://hypar.github.io/

    '''

    # os.system('jsrun -n1 -r1 -a' + str(n_proc * n_proc) + ' -g0 -c' + str(n_proc * n_proc) + ' /usr/WS2/lei4/Projects/ARPAE/hypar/bin/HyPar')
    os.system(f'srun -n{n_proc} /usr/WS2/lei4/Projects/ARPAE/hypar/bin/HyPar | tee stdout.txt')


def post_process_data(nt, dir):
    fs, xs, vs = read_f(nt, data_dir=dir)
    es, xs_2 = read_efield(nt, data_dir=dir)
    assert np.all(xs == xs_2)
    return fs, es, xs, vs


def read_f(nt, data_dir=''):
    '''
    Process the HyPar outputs into a numpy dataset
    '''
    print('Post-processing op data...')
    data_dat = 'op_{t:05d}.dat'
    fs = []

    for t in range(nt):
        if t % 100 == 0:
            print(f'\t{t}/{nt}')

        file = np.loadtxt(f'{data_dir}/{data_dat.format(t = t)}', skiprows=2)
        fs.append(file[:, -1])
    fs = np.array(fs)
    xs = np.unique(file[:,-3])
    xs.sort()
    vs = np.unique(file[:,-2])
    vs.sort()
    return fs, xs, vs


def read_efield(nt, data_dir=''):
    '''
    Process the HyPar outputs into a numpy dataset
    '''
    print('Post-processing efield data...')
    data_dat = 'efield_{t:05d}.dat'
    es = []

    for t in range(nt):
        if t % 100 == 0:
            print(f'\t{t}/{nt}')

        file = np.loadtxt(f'{data_dir}/{data_dat.format(t = t)}', skiprows=0)
        es.append(file[:, -1])
    es = np.array(es)
    xs = np.unique(file[:, -2])
    xs.sort()
    return es, xs