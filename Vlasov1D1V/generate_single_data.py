from solver import *
import numpy as np
import time
import os, sys
import argparse
from plot import plot_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate test data for Vlasov 1D1V')
    parser.add_argument('-v_drift', type=float, default=2.0)
    parser.add_argument('-T1', type=float, default=1.0)
    parser.add_argument('-T2', type=float, default=1.0)
    parser.add_argument('-k_Dr', type=float, default=1.0)
    parser.add_argument('-eps', type=float, default=0.1)
    parser.add_argument('-n_x', type=int, default=256)
    parser.add_argument('-n_v', type=int, default=1024)
    parser.add_argument('-n_iter', type=int, default=20000)
    parser.add_argument('-dt', type=float, default=0.005)
    parser.add_argument('-out_dt', type=float, default=0.02)
    args = parser.parse_args()

    params = {
        'v_drift': args.v_drift,
        'T1': args.T1,
        'T2': args.T2,
        'k_Dr': args.k_Dr,
        'eps': args.eps,
        'n_x': args.n_x,
        'n_v': args.n_v,
        'n_iter': args.n_iter,
        'dt': args.dt,
        'output_iter': int(args.out_dt / args.dt)
    }
    data_dir = (f'./data_hypar/vd{args.v_drift}_T{args.T1},{args.T2}_kDr{args.k_Dr}_eps{args.eps}_'
                f'nx{args.n_x}_nv{args.n_v}_dt{args.dt}_niter{args.n_iter}_outdt{args.out_dt}')
    os.makedirs(data_dir, exist_ok=True)

    write_files(params, dir=data_dir, iproc=[4, 8])
    run_hypar(32, dir=data_dir)
    os.chdir('/usr/workspace/lei4/Projects/ARPAE/GPLaSDI/Vlasov1D1V')

    n_frames = params['n_iter'] // params['output_iter']
    fs, es, xs, vs = post_process_data(n_frames + 1, dir=data_dir)
    ts = params['output_iter'] * params['dt'] * np.arange(n_frames + 1)
    np.save(f'./{data_dir}/fs.npy', fs.reshape(fs.shape[0], args.n_v, args.n_x))
    np.save(f'./{data_dir}/es.npy', es)
    np.save(f'./{data_dir}/xs.npy', xs)
    np.save(f'./{data_dir}/vs.npy', vs)
    np.save(f'./{data_dir}/ts.npy', ts)
    os.system(f'rm -r {data_dir}/*.dat')
    plot_all(data_dir)
