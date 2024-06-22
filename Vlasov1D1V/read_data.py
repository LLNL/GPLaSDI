import pdb

import numpy as np
import h5py
from solver import post_process_data

locking = True

def normalized_fft(signal, t0, tf, butterfly=False, remove_avg=False, axis=-1, shift=True, norm_freq=False):
    if remove_avg:
        signal = signal - np.mean(signal)

    if butterfly:
        signal = np.concatenate((signal[:-1], signal[::-1]))
        tf = 2 * tf - t0

    N = signal.shape[axis]
    T = tf - t0
    dt = T / N

    freq = np.linspace(-(N // 2), ((N - 1) // 2), N)
    if norm_freq:
        c_freq = 2 * np.pi / (dt * N)
        freq = freq * c_freq

    c_fft = dt / np.sqrt(2 * np.pi)
    fft = c_fft * np.fft.fft(signal, axis=axis)
    if shift:
        fft = np.fft.fftshift(fft, axes=axis)

    if butterfly:
        fft /= 2
        # fft = fft[N//2:]
        # freq = freq[N//2:]

    return freq, fft


def print_parameters(filename: str):
    hdf_file = h5py.File(filename, 'r')

    print('Parameters:')
    for k, v in hdf_file.attrs.items():
        print(f'{k:>10}: {v}')
    print()


def read_data(x, v, t, f, e) -> dict[str, np.ndarray]:
    n_t = f.shape[0]
    print(f'n_t: {n_t}')

    del_f = f - f[0]
    k_x, fhat = normalized_fft(f, x[0], x[-1], axis=2)
    del_fhat = fhat - fhat[0]

    k_x, E_hat = normalized_fft(e, x[0], x[-1], axis=1)

    t_plot = t
    data = {
        'x': x,
        'v': v,
        't': t,
        't_plot': t_plot,
        'f': f,
        'del_f': del_f,
        'fhat': fhat,
        'del_fhat': del_fhat,
        'E': e,
        'k_x': k_x,
        'E_hat': E_hat,
        # 'envelope': envelope,
        # 'stud': stud,
        # 'k_dr': k_dr,
        # 'omega_dr': omega_dr,
        # 'v_phase_dr': v_phase_dr
    }

    return data


def read_data_dat():
    fs, es, xs, vs = post_process_data(2501)
    np.save(f'./data/combined_data.npy', {'fs': fs, 'es': es, 'xs': xs, 'vs': vs}, allow_pickle=True)


if __name__ == '__main__':
    read_data_dat()