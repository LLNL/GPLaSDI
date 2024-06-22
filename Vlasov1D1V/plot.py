from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
from matplotlib import ticker
from read_data import normalized_fft
import argparse
from datetime import datetime
import pickle

from read_data import read_data


def E_field_movie(data, folder='../figures', fps=30, anim_time=10, title=''):
    if not os.path.exists(folder):
        os.makedirs(folder)

    x = data['x']
    t = data['t']
    Ex = data['E']

    min_Ex, max_Ex = 1.2 * Ex.min(), 1.2 * Ex.max()

    # Initial plot
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(title, y=0.9)
    fig.subplots_adjust(top=0.8)

    l = ax.plot(x, Ex[0])[0]
    ax.set_title('Electric field at t=0.0')
    ax.set_xlabel('x')
    ax.set_ylabel('E(x, t)')
    ax.set_ylim(min_Ex, max_Ex)
    # plot max and min of E
    ax.axhline(y=Ex.min(), color='r', linestyle='--')
    ax.axhline(y=Ex.max(), color='r', linestyle='--')

    def animate(frame):
        l.set_ydata(Ex[frame])
        ax.set_title(f'Electric field at t={t[frame]:.1f}')

    n_frames = min(int(fps * anim_time), len(t))
    fps = n_frames / anim_time
    interval = int(1000 / fps)
    frames = np.linspace(0, len(t) - 1, n_frames).astype(int)
    anim = FuncAnimation(fig, animate, frames=(frames), interval=interval)

    anim.save(f'{folder}/E_field.mp4', writer='ffmpeg', fps=fps)


def E_hat_movie(data, folder='../figures', fps=30, anim_time=10, k_max=10, title=''):
    if not os.path.exists(folder):
        os.makedirs(folder)

    k = data['k_x']
    t = data['t']
    E_hat = np.abs(data['E_hat'])

    min_Ex, max_Ex = 0, 1.2 * E_hat.max()

    # Initial plot
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(title, y=0.9)
    fig.subplots_adjust(top=0.8)

    # l = ax.plot(k, E_hat[0])[0]
    # make stem plot
    ax.stem(k, E_hat[0])
    ax.set_title('t=0.0')
    ax.set_xlabel('k_x')
    ax.set_ylabel('$\\hat{E}(x, t)$')
    ax.set_ylim(min_Ex, max_Ex)
    ax.set_xlim(0, k_max)
    ax.axhline(y=E_hat.max(), color='r', linestyle='--')

    def animate(frame):
        # l.set_ydata(E_hat[frame])
        ax.cla()
        ax.stem(k, E_hat[frame])
        ax.set_title('t=0.0')
        ax.set_xlabel('x')
        ax.set_ylabel('$\\hat{E}(x, t)$')
        ax.set_ylim(min_Ex, max_Ex)
        ax.set_xlim(0, k_max)
        ax.axhline(y=E_hat.max(), color='r', linestyle='--')
        ax.set_title(f't={t[frame]:.1f}')

    n_frames = min(int(fps * anim_time), len(t))
    fps = n_frames / anim_time
    interval = int(1000 / fps)
    frames = np.linspace(0, len(t) - 1, n_frames).astype(int)
    anim = FuncAnimation(fig, animate, frames=(frames), interval=interval)

    anim.save(f'{folder}/E_hat.mp4', writer='ffmpeg', fps=fps)


def rho_hat_plot(data, k, folder='../figures', title=''):
    if not os.path.exists(folder):
        os.makedirs(folder)

    if isinstance(k, int):
        k = range(1, k + 1)

    t = data['t']
    k_x = data['k_x']
    E_hat = np.abs(np.fft.ifftshift(data['E_hat'], axes=1))

    # envelope = data['envelope']
    # stud = data['stud']

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    fig.suptitle(title, y=0.9)
    fig.subplots_adjust(top=0.8)

    lines = []
    col = ['r', 'b', 'orange', 'g']
    ax2 = axes.twinx()
    for i, kk in enumerate(k):
        lines.append(axes.plot(t, kk * E_hat[:, kk], label=f'$\\hat{{\\rho}}^{{({kk})}}(t)$')[0])
        axes.set_title(f'k = {kk}')
        axes.set_xlabel('t')

    # ax2.plot(t, (envelope * stud) ** 2, label='envelope', color='black', ls='--')

    axes.legend()

    fig.tight_layout()
    fig.savefig(f'{folder}/E_hat.pdf')


def rho_hat_hat_plot(data, k, folder='../figures', t_start=500, title=''):
    if not os.path.exists(folder):
        os.makedirs(folder)

    if isinstance(k, int):
        k = range(1, k + 1)

    t = data['t']
    k_x = data['k_x']
    E_hat = np.abs(np.fft.ifftshift(data['E_hat'], axes=1))

    t_sel = t > t_start

    envelope = data['envelope']
    stud = data['stud']

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    fig.suptitle(title)
    lines = []
    col = ['r', 'b', 'orange', 'g']
    ax2 = axes.twinx()
    for i, kk in enumerate(k):
        freq, rho_hat_hat = normalized_fft(kk * E_hat[t_sel, kk], t[t_sel][0], t[t_sel][-1], butterfly=True,
                                           norm_freq=True, remove_avg=True)
        lines.append(
            axes.plot(freq, np.abs(rho_hat_hat), label=f'$\\hat{{\\hat{{\\rho}}}}^{{({kk})}}(t)$', color=col[i])[0])
        axes.set_xlabel('$\\omega$')

    # Set grid
    axes.grid()
    axes.set_xlim(0, 0.5)
    axes.legend()

    fig.tight_layout()
    fig.savefig(f'{folder}/E_hat.pdf')


def f_movie(data, delta=False, folder='../figures', fps=30, anim_time=10, vlim=None, log=False, title=''):
    if not os.path.exists(folder):
        os.makedirs(folder)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    fig.suptitle(title, y=0.9)
    fig.subplots_adjust(top=0.8)

    t = data['t_plot']
    x = data['x']
    v = data['v']
    if delta:
        f = data['del_f']
    else:
        f = data['f']

    if vlim:
        v_sel = np.logical_and(v > vlim[0], v < vlim[1])
    else:
        v_sel = np.ones_like(v, dtype=bool)

    f_min, f_max = f[:, v_sel, :].min(), f[:, v_sel, :].max()

    # ctrf = ax.contourf(x, v[v_sel], f[0, v_sel]+1e-10, cmap='viridis', locator=ticker.LogLocator(), vmin=f_min, vmax=f_max)
    if log:
        levels = np.logspace(np.log10(f_min), np.log10(f_max), 100)
    else:
        levels = np.linspace(f_min, f_max, 100)

    ctrf = ax.contourf(x, v[v_sel], f[0, v_sel], cmap='viridis', levels=levels)
    if vlim:
        ax.set_ylim(vlim)
    ax.set_xlabel('x')
    ax.set_ylabel('v')
    if delta:
        title = lambda t: f'$\\delta f(x, v, t={t:.1f})$'
    else:
        title = lambda t: f'$f(x, v, t={t:.1f})$'

    ax.set_title(title(0.0))
    fig.colorbar(ctrf, ax=ax)

    def animate(frame, ctrf):
        # ctrf.set_paths([])
        # ctrf = ax.contourf(x, v[v_sel], f[frame,v_sel, :]+1e-10, cmap='viridis', locator=ticker.LogLocator(), vmin=f_min, vmax=f_max)
        ax.cla()
        ctrf = ax.contourf(x, v[v_sel], f[frame, v_sel, :], cmap='viridis', levels=levels)
        # print(f'frame {frame}')
        if vlim:
            ax.set_ylim(vlim)
        ax.set_title(title(t[frame]))

    n_frames = min(int(fps * anim_time), len(t))
    fps = n_frames / anim_time
    interval = int(1000 / fps)
    frames = np.linspace(0, len(t) - 1, n_frames).astype(int)
    anim = FuncAnimation(fig, animate, frames=(frames), interval=interval, fargs=(ctrf,))

    if delta:
        filename = f'{folder}/delta_f_movie.mp4'
    else:
        filename = f'{folder}/f_movie.mp4'

    anim.save(filename, writer='ffmpeg', fps=fps)


def f_hat_0_movie(data, delta=False, folder='../figures', fps=30, anim_time=10, xlim=None, title=''):
    if not os.path.exists(folder):
        os.makedirs(folder)

    k_x = data['k_x']
    t = data['t_plot']
    v = data['v']
    if delta:
        f_hat = data['del_fhat']
    else:
        f_hat = data['fhat']

    min_fhat, max_fhat = np.abs(f_hat).min(), np.abs(f_hat).max()

    idx_k_0 = np.argmin(np.abs(k_x))

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    fig.suptitle(title, y=0.9)
    fig.subplots_adjust(top=0.8)

    l = ax.plot(v, np.abs(f_hat[0, :, idx_k_0]))[0]
    if xlim:
        ax.set_xlim(xlim)

    ax.set_xlabel('v')
    if delta:
        ax.set_ylabel('$\\hat{\\delta f}^{(0)}(v, t)$')
    else:
        ax.set_ylabel('$\\hat{f}^{(0)}(v, t)$')
    ax.set_ylim(min_fhat, max_fhat)
    ax.set_title('t=0.0')

    def animate(frame):
        ax.cla()
        l.set_ydata(np.abs(f_hat[frame, :, idx_k_0]))
        ax.set_title(f't={t[frame]:.1f}')

    n_frames = min(int(fps * anim_time), len(t))
    fps = n_frames / anim_time
    interval = int(1000 / fps)
    frames = np.linspace(0, len(t) - 1, n_frames).astype(int)
    anim = FuncAnimation(fig, animate, frames=(frames), interval=interval)

    if delta:
        filename = f'{folder}/delta_f_hat_0.mp4'
    else:
        filename = f'{folder}/f_hat_0.mp4'
    anim.save(filename, writer='ffmpeg', fps=fps)


def source_plot(data, folder='../figures', intesity=True):
    if not os.path.exists(folder):
        os.makedirs(folder)

    t = data['t']
    envelope = data['envelope']
    stud = data['stud']

    fig, ax = plt.subplots(figsize=(10, 5))

    if intesity:
        v = (stud * envelope) ** 2
    else:
        v = stud * envelope
    ax.plot(t, v)
    ax.set_xlabel('t')

    fig.savefig(f'{folder}/source.pdf')


def plot_all(folder_data):
    fs = np.load(f'{folder_data}/fs.npy')
    es = np.load(f'{folder_data}/es.npy')
    xs = np.load(f'{folder_data}/xs.npy')
    vs = np.load(f'{folder_data}/vs.npy')
    ts = np.load(f'{folder_data}/ts.npy')
    fs = np.nan_to_num(fs)
    es = np.nan_to_num(es)
    print(ts.shape)
    assert fs.shape[0] == ts.shape[0]

    print(f'Starting making plots at {datetime.now()}')

    print('  Reading data ...')
    data = read_data(xs, vs, ts, fs, es)

    folder_plots = f'{folder_data}/plots'
    os.makedirs(folder_plots, exist_ok=True)

    pickle.dump(data, open(f'{folder_data}/data.pkl', 'wb'))

    # print('  Plotting source ...')
    # source_plot(data, folder=folder_plots)
    title = folder_data.split('/')[-1]
    print('  Plotting E_field_movie ...')
    E_field_movie(data, folder=folder_plots, fps=30, anim_time=20, title=title)
    print('  Plotting E_hat_movie ...')
    E_hat_movie(data, folder=folder_plots, fps=30, anim_time=20, k_max=10, title=title)

    print('  Plotting rho_hat_plot ...')
    rho_hat_plot(data, k=10, folder=folder_plots, title=title)

    # v_phase = data['v_phase_dr']
    print('  Plotting f_movie ...')
    f_movie(data, delta=False, vlim=[-3.5, 3.5], log=False, folder=folder_plots, anim_time=20, title=title)
    print('  Plotting df_movie ...')
    f_movie(data, delta=True, vlim=[-3.5, 3.5], log=False, folder=folder_plots, anim_time=20, title=title)

    print('  Plotting f_hat_0_movie ...')
    f_hat_0_movie(data, delta=False, folder=folder_plots, anim_time=20, title=title)
    print('  Plotting df_hat_0_movie ...')
    f_hat_0_movie(data, delta=True, folder=folder_plots, anim_time=20, title=title)
