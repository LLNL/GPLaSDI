import pdb

from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np


def plot_z_hat(data_dir):
    z = np.load(f'{data_dir}/z.npy')
    z_hat = np.load(f'{data_dir}/z_hat.npy')
    ts = np.load(f"{'/'.join(data_dir.split('/')[:-2])}/ts.npy")
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), dpi=150)
    ax[0].plot(ts, z)
    ax[0].set_ylabel('Z')
    ax[0].set_xlabel('Time')
    ax[1].plot(ts, z_hat)
    ax[1].set_ylabel('Z_hat')
    ax[1].set_xlabel('Time')
    plt.savefig(f'{data_dir}/z_hat.png')


def animate_f_hat(data_dir):
    f_hat = np.load(f'{data_dir}/f_hat.npy')
    f_hat = f_hat.reshape(f_hat.shape[0], 128, 128)
    fs = np.load(f"{'/'.join(data_dir.split('/')[:-2])}/fs.npy") # (n_t, n_x, n_v)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), dpi=150)
    ax[0].set_title('f_hat')
    ax[1].set_title('f')
    ims = []
    for i in range(f_hat.shape[0]):
        im0 = ax[0].imshow(f_hat[i], cmap='viridis', animated=True)
        im1 = ax[1].imshow(fs[i], cmap='viridis', animated=True)
        ims.append([im0, im1])
    ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True)
    ani.save(f'{data_dir}/f_hat.gif', writer='ffmpeg', fps=30)


if __name__ == '__main__':
    data_dir = './data_hypar/vd1.0_T0.1,0.1_kDr1.11_eps0.1_nx128_nv128_dt0.002_niter25000_outdt0.02/LaSDI/debug_hdims1000,500,100,100,100_nz3_coefw1e-05_poly1_lr0.0001_bs32_niter300000'
    animate_f_hat(data_dir)