import torch
import time
import numpy as np
import sys, os
import argparse

from overfitting_single_data.autoencoder import Autoencoder
from overfitting_single_data.train_utils import ScoreMeter, Logger, CSVWriter
from overfitting_single_data.plot_latent import plot_z_hat
from utils.utils_sindy import *


class TrainLaSDI:
    def __init__(self,
                 data_dir,
                 save_dir,
                 hidden_units=(1000, 200, 50, 50, 50),
                 n_z=5,
                 lr=1e-4,
                 n_iter=300000,
                 Dt=0.02,
                 poly_order=1,
                 sine_term=False,
                 sindy_weight=0.1,
                 coef_weight=1e-5,
                 device=0,
                 seed=42):

        self.set_seed(seed)
        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
        X_train = np.load(f'{data_dir}/fs.npy')
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        self.X_train = X_train
        time_dim = X_train.shape[0]
        space_dim = X_train.shape[1]
        self.X_train = self.X_train[None, ...]
        self.data_dir = data_dir
        self.checkpoint_path = f'{save_dir}/checkpoint.pt'
        self.save_dir = save_dir

        self.autoencoder = Autoencoder(space_dim, hidden_units, n_z).to(self.device)
        self.time_dim = time_dim
        self.space_dim = space_dim
        self.n_z = n_z

        self.n_iter = n_iter
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        self.MSE = torch.nn.MSELoss()

        self.Dt = Dt
        self.t_grid = np.arange(0, time_dim) * Dt
        self.poly_order = poly_order
        self.lib_size = lib_size(n_z, poly_order)
        self.sine_term = sine_term
        self.sindy_weight = sindy_weight
        self.coef_weight = coef_weight

    def train(self):
        autoencoder = self.autoencoder
        optimizer = self.optimizer

        tic_start = time.time()
        start_train_phase = []
        checkpoint_history = []

        start_train_phase.append(tic_start)
        n_greedy = 500
        score_meter = ScoreMeter(['loss', 'loss_ae', 'loss_sindy', 'loss_coef', 'rel_error'])
        writer = CSVWriter(f'{self.save_dir}/train.csv', 'iter, loss, rel_error')

        autoencoder.train()
        X_train = self.X_train.to(self.device)

        for iter in range(self.n_iter):
            optimizer.zero_grad()
            Z = autoencoder.encoder(X_train)
            X_pred = autoencoder.decoder(Z)
            # Z = Z.cpu()

            loss_ae = self.MSE(X_train, X_pred)
            loss_sindy, loss_coef, sindy_coef = find_sindy_coef(Z, self.Dt, 1, self.MSE,
                                                                self.poly_order, self.lib_size,
                                                                self.sine_term)
            max_coef = np.abs(np.array(sindy_coef)).max()
            loss = loss_ae + self.sindy_weight * loss_sindy + self.coef_weight * loss_coef

            loss.backward()
            optimizer.step()

            relative_error = torch.norm(X_train - X_pred) / torch.norm(X_train)
            score_meter.update(np.array([loss.item(),
                                         loss_ae.item(),
                                         loss_sindy.item(),
                                         loss_coef.item(),
                                         relative_error.item()]))
            iter += 1
            if iter % 100 == 0:
                print(f"Iter {iter:06d}/{self.n_iter} | {score_meter.stats_string()} | "
                      f"time {time.time() - tic_start:.2f}", flush=True)
                torch.save(autoencoder.state_dict(), self.checkpoint_path)
                best_loss = loss.item()
                checkpoint_history.append(iter)
                writer.write(f'{iter}, {loss.item()}, {relative_error.item()}')
                score_meter.reset()
            if iter >= self.n_iter:
                break

        tic_end = time.time()
        total_time = tic_end - tic_start
        print("Total time: %3.2f" % total_time)

    @torch.no_grad()
    def eval(self):
        self.autoencoder.load_state_dict(torch.load(self.checkpoint_path))
        self.autoencoder.eval()
        Z = self.autoencoder.encoder(self.X_train)

        _, _, sindy_coef = find_sindy_coef(Z, self.Dt, 1, self.MSE,
                                                            self.poly_order, self.lib_size,
                                                            self.sine_term)
        Z0 = self.autoencoder.encoder(self.X_train[0, 0, :][None, ...]).cpu().numpy()
        Z_hat = simulate_sindy(sindy_coef, Z0, self.t_grid, self.n_z, self.poly_order, self.lib_size, self.sine_term)
        X_hat = self.autoencoder.decoder(torch.Tensor(Z_hat).to(self.device))
        loss = self.MSE(self.X_train, X_hat)
        mse = np.mean(((self.X_train - X_hat) ** 2).detach().cpu().numpy())
        rel_error = torch.norm(self.X_train - X_hat) / torch.norm(self.X_train)
        print(f'Recon error | loss {loss} | MSE {mse} | Relative error {rel_error.item()}')
        eval_writer = CSVWriter(f'{self.save_dir}/eval.csv', 'loss, rel_error')
        eval_writer.write(f'{loss.item()}, {rel_error.item()}')
        np.save(f'{self.save_dir}/z_hat.npy', Z_hat[0])
        np.save(f'{self.save_dir}/z.npy', Z[0].detach().cpu().numpy())
        np.save(f'{self.save_dir}/f_hat.npy', X_hat[0].detach().cpu().numpy())
        plot_z_hat(self.save_dir)

    def set_seed(self, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_z', type=int, default=5)
    parser.add_argument('--hdims', type=str, default='1000,200,50,50,50')
    parser.add_argument('--n_iter', type=int, default=300000)
    parser.add_argument('--Dt', type=float, default=0.02)
    parser.add_argument('--sindy_weight', type=float, default=0.1)
    parser.add_argument('--coef_weight', type=float, default=1e-5)
    parser.add_argument('--poly_order', type=int, default=1)
    parser.add_argument('--device', type=str, default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    expr_name = f'hdims{args.hdims}_nz{args.n_z}_sindyw{args.sindy_weight}_coefw{args.coef_weight}_poly{args.poly_order}_lr{args.lr}_niter{args.n_iter}_Dt{args.Dt}'

    if args.debug:
        out_dir = f'{args.data_dir}/LaSDI/debug_{expr_name}'
        os.makedirs(out_dir, exist_ok=True)
        sys.stdout = Logger(f'{out_dir}/log.txt')
    else:
        out_dir = f'{args.data_dir}/LaSDI/{expr_name}'
        os.makedirs(out_dir, exist_ok=True)
        sys.stdout = open(f'{out_dir}/log.txt', 'w')

    trainer = TrainLaSDI(data_dir=args.data_dir,
                         save_dir=out_dir,
                         hidden_units=[int(h) for h in args.hdims.split(',')],
                         n_z=args.n_z,
                         lr=args.lr,
                         Dt=args.Dt,
                         n_iter=args.n_iter,
                         sindy_weight=args.sindy_weight,
                         coef_weight=args.coef_weight,
                         device=args.device,
                         poly_order=args.poly_order,
                         sine_term=False)
    if args.eval:
        trainer.eval()
    else:
        trainer.train()
        trainer.eval()
