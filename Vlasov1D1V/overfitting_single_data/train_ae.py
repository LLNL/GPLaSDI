import torch
import time
import numpy as np
import sys, os
import argparse
from torch.utils.data import DataLoader, TensorDataset

from overfitting_single_data.autoencoder import Autoencoder
from overfitting_single_data.train_utils import ScoreMeter, Logger, CSVWriter


class TrainAE:
    def __init__(self,
                 data_dir,
                 out_dir,
                 hidden_units=(1000, 200, 50, 50, 50),
                 n_z=5,
                 lr=1e-4,
                 n_iter=300000,
                 batch_size=16,
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
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.checkpoint_path = f'{out_dir}/checkpoint.pt'
        self.train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)

        self.autoencoder = Autoencoder(space_dim, hidden_units, n_z).to(self.device)
        self.time_dim = time_dim
        self.space_dim = space_dim
        self.n_z = n_z

        self.n_iter = n_iter
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        self.MSE = torch.nn.MSELoss()

    def train(self):
        autoencoder = self.autoencoder
        optimizer = self.optimizer
        tic_start = time.time()
        start_train_phase = []
        checkpoint_history = []

        start_train_phase.append(tic_start)
        iter = 1
        n_epochs = 1e8
        score_meter = ScoreMeter(['loss', 'rel_error'])
        writer = CSVWriter(f'{self.out_dir}/train.csv', 'iter, loss, rel_error')

        autoencoder.train()
        for epoch in range(int(n_epochs)):
            for i, (X_train,) in enumerate(self.train_loader):
                X_train = X_train.to(self.device)
                optimizer.zero_grad()
                Z = autoencoder.encoder(X_train)
                X_pred = autoencoder.decoder(Z)

                loss = self.MSE(X_train, X_pred)
                relative_error = torch.norm(X_train - X_pred) / torch.norm(X_train)

                loss.backward()
                optimizer.step()
                score_meter.update(np.array([loss.item(), relative_error.item()]))
                if iter % 100 == 0:
                    print(f"Iter {iter:06d}/{self.n_iter} | {score_meter.stats_string()} | "
                          f"time {time.time() - tic_start:.2f}", flush=True)
                    torch.save(autoencoder.state_dict(), self.checkpoint_path)
                    best_loss = loss.item()
                    checkpoint_history.append(iter)
                    writer.write(f'{iter}, {loss.item()}, {relative_error.item()}')
                    score_meter.reset()
                iter += 1
                if iter > self.n_iter:
                    break
            if iter > self.n_iter:
                break

        tic_end = time.time()
        total_time = tic_end - tic_start
        print("Total time: %3.2f" % total_time)

    def eval(self):
        self.autoencoder.load_state_dict(torch.load(self.checkpoint_path))
        self.autoencoder.eval()
        X_train = np.load(f'{self.data_dir}/fs.npy')
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        Z = self.autoencoder.encoder(X_train)
        X_pred = self.autoencoder.decoder(Z)
        loss = self.MSE(X_train, X_pred)
        mse = np.mean(((X_train - X_pred) ** 2).detach().cpu().numpy())
        rel_error = torch.norm(X_train - X_pred) / torch.norm(X_train)
        print(f'Recon error | loss {loss} | MSE {mse} | Relative error {rel_error.item()}')
        eval_writer = CSVWriter(f'{self.out_dir}/eval.csv', 'loss, rel_error')
        eval_writer.write(f'{loss.item()}, {rel_error.item()}')

    def set_seed(self, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_z', type=int, default=5)
    parser.add_argument('--hdims', type=str, default='1000,200,50,50,50')
    parser.add_argument('--n_iter', type=int, default=100000)
    parser.add_argument('--device', type=str, default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    expr_name = f'hdims{args.hdims}_nz{args.n_z}_lr{args.lr}_bs{args.batch_size}_niter{args.n_iter}'

    if args.debug:
        out_dir = f'{args.data_dir}/autoencoder/debug_{expr_name}'
        os.makedirs(out_dir, exist_ok=True)
        sys.stdout = Logger(f'{out_dir}/log.txt')
    else:
        out_dir = f'{args.data_dir}/autoencoder/{expr_name}'
        os.makedirs(out_dir, exist_ok=True)
        sys.stdout = open(f'{out_dir}/log.txt', 'w')

    trainer = TrainAE(data_dir=args.data_dir,
                      out_dir=out_dir,
                      lr=args.lr,
                      batch_size=args.batch_size,
                      hidden_units=[int(h) for h in args.hdims.split(',')],
                      n_z=args.n_z,
                      n_iter=args.n_iter,
                      device=args.device)
    if args.eval:
        trainer.eval()
    else:
        trainer.train()
        trainer.eval()
