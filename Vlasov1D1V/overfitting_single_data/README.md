## Generate new data in using Hypar
On Ruby pdebug node, run
```bash
python generate_single_data.py -v_drift 1.0 -T1 0.1 -T2 0.1 -k_Dr 1.11 -eps 0.1 -n_x 128 -n_v 128 -n_iter 25000 -dt 0.002 -out_dt 0.02
```
Data will be saved in `data_hypar/vd1.0_T0.1,0.1_kDr1.11_eps0.1_nx128_nv128_dt0.002_niter25000_outdt0.02`

To submit a job to pbatch, run
```bash
sbatch slurm/generate_single_data.slurm 1.0 0.1 0.1 1.11 0.1 128 128 25000 0.002 0.02
```

## Overfit single simulation using Autoencoder
On Lassen pdebug node, run
```bash
python compress_single_data/train_ae.py 
--data_dir ./data_hypar/vd1.0_T0.1,0.1_kDr1.11_eps0.1_nx128_nv128_dt0.002_niter25000_outdt0.02 --device 0 
--hdims '1000,500,100,100,100' --n_z 5 --lr 1e-4 --batch_size 32 --n_epochs 100000
```
Arguments:
- `data_dir`: directory of the data
- `batch_size`: batch size for training
- `lr`: learning rate
- `n_z`: latent dimension
- `hdims`: hidden dimensions of the autoencoder
- `n_iter`: number of training iterations
- `device`: GPU device number
- `debug`: debug mode, print info to terminal
- `eval`: evaluate the model after training

Submit a job to pbatch
```bash
sbatch slurm/overfit_single_ae.slurm
```

## Overfit single simulation using LaSDI (Autoencoder + SINDy)
On Lassen pdebug node, run
```bash
python overfitting_single_data/train_lasdi.py 
--data_dir ./data_hypar/vd1.0_T0.1,0.1_kDr1.11_eps0.1_nx128_nv128_dt0.002_niter25000_outdt0.02 --device 0 
--hdims '1000,200,50,50,50' --n_z 5 --poly 1 --n_iter 300000 --Dt 0.02
```
Arguments:
- `data_dir`: directory of the data
- `lr`: learning rate
- `n_z`: latent dimension
- `hdims`: hidden dimensions of the autoencoder
- `n_iter`: number of training iterations
- `Dt`: time step of the data
- `sindy_weight`: weight of the SINDy loss
- `coef_weight`: weight of the coefficient loss
- `poly_order`: order of the polynomial SINDy library
- `device`: GPU device number
- `debug`: debug mode, print info to terminal
- `eval`: evaluate the model after training

Submit a job to pbatch
```bash
sbatch slurm/overfit_single_LaSDI.slurm
```
