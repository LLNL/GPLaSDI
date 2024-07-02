#!/usr/bin/env bash
#BSUB -G # bank_name
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 12:00

# cd path_to_project(Vlasov1D1V)
# activate python environment

DATA_DIR=... # path to data
DEVICE=0

python compress_single_data/train_lasdi.py --data_dir $DATA_DIR --device $DEVICE \
--sindy_weight 0.2 --coef_weight 1e-3 \
--hdims '1000,500,100,100,100' --n_z 5 --poly_order 1 --Dt 0.02 \
--lr 1e-5 --n_iter 300000 &
