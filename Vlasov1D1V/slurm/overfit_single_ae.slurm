#!/usr/bin/env bash
#BSUB -G # bank_name
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 12:00

# cd path_to_project(Vlasov1D1V)
# activate python environment

DATA_DIR=... # path to data
DEVICE=0

python compress_single_data/train_ae.py --data_dir $DATA_DIR --device $DEVICE \
--hdims '1000,500,100,100,100' --n_z 5 --lr 1e-4 --batch_size 32 --n_epochs 100000 \
