#!/bin/sh

#SBATCH -N 1
#SBATCH -J data_vlasov
#SBATCH -p pbatch
#SBATCH -t 24:00:00
#SBATCH -A # bank_name


# cd path_to_project(Vlasov1D1V)
# activate python environment

v_drift=$1
T1=$2
T2=$3
k_Dr=$4
eps=$5
n_x=$6
n_v=$7
n_iter=$8
dt=$9
out_dt=${10}

python generate_single_data.py -v_drift $v_drift -T1 $T1 -T2 $T2 -k_Dr $k_Dr -eps $eps \
-n_x $n_x -n_v $n_v -n_iter $n_iter -dt $dt -out_dt $out_dt
