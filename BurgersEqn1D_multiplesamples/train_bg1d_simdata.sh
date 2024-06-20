#!/bin/bash
#SBATCH -N 1
#SBATCH -t 0:20:00

chmod +x generate_data_sim.py

#source  /usr/workspace/$USER/local/$SYS_TYPE/GPlasdi_venv/bin/activate
source /collab/usr/gapps/python/toss_4_x86_64_ib/anaconda3-2023.03/bin/activate
conda activate GPLaSDI

./generate_data_sim.py;