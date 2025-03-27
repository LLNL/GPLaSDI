#!/bin/bash
#SBATCH -N 1
#SBATCH -t 06:00:00

chmod +x generate_data.py
chmod +x train_model1.py

#source  /usr/workspace/$USER/local/$SYS_TYPE/GPlasdi_venv/bin/activate
#source /collab/usr/gapps/python/toss_4_x86_64_ib/anaconda3-2023.03/bin/activate
#conda activate GPLaSDI_updated
source /p/lustre1/wilander/store_venvs/GPLaSDI_updated/bin/activate

#./generate_data.py;
./train_model1.py;
