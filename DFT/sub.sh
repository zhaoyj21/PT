#!/bin/bash
#SBATCH -p v6_384
#SBATCH -J crss-vasp
#SBATCH -N 3
#SBATCH -n 288
export PYTHONUNBUFFERED=1
source /public1/soft/modules/module.sh
module load mpi/intel/17.0.5-cjj
export PATH=/public1/home/sch2327/zyj/software/vasp.5.4.4/bin:$PATH

module load anaconda
source activate drain
python ./strength.py

