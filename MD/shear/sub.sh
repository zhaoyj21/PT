#!/bin/bash
#SBATCH -J shear
#SBATCH -p cnall
#SBATCH -N 1
#SBATCH --ntasks-per-node=56

module load compilers/intel/oneapi-2023/config
module load compilers/gcc/v12.2.0
mpirun -n 56 /apps/soft/lammps/lammps-22Dec2022/src/lmp_oneapi -in in.lmp
chmod 660 *

