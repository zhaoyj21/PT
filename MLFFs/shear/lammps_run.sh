#!/bin/bash
#SBATCH -J shear
#SBATCH -p cnall
#SBATCH -N 1
#SBATCH --ntasks-per-node=56

source ~/WORK/cqian/.cq_env.sh 
module load compilers/gcc/v12.2.0 tools/cmake/v3.25.2 mpi/openmpi/v4.1.4

mpirun -np 56 lmp -in in.lmp 
