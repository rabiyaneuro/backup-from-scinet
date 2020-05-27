#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=01:30:00
#SBATCH --output=outputLOG%j.txt
 
cd $SLURM_SUBMIT_DIR

mkdir $SLURM_JOB_ID

# load the corresponding modules 
module load intel/2018.2 intelmpi/2018.2 intelpython3/2018.2

# Activate your environment
source activate ParPython


# run
mpirun -np 40 python quick_run_pli3.py





