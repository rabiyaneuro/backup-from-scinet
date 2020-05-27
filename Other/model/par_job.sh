#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks=80
#SBATCH --time=1:30:00
#SBATCH --job-name mpi_job1
#SBATCH --output=%j/outputLOG%j.txt
 
cd $SLURM_SUBMIT_DIR

# load the corresponding modules 
module load intel/2018.2 intelmpi/2018.2 intelpython3/2018.2

# Activate your environment
# source activate ParPython

# UPDATE NAME OF quick_run.py IN THIS SECTION
fname = "quick_run.py"

# run
mpirun -np 80 python ${fname}

cp ${fname} %j/quick_run%j.py



