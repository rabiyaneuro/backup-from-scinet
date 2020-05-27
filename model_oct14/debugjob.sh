#!/bin/bash

# load the corresponding modules 
module load NiaEnv/2019b python intel/2019u3 intelmpi/2019u3


# Activate your environment
#source $SCRATCH/TEST/bin/activate
source $HOME/WM/bin/activate


# Run the main simulation program, it saves the results in the directory made earlier, it
# also loads parameters from the appropriate file
# 1st arg is the variable script, 2nd arg is the job_id
mpirun -np $1 python $2 $3 111 > $4


# dos2unix par_job.sh
# qsub -v par_name=par_value[,par_name=par_value...] script.sh
# qsub -v NODES=1,CORES=40, TIME="01:30:00", RUN_FILE="run_corr_scinet", vars=1  par_job2.sh





