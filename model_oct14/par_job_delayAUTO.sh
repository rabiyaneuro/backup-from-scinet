#!/bin/bash


#SBATCH --output=outputLOG%j.txt

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=rabiya.noori@gmail.com

cd $SLURM_SUBMIT_DIR

# Create a directory to hold the solution files
mkdir $SLURM_JOB_ID
mkdir $SLURM_JOB_ID/scratch_


# load the corresponding modules 
# module load NiaEnv/2018a intel/2018.2 intelmpi/2018.2 intelpython3/2018.2
# module load NiaEnv/2019b intelpython3/2019u3 intel/2019u3 intelmpi/2019u3
module load NiaEnv/2019b python intel/2019u3 intelmpi/2019u3


# Activate your environment
# source activate ParPython
source $HOME/WM/bin/activate



# Run the main simulation program, it saves the results in the directory made earlier, it
# also loads parameters from the appropriate file
# 1st arg is the variable script, 2nd arg is the job_id
# test - mpirun -np 10 python run_delay_scinet_pli_corr.py load_genvars_delays_pli_corr2.py 111 

mpirun -np $SLURM_NTASKS python ${RUN_FILE} ${VARS} $SLURM_JOB_ID

# Creating a text file that holds the parameters used in this script
echo "--nodes = ${SLURM_JOB_NUM_NODES} --ntasks= ${SLURM_NTASKS} --param_file = ${VARS} job_id = ${SLURM_JOB_ID}" >> params${SLURM_JOB_ID}.txt

# Moving the text file params${SLURM_JOB_ID}.txt and also the main run file ${RUN_FILE} into the directory for this
# specific job

mv  params${SLURM_JOB_ID}.txt $SLURM_JOB_ID
cp ${RUN_FILE} $SLURM_JOB_ID
cp ${VARS} $SLURM_JOB_ID
mv ${VARS} VARS_files/vars${SLURM_JOB_ID}.py
cp outputLOG${SLURM_JOB_ID}.txt $SLURM_JOB_ID

# make directory of todays date if it doesn't exist
mkdir -p Output/`date +%Y-%m-%d`

# move everything into the appropriate directory 
mv $SLURM_JOB_ID Output/`date +%Y-%m-%d`


# dos2unix par_job.sh
# qsub -v par_name=par_value[,par_name=par_value...] script.sh
# qsub -v NODES=1,CORES=40, TIME="01:30:00", RUN_FILE="run_corr_scinet", vars=1  par_job2.sh





