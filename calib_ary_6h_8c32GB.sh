#!/bin/bash

#SBATCH --job-name=calib_ary_6h_8c32GB                   #This is the name of your job
#SBATCH --cpus-per-task=8                  #This is the number of cores reserved
#SBATCH --mem-per-cpu=32G              #This is the memory reserved per core.
#Total memory reserved: 256GB

# Are you sure that you need THAT much memory?

#SBATCH --time=06:00:00        #This is the time that your task will run
#SBATCH --qos=6hours           #You will run in this queue

# Paths to STDOUT or STDERR files should be absolute or relative to current working directory
#SBATCH --output=myrun.o%j     #These are the STDOUT and STDERR files
#SBATCH --error=myrun.e%j

#You selected an array of jobs from 1 to 26 with 26 simultaneous jobs
#SBATCH --array=1-26%26
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=raul.hochuli@unibas.ch        #You will be notified via email when your task ends or fails

cd $HOME/OptimalPV_RH/

#Remember:
#The variable $TMPDIR points to the local hard disks in the computing nodes.
#The variable $HOME points to your home directory.
#The variable $SLURM_JOBID stores the ID number of your job.


#load your required modules below
#################################
ml purge
ml Python/3.11.5-GCCcore-13.2.0
source $HOME/OptimalPV_RH/.venv_optimalpv_rh/bin/activate


#export your required environment variables below
#################################################


#add your command lines below
#############################
$(head -$SLURM_ARRAY_TASK_ID calib_ary_kt_launch.cmd | tail -1)


