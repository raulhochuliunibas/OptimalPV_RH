#!/bin/bash

#SBATCH --job-name=OPT_LRG_30m16c16G                   #This is the name of your job
#SBATCH --cpus-per-task=16                  #This is the number of cores reserved
#SBATCH --mem-per-cpu=16G              #This is the memory reserved per core.
#Total memory reserved: 256GB

# Are you sure that you need THAT much memory?

#SBATCH --time=00:30:00        #This is the time that your task will run
#SBATCH --qos=30min           #You will run in this queue

# Paths to STDOUT or STDERR files should be absolute or relative to current working directory
#SBATCH --output=stdouterr_files/myrun.o%A_%a     #These are the STDOUT and STDERR files
#SBATCH --error=stdouterr_files/myrun.e%A_%a

#You selected an array of jobs from 1 to 350 with 50 simultaneous jobs
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
export SLURM_ARRAY_TASK_ID_ENV=$SLURM_ARRAY_TASK_ID
export SLURM_ARRAY_JOB_ID_ENV=$SLURM_ARRAY_JOB_ID   


#add your command lines below
#############################
python main_OPTIM_LRG.py



