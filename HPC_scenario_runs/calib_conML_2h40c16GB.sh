#!/bin/bash

#SBATCH --job-name=calib_conML2h40c16GB                   #This is the name of your job
#SBATCH --cpus-per-task=40                  #This is the number of cores reserved
#SBATCH --mem-per-cpu=16G              #This is the memory reserved per core.
#Total memory reserved: 640GB

# Are you sure that you need THAT much memory?

#SBATCH --time=02:00:00        #This is the time that your task will run
#SBATCH --qos=6hours           #You will run in this queue

# Paths to STDOUT or STDERR files should be absolute or relative to current working directory
#SBATCH --output=data/calibration/myrun.o%j     #These are the STDOUT and STDERR files
#SBATCH --error=data/calibration/myrun.e%j
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
python src/calibration_array_by_bfs_concat.py


