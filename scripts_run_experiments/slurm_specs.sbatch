#!/bin/bash

# DO NOT FORGET TO DO """chmod 755 slurm_specs.sbatch""" before

#The name of the job is test_job
#SBATCH -J diff_nmt

#The job requires 1 compute node
#SBATCH -N 1

#The job requires 1 task per node
#SBATCH --ntasks-per-node=1

#The maximum walltime of the job is a 5 hours
#SBATCH -t 05:00:00

#SBATCH --mem=7G

#Leave this here if you need a GPU for your job
#SBATCH --partition=gpu

#SBATCH --gres=gpu:tesla:1

#SBATCH --exclude=falcon3

# OUR COMMANDS GO HERE

module load python/3.6.3/CUDA-8.0

source activate konstrukt-cuda8
