#!/bin/bash

#The name of the job is test_job
#SBATCH -J mt_tut

#The job requires 1 compute node
#SBATCH -N 1

#The job requires 1 task per node
#SBATCH --ntasks-per-node=1

#The maximum walltime of the job is a 8 days
#SBATCH -t 24:05:00

#SBATCH --mem=25G

#Leave this here if you need a GPU for your job
#SBATCH --partition=gpu

#SBATCH --gres=gpu:tesla:1

#SBATCH --exclude=falcon3

# OUR COMMANDS GO HERE

module load python/3.6.3/CUDA-8.0

source activate konstrukt-cuda8

allennlp train --include-package my_library experiments/softmax.json -s models/softmax_params1
