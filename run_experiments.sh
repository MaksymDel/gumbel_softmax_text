#!/bin/bash

for file in scripts_run_experiments/*
do
  sbatch scripts_run_experiments/$file
done