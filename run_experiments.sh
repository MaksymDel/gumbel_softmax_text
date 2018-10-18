#!/bin/bash

for experiment in scripts_run_experiments/*
do
  sbatch $experiment
done