#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=ebronstein@berkeley.edu
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1

set -x

python sweep_extract.py
