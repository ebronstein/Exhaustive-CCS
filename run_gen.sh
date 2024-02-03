#!/bin/bash
#SBATCH --job-name=gen_extract
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=ebronstein@berkeley.edu
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --exclude=shadowfax,sunstone,smokyquartz,rainbowquartz,smaug

set -ex

source /accounts/projects/jsteinhardt/$(whoami)/.bashrc
cd /scratch/users/$(whoami)/Exhaustive-CCS
conda activate exhaustive_ccs

models=(deberta-v2-xxlarge-mnli)
for model in "${models[@]}"; do
    python generation_main.py --model $model --datasets all --cal_zeroshot 0 --swipe  --states_index -1 -3 -5 -7 -9 --prefix "normal-thatsright"
    python generation_main.py --model $model --datasets all --cal_zeroshot 0 --swipe  --states_index -1 -3 -5 -7 -9 --prefix "normal-mark"
done
