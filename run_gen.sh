#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=ebronstein@berkeley.edu
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --nodelist=airl.ist.berkeley.edu,sac.ist.berkeley.edu,cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu
#SBATCH --nodes=1
#SBATCH --qos=scavenger

set -ex

cd /nas/ucb/ebronstein/Exhaustive-CCS
eval "$(/nas/ucb/ebronstein/anaconda3/bin/conda shell.bash hook)"
conda activate exhaustive-ccs

models=(/nas/ucb/nlauffer/datasets/llama-2-7b-chat/)
for model in "${models[@]}"; do
    # python generation_main.py --model $model --datasets all --cal_zeroshot 0 --swipe  --states_index -3 -5 -7 -9 --prefix "normal-bananashed"
    python generation_main.py --model $model --datasets all --cal_zeroshot 0 --swipe  --states_index -3 -5 -7 -9 --prefix "normal"
done
