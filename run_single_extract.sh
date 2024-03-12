#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=ebronstein@berkeley.edu
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1

set -ex

source /accounts/projects/jsteinhardt/$(whoami)/.bashrc
cd /scratch/users/$(whoami)/Exhaustive-CCS
conda activate exhaustive-ccs

DATASETS=${DATASETS:-"imdb"}
LABELED_DATASETS=${LABELED_DATASETS:-"imdb"}
EVAL_DATASETS=${EVAL_DATASETS:-"burns"}
LR=${LR:-1e-2}
SUP_WEIGHT=${SUP_WEIGHT:-3}
UNSUP_WEIGHT=${UNSUP_WEIGHT:-1}
METHOD_LIST=${METHOD_LIST:-'["CCS+LR"]'}
MODE=${MODE:-"concat"}
MODEL=${MODEL:-"/scratch/data/meta-llama/Llama-2-7b-chat-hf"}
PREFIX=${PREFIX:-"normal-bananashed"}
NUM_SEEDS=${NUM_SEEDS:-10}

if [[ -z "$NAME" ]]; then
    echo "Error: NAME is not set"
    exit 1
fi

args=(
    name=$NAME \
    model=$MODEL \
    datasets=$DATASETS \
    labeled_datasets=$LABELED_DATASETS \
    eval_datasets=$EVAL_DATASETS \
    prefix=$PREFIX \
    method_list=$METHOD_LIST \
    sup_weight=$SUP_WEIGHT \
    unsup_weight=$UNSUP_WEIGHT \
    n_tries=10 \
    n_epochs=1000 \
    lr=$LR \
    save_states=False \
    save_params=False
)

for (( seed=0; seed<NUM_SEEDS; seed++ )); do
    args_copy=("${args[@]}")
    args_copy+=(seed=$seed)
    # echo "Args: ${args_copy[@]}"
    srun python extract.py with "${args_copy[@]}" "$@"
done
