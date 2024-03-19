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

MODEL=${MODEL:-"/scratch/data/meta-llama/Llama-2-7b-chat-hf"}
DATASETS=${DATASETS:-"imdb"}
LABELED_DATASETS=${LABELED_DATASETS:-'[]'}
EVAL_DATASETS=${EVAL_DATASETS:-"burns"}
PREFIX=${PREFIX:-"normal-bananashed"}
TEST_PREFIX=${TEST_PREFIX:-"normal-bananashed"}
MODE=${MODE:-"concat"}
METHOD_LIST=${METHOD_LIST:-'["CCS+LR"]'}
SUP_WEIGHT=${SUP_WEIGHT:-3}
UNSUP_WEIGHT=${UNSUP_WEIGHT:-1}
LR=${LR:-1e-2}
N_EPOCHS=${N_EPOCHS:-1000}
NUM_ORTHOGONAL_DIRS=${NUM_ORTHOGONAL_DIRS:-4}
OPT=${OPT:-"sgd"}
NUM_SEEDS=${NUM_SEEDS:-10}
N_TRIES=${N_TRIES:-10}
C=${C:-0.1}
MAX_ITER=${MAX_ITER:-10000}
PENALTY=${PENALTY:-"l2"}

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
    test_prefix=$TEST_PREFIX \
    mode=$MODE \
    method_list=$METHOD_LIST \
    sup_weight=$SUP_WEIGHT \
    unsup_weight=$UNSUP_WEIGHT \
    n_tries=$N_TRIES \
    n_epochs=$N_EPOCHS \
    lr=$LR \
    num_orthogonal_dirs=$NUM_ORTHOGONAL_DIRS \
    opt=$OPT \
    log_reg.C=$C \
    log_reg.max_iter=$MAX_ITER \
    log_reg.penalty=$PENALTY \
    save_states=False \
    save_params=False
)

for (( seed=0; seed<NUM_SEEDS; seed++ )); do
    args_copy=("${args[@]}")
    args_copy+=(seed=$seed)
    # echo "Args: ${args_copy[@]}"
    srun python extract.py with "${args_copy[@]}" "$@"
done
