#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=ebronstein@berkeley.edu
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --mem=64gb
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --qos=scavenger
#SBATCH --nodes=1

set -ex

cd /nas/ucb/ebronstein/Exhaustive-CCS
eval "$(/nas/ucb/ebronstein/anaconda3/bin/conda shell.bash hook)"
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
NUM_ORTHOGONAL_DIRECTIONS=${NUM_ORTHOGONAL_DIRECTIONS:-4}
OPT=${OPT:-"sgd"}
NUM_SEEDS=${NUM_SEEDS:-10}
N_TRIES=${N_TRIES:-10}
C=${C:-0.1}
MAX_ITER=${MAX_ITER:-10000}
PENALTY=${PENALTY:-"l2"}
SAVE_ORTHOGONAL_DIRECTIONS=${SAVE_ORTHOGONAL_DIRECTIONS:-False}
SAVE_PARAMS=${SAVE_PARAMS:-False}
SAVE_FIT_RESULT=${SAVE_FIT_RESULT:-True}
SAVE_FIT_PLOTS=${SAVE_FIT_PLOTS:-False}
SPAN_DIRS_COMBINATION=${SPAN_DIRS_COMBINATION:-"convex"}

if [[ -z "$NAME" ]]; then
    echo "Error: NAME is not set"
    exit 1
fi

args=(
    name=$NAME
    model=$MODEL
    datasets=$DATASETS
    labeled_datasets=$LABELED_DATASETS
    eval_datasets=$EVAL_DATASETS
    prefix=$PREFIX
    test_prefix=$TEST_PREFIX
    mode=$MODE
    method_list=$METHOD_LIST
    sup_weight=$SUP_WEIGHT
    unsup_weight=$UNSUP_WEIGHT
    n_tries=$N_TRIES
    n_epochs=$N_EPOCHS
    lr=$LR
    num_orthogonal_directions=$NUM_ORTHOGONAL_DIRECTIONS
    opt=$OPT
    log_reg.C=$C
    log_reg.max_iter=$MAX_ITER
    log_reg.penalty=$PENALTY
    save_orthogonal_directions=$SAVE_ORTHOGONAL_DIRECTIONS
    span_dirs_combination=$SPAN_DIRS_COMBINATION
    save_states=False
    save_params=$SAVE_PARAMS
    save_fit_result=$SAVE_FIT_RESULT
    save_fit_plots=$SAVE_FIT_PLOTS
)

# Append LOAD_ORTHOGONAL_DIRECTIONS_DIR if it is set.
if [[ -n "$LOAD_ORTHOGONAL_DIRECTIONS_DIR" ]]; then
    args+=(load_orthogonal_directions_dir=$LOAD_ORTHOGONAL_DIRECTIONS_DIR)
fi

for ((seed = 0; seed < NUM_SEEDS; seed++)); do
    args_copy=("${args[@]}")
    args_copy+=(seed=$seed)
    # echo "Args: ${args_copy[@]}"
    python extract.py with "${args_copy[@]}" "$@"
done
