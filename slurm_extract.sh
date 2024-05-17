#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=ebronstein@berkeley.edu
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=16gb
#SBATCH --gres=shard:4
#SBATCH --nodes=1
#SBATCH --qos=scavenger

set -ex

cd /nas/ucb/ebronstein/Exhaustive-CCS
eval "$(/nas/ucb/ebronstein/anaconda3/bin/conda shell.bash hook)"
conda activate exhaustive-ccs

MODEL=${MODEL:-"/scratch/data/meta-llama/Llama-2-7b-chat-hf"}
DATASETS=${DATASETS:-"imdb"}
LABELED_DATASETS=${LABELED_DATASETS:-'[]'}
EVAL_DATASETS=${EVAL_DATASETS:-"burns"}
PREFIX=${PREFIX:-"normal"}
MODE=${MODE:-"concat"}
LAYER=${LAYER:--1}
METHOD_LIST=${METHOD_LIST:-'["CCS+LR"]'}
SUP_WEIGHT=${SUP_WEIGHT:-3}
UNSUP_WEIGHT=${UNSUP_WEIGHT:-1}
CONSISTENCY_WEIGHT=${CONSISTENCY_WEIGHT:-1}
CONFIDENCE_WEIGHT=${CONFIDENCE_WEIGHT:-1}
LR=${LR:-1e-2}
N_EPOCHS=${N_EPOCHS:-1000}
NUM_ORTHOGONAL_DIRECTIONS=${NUM_ORTHOGONAL_DIRECTIONS:-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0}
OPT=${OPT:-"sgd"}
NUM_SEEDS=${NUM_SEEDS:-10}
N_TRIES=${N_TRIES:-10}
C=${C:-0.1}
MAX_ITER=${MAX_ITER:-10000}
PENALTY=${PENALTY:-"l2"}
DEVICE=${DEVICE:-"cuda"}
# Pseudo-label
PSEUDOLABEL_N_ROUNDS=${PSEUDOLABEL_N_ROUNDS:-1}
PSEUDOLABEL_SELECT_FN=${PSEUDOLABEL_SELECT_FN:-"high_confidence_consistency"}
PSEUDOLABEL_PROB_THRESHOLD=${PSEUDOLABEL_PROB_THRESHOLD:-0.8}
PSEUDOLABEL_LABEL_FN=${PSEUDOLABEL_LABEL_FN:-"argmax"}
PSEUDOLABEL_CONSISTENCY_ERR_THRESHOLD=${PSEUDOLABEL_CONSISTENCY_ERR_THRESHOLD:-0.1}
PSEUDOLABEL_SOFTMAX_TEMP=${PSEUDOLABEL_SOFTMAX_TEMP:-0.3}
# Saving
SAVE_DIR=${SAVE_DIR:-"extraction_results"}
SAVE_FIT_RESULT=${SAVE_FIT_RESULT:-True}
SAVE_FIT_PLOTS=${SAVE_FIT_PLOTS:-False}
SAVE_STATES=${SAVE_STATES:-False}
SAVE_PARAMS=${SAVE_PARAMS:-False}
SPAN_DIRS_COMBINATION=${SPAN_DIRS_COMBINATION:-"convex"}
SAVE_ORTHOGONAL_DIRECTIONS=${SAVE_ORTHOGONAL_DIRECTIONS:-False}

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
    mode=$MODE
    layer=$LAYER
    method_list=$METHOD_LIST
    sup_weight=$SUP_WEIGHT
    unsup_weight=$UNSUP_WEIGHT
    consistency_weight=$CONSISTENCY_WEIGHT
    confidence_weight=$CONFIDENCE_WEIGHT
    n_tries=$N_TRIES
    n_epochs=$N_EPOCHS
    lr=$LR
    num_orthogonal_directions=$NUM_ORTHOGONAL_DIRECTIONS
    weight_decay=$WEIGHT_DECAY
    opt=$OPT
    log_reg.C=$C
    log_reg.max_iter=$MAX_ITER
    log_reg.penalty=$PENALTY
    save_orthogonal_directions=$SAVE_ORTHOGONAL_DIRECTIONS
    span_dirs_combination=$SPAN_DIRS_COMBINATION
    device=$DEVICE
    # Pseudo-label
    pseudolabel.n_rounds=$PSEUDOLABEL_N_ROUNDS
    pseudolabel.select_fn=$PSEUDOLABEL_SELECT_FN
    pseudolabel.prob_threshold=$PSEUDOLABEL_PROB_THRESHOLD
    pseudolabel.label_fn=$PSEUDOLABEL_LABEL_FN
    pseudolabel.consistency_err_threshold=$PSEUDOLABEL_CONSISTENCY_ERR_THRESHOLD
    pseudolabel.softmax_temp=$PSEUDOLABEL_SOFTMAX_TEMP
    # Save
    save_dir=$SAVE_DIR
    save_states=$SAVE_STATES
    save_params=$SAVE_PARAMS
    save_fit_result=$SAVE_FIT_RESULT
    save_fit_plots=$SAVE_FIT_PLOTS
)

# Append optional environment variables if they are set.
if [[ -n "$LABELED_PREFIX" ]]; then
    args+=(labeled_prefix=$LABELED_PREFIX)
fi

if [[ -n "$TEST_PREFIX" ]]; then
    args+=(test_prefix=$TEST_PREFIX)
fi

# Surround the prompt indices with quotes to prevent the shell from expanding
# since the prompt indices are dictionaries with lists as values.
if [[ -n "$PROMPT_IDX" ]]; then
    args+=("prompt_idx=$PROMPT_IDX")
fi
if [[ -n "$LABELED_PROMPT_IDX" ]]; then
    args+=("labeled_prompt_idx=$LABELED_PROMPT_IDX")
fi
if [[ -n "$TEST_PROMPT_IDX" ]]; then
    args+=("test_prompt_idx=$TEST_PROMPT_IDX")
fi
if [[ -n "$PROMPT_SUBSET" ]]; then
    args+=(prompt_subset=$PROMPT_SUBSET)
fi
if [[ -n "$LABELED_PROMPT_SUBSET" ]]; then
    args+=(labeled_prompt_subset=$LABELED_PROMPT_SUBSET)
fi
if [[ -n "$TEST_PROMPT_SUBSET" ]]; then
    args+=(test_prompt_subset=$TEST_PROMPT_SUBSET)
fi

if [[ -n "$LOAD_ORTHOGONAL_DIRECTIONS_DIR" ]]; then
    args+=(load_orthogonal_directions_dir=$LOAD_ORTHOGONAL_DIRECTIONS_DIR)
fi

# Projection
if [[ -n "$PROJECTION_METHOD" ]]; then
    args+=(projection_method=$PROJECTION_METHOD)
fi
if [[ -n "$PROJECTION_N_COMPONENTS" ]]; then
    args+=(projection_n_components=$PROJECTION_N_COMPONENTS)
fi

for ((seed = 0; seed < NUM_SEEDS; seed++)); do
    args_copy=("${args[@]}")
    args_copy+=(seed=$seed)
    # echo "Args: ${args_copy[@]}"
    python extract.py with "${args_copy[@]}" "$@"
done
