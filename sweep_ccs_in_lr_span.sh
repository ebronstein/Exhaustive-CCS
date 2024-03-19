#!/bin/bash

# ALL_DATASETS=(imdb amazon-polarity ag-news dbpedia-14 copa rte boolq qnli piqa)

# For CCS+LR, these are the unlabeled train datasets. For other methods that
# use labeled only datasets (e.g., LR) or unlabeled only datasets (e.g., CCS),
# these are the only train datasets.
MAIN_TRAIN_SETS=(amazon-polarity ag-news dbpedia-14)
LABELED_TRAIN_SETS=(imdb)
LRS=(1e-3)
SUP_WEIGHTS=(10)
EPOCHS=(1000)
N_TRIES=1
NUM_ORTHOGONAL_DIRS=4
NUM_SEEDS=1
MODE="concat"
OPT="sgd"
TRAIN_PREFIX="normal-bananashed"
TEST_PREFIX="normal-bananashed"

BASE_NAME="Llama-2-7b-chat-hf_normal-bananashed_CCS-in-LR-span_debug"

# CCS+LR
for lr in "${LRS[@]}"; do
    for sup_weight in "${SUP_WEIGHTS[@]}"; do
        for n_epochs in "${EPOCHS[@]}"; do
            for unlabeled_ds in "${MAIN_TRAIN_SETS[@]}"; do
                for labeled_ds in "${LABELED_TRAIN_SETS[@]}"; do
                    NAME=$BASE_NAME/orth_dirs=$NUM_ORTHOGONAL_DIRS-sup_weight=$sup_weight-lr=$lr-n_epochs=$n_epochs DATASETS=$unlabeled_ds LABELED_DATASETS=$labeled_ds EVAL_DATASETS=burns PREFIX=$TRAIN_PREFIX TEST_PREFIX=$TEST_PREFIX MODE=$MODE METHOD_LIST='["CCS-in-LR-span"]' NUM_ORTHOGONAL_DIRS=$NUM_ORTHOGONAL_DIRS SUP_WEIGHT=$sup_weight UNSUP_WEIGHT=1 LR=$lr N_EPOCHS=$n_epochs OPT=$OPT NUM_SEEDS=$NUM_SEEDS N_TRIES=$N_TRIES save_states=False save_params=False save_fit_plots=False sbatch -p jsteinhardt run_single_extract.sh
                done
            done
        done
    done
done
