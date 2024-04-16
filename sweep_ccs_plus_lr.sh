#!/bin/bash

# ALL_DATASETS=(imdb amazon-polarity ag-news dbpedia-14 copa rte boolq qnli piqa)

# For CCS+LR, these are the unlabeled train datasets. For other methods that
# use labeled only datasets (e.g., LR) or unlabeled only datasets (e.g., CCS),
# these are the only train datasets.
MAIN_TRAIN_SETS=(imdb dbpedia-14)
LABELED_TRAIN_SETS=(imdb dbpedia-14)
EVAL_DATASETS='["imdb", "dbpedia-14"]'
MODEL="meta-llama/Llama-2-7b-chat-hf"
LRS=(1e-3)
SUP_WEIGHTS=(10)
EPOCHS=(1000)
N_TRIES=1
NUM_SEEDS=1
MODE="concat"
OPT="sgd"
TRAIN_PREFIX="normal-bananashed"
TEST_PREFIX="normal-bananashed"

BASE_NAME="Llama-2-7b-chat-hf_normal-bananashed_CCS+LR_debug"

# CCS+LR
for lr in "${LRS[@]}"; do
    for sup_weight in "${SUP_WEIGHTS[@]}"; do
        for n_epochs in "${EPOCHS[@]}"; do
            for unlabeled_ds in "${MAIN_TRAIN_SETS[@]}"; do
                for labeled_ds in "${LABELED_TRAIN_SETS[@]}"; do
                    NAME=$BASE_NAME/sup_weight=$sup_weight-unsup_weight=1-lr=$lr-n_epochs=$n_epochs DATASETS=$unlabeled_ds LABELED_DATASETS=$labeled_ds MODEL=$MODEL EVAL_DATASETS=$EVAL_DATASETS PREFIX=$TRAIN_PREFIX TEST_PREFIX=$TEST_PREFIX MODE=$MODE METHOD_LIST='["CCS+LR"]' SUP_WEIGHT=$sup_weight UNSUP_WEIGHT=1 LR=$lr N_EPOCHS=$n_epochs N_TRIES=$N_TRIES OPT=$OPT NUM_SEEDS=$NUM_SEEDS sbatch run_single_extract.sh
                done
            done
        done
    done
done
