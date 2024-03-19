#!/bin/bash

# ALL_DATASETS=(imdb amazon-polarity ag-news dbpedia-14 copa rte boolq qnli piqa)
MAIN_TRAIN_SETS=(imdb amazon-polarity ag-news dbpedia-14 copa rte boolq qnli piqa)
NUM_SEEDS=10
MODE="concat"
TRAIN_PREFIX="normal"
TEST_PREFIX="normal-bananashed"

# Logistic regression config
CS=(0.01)
MAX_ITERS=(10000)
PENALTIES=("l2")

BASE_NAME="Llama-2-7b-chat-hf_train_prefix_normal-test_prefix_normal-bananashed_LR-mode=concat"

for dataset in "${MAIN_TRAIN_SETS[@]}"; do
    for c in "${CS[@]}"; do
        for max_iter in "${MAX_ITERS[@]}"; do
            for penalty in "${PENALTIES[@]}"; do
                NAME=$BASE_NAME/C=$c-max_iter=$max_iter-penalty=$penalty DATASETS=$dataset EVAL_DATASETS=burns PREFIX=$TRAIN_PREFIX TEST_PREFIX=$TEST_PREFIX MODE=$MODE METHOD_LIST='["LR"]' C=$c MAX_ITER=$max_iter PENALTY=$penalty NUM_SEEDS=$NUM_SEEDS sbatch -p jsteinhardt run_single_extract.sh
            done
        done
    done
done
