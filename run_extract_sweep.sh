#!/bin/bash

UNLABELED_TRAIN_SETS=(imdb amazon-polarity ag-news dbpedia-14)
LABELED_TRAIN_SETS=(imdb amazon-polarity ag-news dbpedia-14)
LRS=(1e-2)
SUP_WEIGHTS=(3)
NUM_SEEDS=10

for lr in "${LRS[@]}"; do
    for sup_weight in "${SUP_WEIGHTS[@]}"; do
        for unlabeled_ds in "${UNLABELED_TRAIN_SETS[@]}"; do
            for labeled_ds in "${LABELED_TRAIN_SETS[@]}"; do
                NAME=Llama-2-7b-chat-hf_normal-bananashed_CCS+LR_transfer_sup_weight=$sup_weight-unsup_weight=1-lr=$lr-n_epochs=1000 DATASETS=$unlabeled_ds LABELED_DATASETS=$labeled_ds EVAL_DATASETS=burns PREFIX=normal-bananashed METHOD_LIST='["CCS+LR"]' SUP_WEIGHT=$sup_weight UNSUP_WEIGHT=1 LR=$lr SEED=$seed NUM_SEEDS=$NUM_SEEDS sbatch -p jsteinhardt run_single_extract.sh
            done
        done
    done
done
