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

cd /nas/ucb/ebronstein/Exhaustive-CCS
eval "$(/nas/ucb/ebronstein/anaconda3/bin/conda shell.bash hook)"
conda activate exhaustive-ccs

METHODS='["LR"]'

MODE="concat"

ALL_DATASETS=(imdb amazon-polarity ag-news dbpedia-14 copa rte boolq qnli piqa)
ALL_DATASETS_STR='["imdb", "amazon-polarity", "ag-news", "dbpedia-14", "copa", "rte", "boolq", "qnli", "piqa"]'

# Models for extraction.
MODEL_NAMES=(/scratch/data/meta-llama/Llama-2-7b-chat-hf)

# Prefixes
PREFIXES=("normal-bananashed")


# Experiment name. Results will be saved to extraction_results/{EXPERIMENT_NAME}
EXPERIMENT_NAME="Llama-2-7b-chat-hf_normal-bananashed_LR"

NUM_SEEDS=10

for ((i_model=0; i_model<${#MODEL_NAMES[@]}; i_model++)); do
    for prefix in "${PREFIXES[@]}"; do
        for ((i_dataset=0; i_dataset<${#ALL_DATASETS[@]}; i_dataset++)); do
            model=${MODEL_NAMES[$i_model]}
            ds=${ALL_DATASETS[$i_dataset]}
            for (( seed=0; seed<NUM_SEEDS; seed++ )); do
                srun python extract.py with model=$model datasets=$ds eval_datasets=burns method_list="$METHODS" prefix=$prefix mode=$MODE seed=$seed save_states=True name=$EXPERIMENT_NAME
            done
        done
    done
done
