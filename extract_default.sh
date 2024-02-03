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

source /accounts/projects/jsteinhardt/$(whoami)/.bashrc
cd /scratch/users/$(whoami)/Exhaustive-CCS
conda activate exhaustive_ccs

methods="CCS LR Random CCS-md LR-md Random-md"

# Datasets for which logistic regression gets >=90% accuracy for each model.
uqa_good_ds="imdb amazon-polarity ag-news dbpedia-14 copa boolq story-cloze"
deberta_good_ds="imdb amazon-polarity ag-news dbpedia-14 copa rte boolq qnli piqa"
gptj_good_ds="imdb amazon-polarity ag-news dbpedia-14"

# Models for extraction.
model_names=(deberta-v2-xxlarge-mnli)
# Datasets to use for each model. Must be in the same order as model_names.
# Each element is the name of the variable containing the datasets for that model.
model_to_ds=(deberta_good_ds)
# Short name for each model. Must be in the same order as model_names.
model_to_short=(deberta)

test_on_trains=("" "--test_on_train")
test_on_train_extensions=("" "/test_on_train")

NUM_SEEDS=10

for i_test_on_train in 0 1; do
    for ((i_model=0; i_model<${#model_names[@]}; i_model++)); do
        model=${model_names[$i_model]}
        ds=${!model_to_ds[$i_model]}
        short=${model_to_short[$i_model]}
        test_on_train=${test_on_trains[$i_test_on_train]}
        test_on_train_extension=${test_on_train_extensions[$i_test_on_train]}
        for (( seed=0; seed<NUM_SEEDS; seed++ )); do
            # if seed == 0, save states
            save_states=""
            if [ $seed -eq 0 ]; then
                save_states="--save_states"
            fi

            python extraction_main.py --model $model --datasets $ds --method_list $methods --prefix normal-dot --seed $seed --save_dir extraction_results$test_on_train_extension $save_states $test_on_train
        done
    done
done
