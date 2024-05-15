#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=ebronstein@berkeley.edu
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
##SBATCH --nodelist=airl.ist.berkeley.edu,sac.ist.berkeley.edu,cirl.ist.berkeley.edu
#SBATCH --exclude=rlhf.ist.berkeley.edu,ppo.ist.berkeley.edu,vae.ist.berkeley.edu
#SBATCH --nodes=1
#SBATCH --qos=scavenger

# NOTE: rlhf.ist.berkeley.edu is excluded from the node list because something
# is wrong with using torch on it. Checkpoing shard loading stalls, and even
# an operation like torch.cuda.is_available() hangs.

set -ex

cd /nas/ucb/ebronstein/Exhaustive-CCS
eval "$(/nas/ucb/ebronstein/anaconda3/bin/conda shell.bash hook)"
conda activate exhaustive-ccs

# models=(/nas/ucb/ebronstein/models/meta-llama/Meta-Llama-3-8B-Instruct /nas/ucb/ebronstein/models/mistralai/Mistral-7B-Instruct-v0.2 /nas/ucb/nlauffer/datasets/llama-2-13b-chat /nas/ucb/nlauffer/datasets/llama-2-7b-chat)
models=(/nas/ucb/ebronstein/models/mistralai/Mistral-7B-Instruct-v0.2 /nas/ucb/nlauffer/datasets/llama-2-13b-chat)

prefixes=("normal")
# DATASETS="all"
DATASETS=("imdb" "amazon-polarity" "ag-news" "dbpedia-14" "copa" "rte" "boolq" "qnli" "piqa")
ALICE_EXPLICIT_OPINION_DATASETS=("imdb" "amazon-polarity" "ag-news" "dbpedia-14" "rte" "boolq" "qnli")
REMAINING_ALICE_EXPLICIT_OPINION_DATASETS=("imdb" "amazon-polarity" "ag-news" "dbpedia-14")
save_base_dir="generation_results"

NUM_DATA=1000

# Sweep models and prefixes. Use --swipe to use all prompts.
# for model in "${models[@]}"; do
#     for prefix in "${prefixes[@]}"; do
#         python generation_main.py --model $model --datasets $DATASETS --num_data $NUM_DATA --cal_zeroshot 0 --swipe --states_index -1 -3 -5 -7 -9 --prefix $prefix --save_base_dir=$save_base_dir --print_more
#     done
# done

# Alice explicit opinion prompts
for model in "${models[@]}"; do
    for prefix in "${prefixes[@]}"; do
        for dataset in "${REMAINING_ALICE_EXPLICIT_OPINION_DATASETS[@]}"; do
            python generation_main.py --model $model --datasets $dataset --num_data $NUM_DATA --cal_zeroshot 0 --states_index -1 --prompt_name alice_explicit_opinion_1 alice_explicit_opinion_2 alice_explicit_opinion_3 alice_explicit_opinion_4 alice_explicit_opinion_5 alice_explicit_opinion_6 alice_explicit_opinion_7 alice_explicit_opinion_8 alice_explicit_opinion_after_reading alice_explicit_opinion_based_on_following_passage alice_explicit_opinion_based_on_previous_passage alice_explicit_opinion_based_only_on alice_explicit_opinion_can_we_infer alice_explicit_opinion_could_you_tell_me alice_explicit_opinion_does_it_follow_that alice_explicit_opinion_does_this_imply alice_explicit_opinion_exam alice_explicit_opinion_exercise alice_explicit_opinion_gpt3_style alice_explicit_opinion_guaranteed_true alice_explicit_opinion_have_all_you_need alice_explicit_opinion_i_wonder alice_explicit_opinion_imply alice_explicit_opinion_justified_in_saying alice_explicit_opinion_mnli_crowdsource alice_explicit_opinion_must_be_true alice_explicit_opinion_possible_to_answer alice_explicit_opinion_should_assume alice_explicit_opinion_valid_binary alice_explicit_opinion_want_to_know alice_explicit_opinion_yes_no_question --prefix $prefix --save_base_dir=$save_base_dir --print_more
        done
    done
done
