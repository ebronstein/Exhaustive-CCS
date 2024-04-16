import itertools
import os

import numpy as np

from extract import ex
from utils_extraction import load_utils

if __name__ == "__main__":
    base_config = {
        "model": "meta-llama/Llama-2-7b-chat-hf",
        # Data
        # "datasets": "dbpedia-14",
        # "labeled_datasets": "imdb",
        # "eval_datasets": ["imdb", "amazon-polarity", "ag-news", "dbpedia-14"],
        # Prefix
        "prefix": "normal-bananashed",
        "test_prefix": "normal-bananashed",
        # Method (general)
        "method_list": ["CCS+LR-in-span"],
        "mode": "concat",
        "seed": 0,
        # LR
        # "log_reg": {"C": 0.01, "max_iter": 10000, "penalty": "l2"},
        # CCS-related
        "n_epochs": 1000,
        "sup_weight": 0.0,
        "unsup_weight": 1.0,
        "lr": 1e-2,
        # "num_orthogonal_directions": 100,
        "projected_sgd": True,
        "span_dirs_combination": "convex",
        "n_tries": 1,
        # Saving
        "save_states": False,
        "save_params": False,
        "save_fit_result": True,
        "save_fit_plots": False,
        "save_train_test_split": False,
        "save_orthogonal_directions": True,
    }

    all_datasets = [
        "imdb",
        "amazon-polarity",
        "ag-news",
        "dbpedia-14",
        "copa",
        "rte",
        "boolq",
        "qnli",
        "piqa",
    ]

    # labeled_datasets = ["imdb", "amazon-polarity", "ag-news", "dbpedia-14"]
    # unlabeled_datasets = ["imdb", "amazon-polarity", "ag-news", "dbpedia-14"]
    train_test_datasets = list(itertools.product(all_datasets, all_datasets))

    # Load orthogonal directions or train them from scratch.
    load_orthogonal_directions_base_dir = None
    # load_orthogonal_directions_base_dir = "/nas/ucb/ebronstein/Exhaustive-CCS/extraction_results/Llama-2-7b-chat-hf_normal-bananashed_CCS-in-LR-span-convex_20_orth_dirs/meta-llama-Llama-2-7b-chat-hf"

    # Iterate over num_orth_dirs and datasets.
    num_orth_dirs_list = [100]
    for num_orth_dirs in num_orth_dirs_list:
        name = f"Llama-2-7b-chat-hf_normal-bananashed_CCS-in-CCS-span-convex_{num_orth_dirs}_orth_dirs"

        for train_ds in all_datasets:
        # for train_ds, test_ds in train_test_datasets:
            # Load LR orthogonal directions from train set.
            # Arbitrarily use imdb as the unlabeled dataset since it wasn't used
            # to get the LR directions.
            # datasets_str = load_utils.get_combined_datasets_str(
            #     ["imdb"], labeled_datasets=[train_ds]
            # )
            # load_orthogonal_directions_dir = os.path.join(
            #     load_orthogonal_directions_base_dir, datasets_str
            # )
            # if not os.path.exists(load_orthogonal_directions_dir):
            #     raise ValueError(
            #         f"Could not find orthogonal directions for {train_ds} in {load_orthogonal_directions_dir}"
            #     )

            new_config = dict(
                name=name,
                datasets=train_ds,
                labeled_datasets=train_ds,  # Unused since sup_weight=0
                eval_datasets=[train_ds],
                # load_orthogonal_directions_dir=load_orthogonal_directions_dir,
                num_orthogonal_directions=num_orth_dirs,
            )
            if set(new_config).intersection(set(base_config)):
                raise ValueError(
                    f"Overlapping keys between base_config and new_config: {set(new_config).intersection(set(base_config))}"
                )

            config_updates = dict(base_config, **new_config)
            try:
                ex.run(config_updates=config_updates)
            except Exception as e:
                print(f"Error: {e}")
                raise e

    # LR: iterate over datasets.
    # for ds in all_datasets:
    #     new_config = dict(
    #         datasets=ds,
    #         eval_datasets=all_datasets,
    #     )
    #     if set(new_config).intersection(set(base_config)):
    #         raise ValueError(
    #             f"Overlapping keys between base_config and new_config: {set(new_config).intersection(set(base_config))}"
    #         )

    #     config_updates = dict(base_config, **new_config)
    #     ex.run(config_updates=config_updates)
