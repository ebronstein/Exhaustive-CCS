import numpy as np

from extract import ex

if __name__ == "__main__":
    config_updates = {
        "name": "Llama-2-7b-chat-hf_normal-bananashed_CCS-select-LR-sweep_num_dirs",
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "datasets": "dbpedia-14",
        "labeled_datasets": "imdb",
        "eval_datasets": ["imdb", "dbpedia-14"],
        "prefix": "normal-bananashed",
        "method_list": ["CCS-select-LR"],
        "n_tries": 1,
        "seed": 0,
        "save_states": False,
        "save_params": False,
        "load_orthogonal_directions_run_dir": "/nas/ucb/ebronstein/Exhaustive-CCS/extraction_results/Llama-2-7b-chat-hf_normal-bananashed_CCS-in-LR-span-sweep_num_dirs/meta-llama-Llama-2-7b-chat-hf/nolabel_dbpedia-14-label_imdb/seed_0/2",
    }

    # num_orth_dirs_list = [2**i for i in range(np.log2(1250).astype(int) + 1)]
    num_orth_dirs_list = range(5, 8)

    for num_orthogonal_directions in num_orth_dirs_list:
        ex.run(
            config_updates=dict(
                config_updates, num_orthogonal_directions=num_orthogonal_directions
            )
        )
