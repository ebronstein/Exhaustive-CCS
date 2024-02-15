import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from utils_generation.save_utils import get_model_short_name

######## JSON Load ########
json_dir = "./registration"

with open("{}.json".format(json_dir), "r") as f:
    global_dict = json.load(f)
registered_dataset_list = global_dict["dataset_list"]
registered_prefix = global_dict["registered_prefix"]


def parse_generation_dir(generation_dir: str) -> Optional[tuple[str, str, str, str, str, str]]:
    """Parse a generation directory to extract its parameters."""
    parts = generation_dir.split("_")
    if len(parts) != 6:
        return None

    short_model_name, dataset, num_examples, prompt_idx, confusion, location = parts
    prompt_idx = int(prompt_idx.replace("prompt", ""))
    num_examples = int(num_examples)
    return short_model_name, dataset, num_examples, prompt_idx, confusion, location


def getDirList(mdl, set_name, load_dir, data_num, confusion, place, prompt_idxs: Optional[list[int]] = None):
    target_short_model_name = get_model_short_name(mdl)
    gen_dirs = []
    subdirs = [d for d in os.listdir(load_dir) if os.path.isdir(os.path.join(load_dir, d))]
    for gen_dir in subdirs:
        parts = parse_generation_dir(gen_dir)
        if parts is None:
            continue
        short_model_name, dataset, num_examples, prompt_idx, confusion_, location = parts
        if (
            short_model_name == target_short_model_name
            and dataset == set_name
            and num_examples == data_num
            and (prompt_idxs is None or prompt_idx in prompt_idxs)
            and confusion_ == confusion
            and location == place
        ):
            gen_dirs.append(gen_dir)

    return [os.path.join(load_dir, d) for d in gen_dirs]


def organizeStates(lis, mode):
    '''
        Whether to do minus, to concat or do nothing
    '''
    if mode in ["0", "1"]:
        return lis[int(mode)]
    elif mode == "minus":
        return lis[0] - lis[1]
    elif mode == "concat":
        return np.concatenate(lis, axis = -1)
    else:
        raise NotImplementedError("This mode is not supported.")

def normalize(data, scale =True, demean = True):
    # demean the array and rescale each data point
    data = data - np.mean(data, axis = 0) if demean else data
    if not scale:
        return data
    norm = np.linalg.norm(data, axis=1)
    avgnorm = np.mean(norm)
    return data / avgnorm * np.sqrt(data.shape[1])

def loadHiddenStates(mdl, set_name, load_dir, promtpt_idx, location = "encoder", layer = -1, data_num = 1000, confusion = "normal", place = "last", scale = True, demean = True, mode = "minus", verbose = True):
    '''Load generated hidden states, return a dict where key is the dataset name and values is a list. Each tuple in the list is the (x,y) pair of one prompt.

    if mode == minus, then get h - h'
    if mode == concat, then get np.concatenate([h,h'])
    elif mode == 0 or 1, then get h or h'

    Raises:
        ValueError: If no hidden states are found.
    '''
    dir_list = getDirList(mdl, set_name, load_dir, data_num, confusion, place, promtpt_idx)
    if not dir_list:
        raise ValueError(
            "No hidden states found for {} {} {} {} {} {} {}".format(
                mdl, set_name, load_dir, data_num, confusion, place, promtpt_idx))

    append_list = ["_" + location + str(layer) for _ in dir_list]

    hidden_states = [
                        organizeStates(
                            [np.load(os.path.join(w, "0{}.npy".format(app))),
                            np.load(os.path.join(w, "1{}.npy".format(app)))],
                            mode = mode)
                        for w, app in zip(dir_list, append_list)
                    ]

    # normalize
    hidden_states = [normalize(w, scale, demean) for w in hidden_states]
    if verbose:
        hs_shape = hidden_states[0].shape if hidden_states else "None"
        print("{} prompts for {}, with shape {}".format(len(hidden_states), set_name, hs_shape))
    labels = [np.array(pd.read_csv(os.path.join(w, "frame.csv"))["label"].to_list()) for w in dir_list]

    return [(u,v) for u,v in zip(hidden_states, labels)]

def getPermutation(data_list, rate = 0.6):
    length = len(data_list[0][1])
    permutation = np.random.permutation(range(length)).reshape(-1)
    return [permutation[: int(length * rate)], permutation[int(length * rate):]]


def getDic(load_dir, mdl_name, dataset_list, prefix = "normal", location=None, layer=-1, prompt_dict = None, data_num = 1000, scale = True, demean = True, mode = "minus", verbose = True):
    """Loads hidden states and labels.

    Args:
        mdl_name: name of the model
        dataset_list: list of all datasets
        prefix: the prefix used for the hidden states
        location: Either 'encoder' or 'decoder'. Determine which hidden states to load.
        layer: An index representing which layer in `location` should we load the hidden state from.
        prompt_dict: dict of prompts to consider. Default is taking all prompts (empty dict). Key is the set name and value is an index list. Only return hiiden states from corresponding prompts.
        data_num: population of the dataset. Default is 1000, and it depends on generation process.
        scale: whether to rescale the whole dataset
        demean: whether to subtract the mean
        mode: how to generate hidden states from h and h'
        verbose: Whether to print more

    Returns: Tuple (data_dict, permutation_dict):
        data_dict: a dict with key equals to set name, and value is a list. Each element in the list is a tuple (state, label) corresponding to a prompt. State has shape (#data * #dim), and label has shape (#data). For example, data_dict["imdb"][0][0] contains the hidden states for the first prompt for the imdb dataset.
        permutation_dict: [train_idx, test_idx], where train_idx is the subset of [#data] that corresponds to the training set, and test_idx is the subset that corresponds to the test set.
    """
    if location not in ["encoder", "decoder"]:
        raise ValueError(f"Location must be either 'encoder' or 'decoder', got {location}")
    elif location == "decoder" and layer < 0:
        raise ValueError(f"Decoder layer must be non-negative, got {layer}.")

    print("start loading {} hidden states {} for {} with {} prefix. Prompt_dict: {}, Scale: {}, Demean: {}, Mode: {}".format(location, layer, mdl_name, prefix, prompt_dict if prompt_dict is not None else "ALL", scale, demean, mode))
    prompt_dict = prompt_dict if prompt_dict is not None else {key: None for key in dataset_list}
    data_dict = {set_name: loadHiddenStates(mdl_name, set_name, load_dir, prompt_dict[set_name], location, layer, data_num = data_num, confusion = prefix, scale = scale, demean = demean, mode = mode, verbose = verbose) for set_name in dataset_list}
    permutation_dict = {set_name: getPermutation(data_dict[set_name]) for set_name in dataset_list}
    return data_dict, permutation_dict

# print("------ Func: get_zeros_acc ------\n\
# ## Input = csv_name, mdl_name, dataset_list, prefix, prompt_dict = None, avg = False\n\
#     csv_name: The name of csv we get accuracy from.\n\
#     mdl_name: The name of the model.\n\
#     dataset_list: List of dataset you want the accuracy from.\n\
#     prefix: The name of prefix.\n\
#     prompt_dict: Same as in getDir(). You can specify which prompt to get using this variable. Default is None, i.e. get all prompts.\n\
#     avg: Whether to average upon return. If True, will return a numbers, otherwise a dict with key from dataset_list and values being a list of accuracy.\n\
# ## Output = number / dict, depending on `avg`\n\
# "
# )
def get_zeros_acc(load_dir, csv_name, mdl_name, dataset_list, prefix, prompt_dict = None, avg = False):
    zeros = pd.read_csv(os.path.join(load_dir, csv_name + ".csv"))
    zeros.dropna(subset=["calibrated"], inplace=True)
    subzeros = zeros.loc[(zeros["model"] == mdl_name) & (zeros["prefix"] == prefix)]

    # Extend prompt_dict to ALL dict if it is None
    if prompt_dict is None:
        prompt_dict = {key: range(1000) for key in dataset_list}

    # Extract accuracy, each key is a set name and value is a list of acc
    acc_dict = {}
    for dataset in dataset_list:
        filtered_csv = subzeros.loc[(subzeros["dataset"] == dataset) & (
            subzeros["prompt_idx"].isin(prompt_dict[dataset])
        )]
        acc_dict[dataset] = filtered_csv["calibrated"].to_list()

    if not avg:
        return acc_dict
    else:
        # get the dataset avg, and finally the global level avg
        return np.mean([np.mean(values) for values in acc_dict.values()])


def load_params(save_dir, name) -> tuple[np.ndarray, Optional[np.ndarray]]:
    path = os.path.join(save_dir, "params")
    coef_path = os.path.join(path, "coef_{}.npy".format(name))
    intercept_path = os.path.join(path, "intercept_{}.npy".format(name))
    if not os.path.exists(coef_path):
        raise FileNotFoundError("No params found for {}".format(name))
    coef = np.load(coef_path)
    intercept = np.load(intercept_path) if os.path.exists(intercept_path) else None
    return coef, intercept
