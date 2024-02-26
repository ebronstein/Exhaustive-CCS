import json
import os
from typing import Optional, Union

import numpy as np
import pandas as pd

from utils.file_utils import get_model_short_name
from utils.types import DataDictType, PermutationDictType

COEF_FILENAME = "coef.npy"
INTERCEPT_FILENAME = "intercept.npy"


def get_combined_datasets_str(datasets: Union[str, list[str]]) -> str:
    return datasets if isinstance(datasets, str) else "+".join(sorted(datasets))


def get_exp_dir(
    save_dir: str,
    name: str,
    model: str,
    train_sets: Union[str, list[str]],
    seed: int,
):
    model_short_name = get_model_short_name(model)
    train_sets_str = get_combined_datasets_str(train_sets)
    return os.path.join(
        save_dir, name, model_short_name, train_sets_str, f"seed_{seed}"
    )


def get_eval_dir(run_dir: str, dataset: str) -> str:
    return os.path.join(run_dir, "eval", dataset)


def get_params_dir(run_dir: str, method: str, prefix: str) -> str:
    return os.path.join(run_dir, "params", f"{method}_{prefix}")


def load_params(
    run_dir: str, method: str, prefix: str
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    path = get_params_dir(run_dir, method, prefix)
    coef_path = os.path.join(path, COEF_FILENAME)
    intercept_path = os.path.join(path, INTERCEPT_FILENAME)
    if not os.path.exists(coef_path):
        raise FileNotFoundError(
            f"No params found in {run_dir} for method={method}, prefix={prefix}"
        )
    coef = np.load(coef_path)
    intercept = (
        np.load(intercept_path) if os.path.exists(intercept_path) else None
    )
    return coef, intercept


def save_params(save_dir, coef: np.ndarray, intercept: Optional[np.ndarray]):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    coef_path = os.path.join(save_dir, COEF_FILENAME)
    np.save(coef_path, coef)
    if intercept is not None:
        intercept_path = os.path.join(save_dir, INTERCEPT_FILENAME)
        np.save(intercept_path, intercept)


def get_train_results_path(run_dir: str) -> str:
    return os.path.join(run_dir, "train.csv")


def get_eval_results_path(run_dir: str, dataset: str) -> str:
    eval_dir = get_eval_dir(run_dir, dataset)
    return os.path.join(eval_dir, "eval.csv")


def get_permutation_dict_path(run_dir: str, dataset: str) -> str:
    eval_dir = get_eval_dir(run_dir, dataset)
    return os.path.join(eval_dir, "train_test_split.npy")


def save_permutation_dict(permutation_dict: PermutationDictType, run_dir: str):
    for ds, ds_permutation_dict in permutation_dict.items():
        permutation_dict_path = get_permutation_dict_path(run_dir, ds)
        np.save(permutation_dict_path, ds_permutation_dict)


def maybe_append_project_suffix(method, project_along_mean_diff):
    if project_along_mean_diff:
        return method + "-md"
    return method


def get_probs_save_path(
    eval_dir: str, method: str, project_along_mean_diff: bool, prompt_idx: int
):
    """Returns the path to the directory where the classifier probabilities are saved."""
    method_str = maybe_append_project_suffix(method, project_along_mean_diff)
    filename = f"probs_{method_str}_{prompt_idx}.csv"
    return os.path.join(eval_dir, filename)


def parse_generation_dir(
    generation_dir: str,
) -> Optional[tuple[str, str, str, str, str, str]]:
    """Parse a generation directory to extract its parameters."""
    parts = generation_dir.split("_")
    if len(parts) != 6:
        return None

    short_model_name, dataset, num_examples, prompt_idx, confusion, location = (
        parts
    )
    prompt_idx = int(prompt_idx.replace("prompt", ""))
    num_examples = int(num_examples)
    return (
        short_model_name,
        dataset,
        num_examples,
        prompt_idx,
        confusion,
        location,
    )


def get_hidden_states_dirs(
    mdl,
    set_name,
    load_dir,
    data_num,
    confusion,
    place,
    prompt_idxs: Optional[list[int]] = None,
) -> list[str]:
    target_short_model_name = get_model_short_name(mdl)
    gen_dirs = []
    subdirs = [
        d
        for d in os.listdir(load_dir)
        if os.path.isdir(os.path.join(load_dir, d))
    ]
    for gen_dir in subdirs:
        parts = parse_generation_dir(gen_dir)
        if parts is None:
            continue
        (
            short_model_name,
            dataset,
            num_examples,
            prompt_idx,
            confusion_,
            location,
        ) = parts
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


def organize_hidden_states(
    hidden_states: tuple[np.ndarray], mode: str
) -> np.ndarray:
    """Organize the hidden states according to `mode`.

    Args:
        hidden_states (tuple): A tuple of hidden states corresponding to class
            "0" and class "1". Note that the classes may be arbitrary when using
            an unsupervised method.
        mode (str): The mode to organize the hidden states.
            Valid values are "0", "1", "minus", and "concat".

    Returns:
        numpy.ndarray: The organized hidden states.

    Raises:
        NotImplementedError: If the mode is not supported.
    """
    if mode in ["0", "1"]:
        return hidden_states[int(mode)]
    elif mode == "minus":
        return hidden_states[0] - hidden_states[1]
    elif mode == "concat":
        return np.concatenate(hidden_states, axis=-1)
    else:
        raise NotImplementedError("This mode is not supported.")


def normalize(data, scale=True, demean=True):
    # demean the array and rescale each data point
    data = data - np.mean(data, axis=0) if demean else data
    if not scale:
        return data
    norm = np.linalg.norm(data, axis=1)
    avgnorm = np.mean(norm)
    return data / avgnorm * np.sqrt(data.shape[1])


def load_hidden_states(
    mdl,
    set_name,
    load_dir,
    prompt_idx,
    location="encoder",
    layer=-1,
    data_num=1000,
    confusion="normal",
    place="last",
    scale=True,
    demean=True,
    mode="minus",
    logger=None,
):
    """Load generated hidden states, return a dict where key is the dataset name and values is a list. Each tuple in the list is the (x,y) pair of one prompt.

    if mode == minus, then get h - h'
    if mode == concat, then get np.concatenate([h,h'])
    elif mode == 0 or 1, then get h or h'

    Raises:
        ValueError: If no hidden states are found.
    """
    dir_list = get_hidden_states_dirs(
        mdl, set_name, load_dir, data_num, confusion, place, prompt_idx
    )
    if not dir_list:
        raise ValueError(
            f"No hidden states found in directory {load_dir} for model={mdl} "
            f"dataset={set_name} data_num={data_num} prefix={confusion} "
            f"location={place} prompt_idx={prompt_idx}"
        )

    append_list = ["_" + location + str(layer) for _ in dir_list]

    hidden_states = [
        organize_hidden_states(
            (
                np.load(os.path.join(w, "0{}.npy".format(app))),
                np.load(os.path.join(w, "1{}.npy".format(app))),
            ),
            mode=mode,
        )
        for w, app in zip(dir_list, append_list)
    ]

    # normalize
    hidden_states = [normalize(w, scale, demean) for w in hidden_states]
    if logger is not None:
        hs_shape = hidden_states[0].shape if hidden_states else "None"
        logger.info(
            "%s prompts for %s, with shape %s"
            % (len(hidden_states), set_name, hs_shape)
        )
    labels = [
        np.array(pd.read_csv(os.path.join(w, "frame.csv"))["label"].to_list())
        for w in dir_list
    ]

    return [(u, v) for u, v in zip(hidden_states, labels)]


def make_permutation_dict(data_list, rate=0.6) -> tuple[np.ndarray, np.ndarray]:
    length = len(data_list[0][1])
    permutation = np.random.permutation(range(length)).reshape(-1)
    return (
        permutation[: int(length * rate)],
        permutation[int(length * rate) :],
    )


def load_hidden_states_for_datasets(
    load_dir,
    mdl_name,
    dataset_list,
    prefix="normal",
    location=None,
    layer=-1,
    prompt_dict=None,
    data_num=1000,
    scale=True,
    demean=True,
    mode="minus",
    logger=None,
) -> tuple[DataDictType, PermutationDictType]:
    """Load hidden states and labels for the given datasets.

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

    Returns: A dict with key equals to set name, and value is a list.
        Each element in the list is a tuple (state, label) corresponding to a
        prompt. State has shape [num_data, hidden_dim], and label has shape
        [num_data]. For example, `data_dict["imdb"][0][0]` and
        `data_dict["imdb"][0][1]` contain the hidden states and labels for the
        first prompt for the imdb dataset, respectively.
    """
    if location not in ["encoder", "decoder"]:
        raise ValueError(
            f"Location must be either 'encoder' or 'decoder', got {location}"
        )
    elif location == "decoder" and layer < 0:
        raise ValueError(f"Decoder layer must be non-negative, got {layer}.")

    if logger is not None:
        logger.info(
            "start loading {} hidden states {} for {} with {} prefix. Prompt_dict: {}, Scale: {}, Demean: {}, Mode: {}".format(
                location,
                layer,
                mdl_name,
                prefix,
                prompt_dict if prompt_dict is not None else "ALL",
                scale,
                demean,
                mode,
            )
        )
    prompt_dict = (
        prompt_dict
        if prompt_dict is not None
        else {key: None for key in dataset_list}
    )
    data_dict = {
        set_name: load_hidden_states(
            mdl_name,
            set_name,
            load_dir,
            prompt_dict[set_name],
            location,
            layer,
            data_num=data_num,
            confusion=prefix,
            scale=scale,
            demean=demean,
            mode=mode,
            logger=logger,
        )
        for set_name in dataset_list
    }

    return data_dict


def get_zeros_acc(
    load_dir,
    csv_name,
    mdl_name,
    dataset_list,
    prefix,
    prompt_dict=None,
    avg=False,
):
    """
    Get the zero-shot accuracies for a given model and prefix.

    Args:
        load_dir (str): The directory where the CSV file is located.
        csv_name (str): The name of the CSV file (without the extension).
        mdl_name (str): The name of the model.
        dataset_list (list): A list of dataset names.
        prefix (str): The prefix.
        prompt_dict (dict, optional): A dictionary mapping dataset names to prompt indices. Defaults to None, which uses all of the prompts.
        avg (bool, optional): Whether to return the average accuracy. Defaults to False.

    Returns:
        dict or float: If avg is False, returns a dictionary where each key is a dataset name and the value is a list of accuracies.
        If avg is True, returns the average accuracy across all datasets.
    """
    zeros = pd.read_csv(os.path.join(load_dir, csv_name + ".csv"))
    zeros.dropna(subset=["calibrated"], inplace=True)
    subzeros = zeros.loc[
        (zeros["model"] == mdl_name) & (zeros["prefix"] == prefix)
    ]

    # Extend prompt_dict to ALL dict if it is None
    if prompt_dict is None:
        prompt_dict = {key: range(1000) for key in dataset_list}

    # Extract accuracy, each key is a set name and value is a list of acc
    acc_dict = {}
    for dataset in dataset_list:
        filtered_csv = subzeros.loc[
            (subzeros["dataset"] == dataset)
            & (subzeros["prompt_idx"].isin(prompt_dict[dataset]))
        ]
        acc_dict[dataset] = filtered_csv["calibrated"].to_list()

    if not avg:
        return acc_dict
    else:
        # get the dataset avg, and finally the global level avg
        return np.mean([np.mean(values) for values in acc_dict.values()])


def run_dir_has_params(run_dir: str) -> bool:
    params_dir = os.path.join(run_dir, "params")
    return os.path.exists(params_dir) and os.listdir(params_dir)


def maximum_existing_run_id(basedir: str, with_params: bool = True) -> int:
    """Return the maximum existing run ID in the given directory."""
    dir_nrs = [
        int(d)
        for d in os.listdir(basedir)
        # Run IDs are directories with a number as the name.
        if os.path.isdir(os.path.join(basedir, d))
        and d.isdigit()
        and (
            # If `with_params` is True, only consider directories with params.
            run_dir_has_params(os.path.join(basedir, d))
            if with_params
            else True
        )
    ]
    if dir_nrs:
        return max(dir_nrs)
    else:
        return 0
