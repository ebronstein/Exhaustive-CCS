from typing import Literal

import numpy as np

from utils.types import DataDictType, PermutationDictType, PromptIndicesDictType
from utils_extraction.projection import myReduction


def getConcat(data_list, axis=0):
    sub_list = [w for w in data_list if w is not None]
    if not sub_list:
        return None
    return np.concatenate(sub_list, axis=axis)


def getPair(
    target_dict: PromptIndicesDictType,
    data_dict: DataDictType,
    permutation_dict: PermutationDictType,
    projection_model: myReduction,
    split: Literal["train", "test"] = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate the data and labels for the desired split.

    `projection_model` is used to transform the hidden states. This may be a
    noop (e.g., if `projection_model.n_components` is -1).

    Args:
        target_dict (dict): Dictionary of prompt indices to use for each
            dataset. Key is dataset name, each value is a list of prompt
            indices for which the corresponding data and labels are returned.
        data_dict (dict): Dictionary of hidden states. Key is dataset name, each
            value is a list with one element per prompt. Each element is a tuple
            pair of (hidden_states, labels).
        permutation_dict (dict): Dictionary of permutations. Key is dataset
            name, value is a tuple pair containing the train split and test
            split indices.
        projection_model (myReduction): Projection model used to transform the
            hidden states.
        split (str, optional): The desired split. Defaults to "train".
    """
    if split == "train":
        split_idx = 0
    elif split == "test":
        split_idx = 1
    else:
        raise ValueError(
            f"split should be either 'train' or 'test', got '{split}'"
        )

    lis = []
    for key, prompt_lis in target_dict.items():
        # Indices of the desired split for the current dataset.
        split_indices = permutation_dict[key][split_idx]
        for idx in prompt_lis:
            hidden_states = data_dict[key][idx][0][split_indices]
            labels = data_dict[key][idx][1][split_indices]
            lis.append(
                [
                    projection_model.transform(hidden_states),
                    labels,
                ]
            )

    data, label = getConcat([w[0] for w in lis]), getConcat([w[1] for w in lis])

    return data, label
