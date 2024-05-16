from typing import Literal, Union

import numpy as np

# data_dict[dataset] is a dictionary mapping a prompt index to a tuple
# (hidden states, labels) for that prompt index.
DataDictType = dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]
# prefix_data_dict[prefix] is a DataDictType for that prefix.
PrefixDataDictType = dict[str, DataDictType]

# permutation_dict[dataset] = (train split indices, test split indices)
PermutationDictType = dict[str, tuple[np.ndarray, np.ndarray]]
# permutation_dict[prefix] is a PermutationDictType for that prefix.
PrefixPermutationDictType = dict[str, PermutationDictType]

PromptIndicesDictType = dict[str, list[int]]
Mode = Literal["concat", "minus", "0", "1"]
# TODO: add "sparse_random", "PCA", "UMAP" when implemented.
ProjectionMethod = Literal["gaussian_random"]

# Used for piecewise linear learning rate or loss weight schedules.
Milestones = list[tuple[int, float]]
PiecewiseLinearSchedule = Union[float, Milestones]
