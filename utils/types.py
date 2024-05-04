from typing import Literal

import numpy as np

# prefix_data_dict[prefix][dataset] = (hidden states, labels)
PrefixDataDictType = dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]]
# data_dict[dataset] = (hidden states, labels)
DataDictType = dict[str, list[tuple[np.ndarray, np.ndarray]]]

# permutation_dict[prefix][dataset] = (train split indices, test split indices)
PrefixPermutationDictType = dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]
# permutation_dict[dataset] = (train split indices, test split indices)
PermutationDictType = dict[str, tuple[np.ndarray, np.ndarray]]

PromptIndicesDictType = dict[str, list[int]]
Mode = Literal["concat", "minus", "0", "1"]
