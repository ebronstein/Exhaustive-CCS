from typing import Literal

import numpy as np

# prefix_data_dict[dataset][prefix] = (hidden states, labels)
PrefixDataDictType = dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]]
# data_dict[dataset] = (hidden states, labels)
DataDictType = dict[str, list[tuple[np.ndarray, np.ndarray]]]
PermutationDictType = dict[str, tuple[np.ndarray, np.ndarray]]
PromptIndicesDictType = dict[str, list[int]]
Mode = Literal["concat", "minus", "0", "1"]
