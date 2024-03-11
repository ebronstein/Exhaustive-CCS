from typing import Literal

import numpy as np

DataDictType = dict[str, list[tuple[np.ndarray, np.ndarray]]]
PermutationDictType = dict[str, tuple[np.ndarray, np.ndarray]]
PromptIndicesDictType = dict[str, list[int]]
Mode = Literal["concat", "minus", "0", "1"]
