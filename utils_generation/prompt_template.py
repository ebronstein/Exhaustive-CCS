from abc import ABC, abstractmethod
from typing import Union

import pandas as pd


class PromptTemplate(ABC):
    @abstractmethod
    def apply(
        self,
        example: Union[dict[str, Union[str, int]], pd.Series],
        choice_labels: list[str],
        partition_index: int,
        qaexamples: tuple[pd.DataFrame, list[int]],
    ) -> str:
        """

        Args:
            example: A dictionary or a pandas Series containing the example
                text.
            prompt_idx: Index of the prompt to use.
            choice_labels: Possible answer choices (i.e., labels). For example,
                ["positive", "negative"] or ["choice 1", "choice 2"].
            partition_index: Partition index.
            qaexamples: A tuple containing a DataFrame of question-answer
                examples. This could be used for in-context learning, for
                example.

        Returns:
            The formatted question.
        """
        pass
