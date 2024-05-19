import collections
from abc import ABC, abstractmethod
from typing import Optional

import torch


class LossWeights(ABC):
    def __init__(
        self, loss_weights: dict[str, float], scale_init_weights: bool = False
    ):
        """Initialize the loss weights.

        NOTE: only the loss weights that are in the loss_weights dictionary
        will be updated.

        Args:
            loss_weights (dict[str, float]): Dictionary mapping loss names
                to loss weights.
            scale_init_weights (bool, optional): Whether to scale the
                initial weights (True) or override them (False). Defaults to
                False.
        """
        self.init_weights = loss_weights
        self.scale_init_weights = scale_init_weights

    @abstractmethod
    def update(self, loss_dict: dict[str, torch.tensor]):
        pass

    @abstractmethod
    def get_loss_weights(self):
        pass


class ConstantLossWeights(LossWeights):
    def update(self, loss_dict: dict[str, torch.tensor]):
        pass

    def get_loss_weights(self):
        return self.init_weights


class NormalizeLossWeights(LossWeights):
    """Normalize the loss weights based on their recent max values."""

    def __init__(
        self,
        loss_weights: dict[str, float],
        scale_init_weights: bool = False,
        window: Optional[int] = None,
    ):
        super().__init__(loss_weights, scale_init_weights)
        self.window = window
        maxlen = window if window is not None else 1
        self.loss_history: dict[str, collections.deque[float]] = {
            loss: collections.deque([], maxlen=maxlen) for loss in loss_weights.keys()
        }

    def update(self, loss_dict: dict[str, torch.tensor]):
        for loss, value in loss_dict.items():
            if loss in self.init_weights:
                loss_history = self.loss_history[loss]
                if self.window is None:
                    # Set the max value of the loss seen so far.
                    max_val = loss_history[0] if loss_history else float("-inf")
                    # Appending the max value replaces the previous value
                    # because the deque has a max length of 1 when window=None.
                    loss_history.append(max(max_val, value.item()))
                else:
                    # Append the loss value to the history.
                    loss_history.append(value.item())

    def get_loss_weights(self):
        if self.scale_init_weights:
            weights = self.init_weights.copy()
        else:
            weights = {loss: 1.0 for loss in self.init_weights.keys()}

        for loss, history in self.loss_history.items():
            if not history:
                continue
            weights[loss] /= max(history)

        return weights


def make_loss_weighting(
    name: str,
    loss_weights: dict[str, float],
    scale_init_weights: bool = False,
    **kwargs,
):
    if name == "constant":
        return ConstantLossWeights(loss_weights, scale_init_weights)
    elif name == "normalize":
        return NormalizeLossWeights(loss_weights, scale_init_weights, **kwargs)
    else:
        raise ValueError(f"Unknown loss weighting method: {name}")
