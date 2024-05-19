import collections
from abc import ABC, abstractmethod
from copy import copy
from typing import Optional

import numpy as np
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
        self.init_weights = collections.OrderedDict(loss_weights)
        self.scale_init_weights = scale_init_weights

    @abstractmethod
    def update(self, loss_dict: dict[str, torch.tensor]):
        """Update the loss weights based on the loss values.

        Args:
            loss_dict (dict[str, torch.tensor]): Dictionary mapping loss names
                to loss values. The losses have not been detached from the
                computational graph yet and use the same device as the model.
        """
        pass

    @abstractmethod
    def get_loss_weights(self) -> dict[str, float]:
        """Compute the loss weights based on the loss values."""
        pass


class ConstantLossWeights(LossWeights):
    def update(self, loss_dict: dict[str, torch.tensor]):
        pass

    def get_loss_weights(self) -> dict[str, float]:
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

    def get_loss_weights(self) -> dict[str, float]:
        if self.scale_init_weights:
            weights = self.init_weights.copy()
        else:
            weights = {loss: 1.0 for loss in self.init_weights.keys()}

        for loss, history in self.loss_history.items():
            if not history:
                continue
            weights[loss] /= max(history)

        return weights


class SoftAdaptLossWeights(LossWeights):
    def __init__(
        self,
        loss_weights: dict[str, float],
        scale_init_weights: bool = False,
        normalize_loss_diff: bool = True,
        loss_weighted: bool = True,
        beta: float = 0.1,
    ):
        """
        Initialize the SoftAdapt loss weights.

        Args:
            loss_weights (dict[str, float]): Dictionary mapping loss names to loss weights.
            scale_init_weights (bool, optional): Whether to scale the initial weights (True) or override them (False). Defaults to False.
            normalize_loss_diff (bool, optional): Whether to normalize the loss differences. Defaults to True.
            loss_weighted (bool, optional): Whether to weight the loss weights
                by the loss values. Defaults to True.
            beta: Inverse temperature parameter for the softmax.
        """
        super().__init__(loss_weights, scale_init_weights)
        self.normalize_loss_diff = normalize_loss_diff
        self.loss_weighted = loss_weighted
        self.beta = beta
        # Make loss_history an OrderedDict to use numpy arrays in the
        # computation and ensure consistent ordering.
        self.loss_history: collections.OrderedDict[str, collections.deque[float]] = (
            collections.OrderedDict(
                {loss: collections.deque([], maxlen=2) for loss in loss_weights.keys()}
            )
        )

    def update(self, loss_dict: dict[str, torch.tensor]):
        """Update the loss weights based on the loss values."""
        for loss_name, loss_value in loss_dict.items():
            self.loss_history[loss_name].append(loss_value.item())

    def get_loss_weights(self) -> dict[str, float]:
        """Compute the loss weights based on the loss values."""
        # Compute the differences in the loss values.
        s = np.array(
            -np.diff(losses, prepend=0.0)[-1] if losses else 1.0
            for losses in self.loss_history.values()
        )
        eps = 1e-8
        if self.normalize_loss_diff:
            # Normalize the differences.
            s /= np.abs(s).sum() + eps

        # Apply softmax to get the loss weights.
        alpha = np.exp(self.beta * (s - s.max()))
        if self.loss_weighted:
            # Weight the loss weights by the loss values.
            loss_values = np.array(
                [losses[-1] if losses else 1.0 for losses in self.loss_history.values()]
            )
            alpha *= loss_values
            alpha /= alpha.sum() + eps

        loss_weights = {k: v for k, v in zip(self.loss_history.keys(), alpha)}

        if self.scale_init_weights:
            # Scale the initial weights.
            loss_weights = {
                k: w * self.init_weights[k] for k, w in loss_weights.items()
            }

        return loss_weights


def make_loss_weighting(
    name: str,
    loss_weights: dict[str, float],
    scale_init_weights: bool = False,
    **kwargs,
):
    if name == "constant":
        return ConstantLossWeights(loss_weights, scale_init_weights=scale_init_weights)
    elif name == "normalize":
        return NormalizeLossWeights(
            loss_weights, scale_init_weights=scale_init_weights, **kwargs
        )
    elif name == "softadapt":
        return SoftAdaptLossWeights(
            loss_weights, scale_init_weights=scale_init_weights, **kwargs
        )
    else:
        raise ValueError(f"Unknown loss weighting method: {name}")
