"""Metrics for evaluating model performance.

CalibrationError implementation is from EleutherAI/ccs/ccs/metrics/calibration.py.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class CalibrationEstimate:
    ece: float
    num_bins: int
    prob_means: Tensor
    label_means: Tensor


@dataclass
class CalibrationError:
    """Monotonic Sweep Calibration Error for binary problems.

    This method estimates the True Calibration Error (TCE) by searching for the largest
    number of bins into which the data can be split that preserves the monotonicity
    of the predicted confidence -> empirical accuracy mapping. We use equal mass bins
    (quantiles) instead of equal width bins. Roelofs et al. (2020) show that this
    estimator has especially low bias in simulations where the TCE is analytically
    computable, and is hyperparameter-free (except for the type of norm used).

    Paper: "Mitigating Bias in Calibration Error Estimation" by Roelofs et al. (2020)
    Link: https://arxiv.org/abs/2012.08668
    """

    labels: list[Tensor] = field(default_factory=list)
    pred_probs: list[Tensor] = field(default_factory=list)

    def update(self, labels: Tensor, probs: Tensor) -> "CalibrationError":
        labels, probs = labels.detach().flatten(), probs.detach().flatten()
        assert labels.shape == probs.shape
        assert torch.is_floating_point(probs)

        self.labels.append(labels)
        self.pred_probs.append(probs)
        return self

    def compute(self, p: int = 2, num_bins: Optional[int] = None) -> CalibrationEstimate:
        """Compute the expected calibration error.

        Args:
            p: The norm to use for the calibration error. Defaults to 2 (Euclidean).
        """
        labels = torch.cat(self.labels)
        pred_probs = torch.cat(self.pred_probs)

        n = len(pred_probs)
        if n < 2:
            raise ValueError("Not enough data to compute calibration error.")

        # Sort the predictions and labels
        pred_probs, indices = pred_probs.sort()
        labels = labels[indices].float()

        # Search for the largest number of bins which preserves monotonicity.
        # Based on Algorithm 1 in Roelofs et al. (2020).
        if num_bins == None:
            # Using a single bin is guaranteed to be monotonic, so we start there.
            b_star = 1
            b_min = 2
            b_max = n + 1
        else:
            b_star = num_bins
            b_min = num_bins
            b_max = num_bins + 1
        accs_star = labels.mean().unsqueeze(0)
        for b in range(b_min, b_max):
            # Split into (nearly) equal mass bins
            freqs = torch.stack([h.mean() for h in labels.tensor_split(b)])

            # This binning is not strictly monotonic, let's break
            if not torch.all(freqs[1:] > freqs[:-1]):
                break

            elif not torch.all(freqs * (1 - freqs)):
                break

            # Save the current binning, it's monotonic and may be the best one
            else:
                accs_star = freqs
                b_star = b

        # Split into (nearly) equal mass bins. They won't be exactly equal, so we
        # still weight the bins by their size.
        conf_bins = pred_probs.tensor_split(b_star)
        w = pred_probs.new_tensor([len(c) / n for c in conf_bins])

        # See the definition of ECE_sweep in Equation 8 of Roelofs et al. (2020)
        mean_confs = torch.stack([c.mean() for c in conf_bins])
        ece = torch.sum(w * torch.abs(accs_star - mean_confs) ** p) ** (1 / p)

        return CalibrationEstimate(float(ece), b_star, mean_confs, accs_star)


def expected_calibration_error(probs, labels, num_bins=10):
    """
    Calculate the Expected Calibration Error (ECE) of a classification model.

    Args:
    - probs (array-like): Predicted probabilities for each sample, in the range [0, 1].
    - labels (array-like): True labels for each sample, in {0, 1}.
    - num_bins (int): Number of bins to divide the interval [0, 1].

    Returns:
    - ece (float): Expected Calibration Error.
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    # Ensure the inputs have the same length
    assert probs.shape[0] == labels.shape[0], "Number of probabilities and labels must be the same."

    # Calculate the bin boundaries
    percentiles = np.linspace(0, 100, 11)
    bin_boundaries = np.percentile(probs, percentiles)

    # Initialize variables to store total confidence and accuracy in each bin
    bin_mean_probs = np.zeros(num_bins)
    bin_mean_labels = np.zeros(num_bins)
    bin_samples = np.zeros(num_bins)

    # Assign each prediction to its corresponding bin
    bin_indices = np.digitize(probs, bin_boundaries) - 1

    # Compute the total confidence and accuracy in each bin
    for i in range(num_bins):
        bin_mask = bin_indices == i
        bin_samples[i] = np.sum(bin_mask)
        if bin_samples[i] > 0:
            bin_mean_probs[i] = np.mean(probs[bin_mask])
            bin_mean_labels[i] = np.mean(labels[bin_mask])

    # Remove empty bins
    non_empty_bins = bin_samples > 0
    bin_mean_probs = bin_mean_probs[non_empty_bins]
    bin_mean_labels = bin_mean_labels[non_empty_bins]
    bin_samples = bin_samples[non_empty_bins]

    ece = np.average(np.abs(bin_mean_labels - bin_mean_probs), weights=bin_samples / np.sum(bin_samples))

    return ece, bin_mean_probs, bin_mean_labels
