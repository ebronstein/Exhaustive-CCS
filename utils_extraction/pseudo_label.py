import typing
from typing import Literal, Optional

import numpy as np
import torch

from utils.types import PermutationDictType, PrefixDataDictType, PromptIndicesDictType
from utils_extraction import projection
from utils_extraction.classifier import (
    FIT_CCS_LR_KWARGS_NAMES,
    fit_ccs_lr,
    make_contrast_pair_data,
)

SelectFnType = Literal["all", "high_confidence_consistency"]
LabelFnType = Literal["argmax"]


def validate_pseudo_label_config(config: dict):
    select_fn = config["select_fn"]
    if select_fn not in typing.get_args(SelectFnType):
        raise ValueError(f"Invalid select_fn: {select_fn}")

    label_fn = config["label_fn"]
    if label_fn not in typing.get_args(LabelFnType):
        raise ValueError(f"Invalid label_fn: {label_fn}")

    n_rounds = config["n_rounds"]
    if not isinstance(n_rounds, int) or n_rounds < 1:
        raise ValueError(f"Invalid n_rounds: {n_rounds}")

    prob_threshold = config["prob_threshold"]
    if (
        not isinstance(prob_threshold, (float, int))
        or prob_threshold < 0
        or prob_threshold > 1
    ):
        raise ValueError(f"Invalid prob_threshold: {prob_threshold}")


def make_pseudolabel_mask(
    config, train_p0: np.ndarray, train_p1: np.ndarray
) -> np.ndarray:
    select_fn = config["select_fn"]
    if select_fn == "all":
        return make_all_mask(config, train_p0, train_p1)
    elif select_fn == "high_confidence_consistency":
        return make_high_confidence_consistency_mask(config, train_p0, train_p1)
    else:
        raise ValueError(f"Invalid select_fn: {select_fn}")


def make_all_mask(config, train_p0: np.ndarray, train_p1: np.ndarray) -> np.ndarray:
    return np.ones_like(train_p0, dtype=bool)


def make_high_confidence_consistency_mask(
    config, train_p0: np.ndarray, train_p1: np.ndarray
) -> np.ndarray:
    prob_threshold = config["prob_threshold"]
    x0_prob_high_mask = train_p0 >= prob_threshold
    x1_prob_high_mask = train_p1 >= prob_threshold
    return np.logical_xor(x0_prob_high_mask, x1_prob_high_mask)


def make_pseudolabels(config, train_p0: np.ndarray, train_p1: np.ndarray):
    label_fn = config["label_fn"]
    if label_fn == "argmax":
        return argmax_pseudolabels(config, train_p0, train_p1)
    else:
        raise ValueError(f"Invalid label_fn: {label_fn}")


def argmax_pseudolabels(config, train_p0: np.ndarray, train_p1: np.ndarray):
    return np.argmax(np.stack([train_p0, train_p1], axis=1), axis=1)


def train_pseudo_label(
    data_dict: PrefixDataDictType,
    labeled_train_data_dict: PromptIndicesDictType,
    unlabeled_train_data_dict: PromptIndicesDictType,
    permutation_dict: PermutationDictType,
    labeled_prefix: str,
    unlabeled_prefix: str,
    pseudo_label_config: dict,
    project_along_mean_diff: bool = False,
    projection_model=None,
    train_kwargs: Optional[dict] = None,
    device="cuda",
    logger=None,
):
    projection_model = projection_model or projection.IdentityReduction()

    # Labeled data.
    (train_sup_x0, train_sup_x1), train_sup_y = make_contrast_pair_data(
        target_dict=labeled_train_data_dict,
        data_dict=data_dict[labeled_prefix],
        permutation_dict=permutation_dict,
        projection_model=projection_model,
        split="train",
        project_along_mean_diff=project_along_mean_diff,
    )
    (test_sup_x0, test_sup_x1), test_sup_y = make_contrast_pair_data(
        target_dict=labeled_train_data_dict,
        data_dict=data_dict[labeled_prefix],
        permutation_dict=permutation_dict,
        projection_model=projection_model,
        split="test",
        project_along_mean_diff=project_along_mean_diff,
    )
    # Unlabeled data.
    (train_unsup_x0, train_unsup_x1), train_unsup_y = make_contrast_pair_data(
        target_dict=unlabeled_train_data_dict,
        data_dict=data_dict[unlabeled_prefix],
        permutation_dict=permutation_dict,
        projection_model=projection_model,
        split="train",
        project_along_mean_diff=project_along_mean_diff,
    )
    (test_unsup_x0, test_unsup_x1), test_unsup_y = make_contrast_pair_data(
        target_dict=unlabeled_train_data_dict,
        data_dict=data_dict[unlabeled_prefix],
        permutation_dict=permutation_dict,
        projection_model=projection_model,
        split="test",
        project_along_mean_diff=project_along_mean_diff,
    )

    train_kwargs = train_kwargs or {}
    train_kwargs = {
        key: train_kwargs[key] for key in FIT_CCS_LR_KWARGS_NAMES if key in train_kwargs
    }

    fit_result = fit_ccs_lr(
        train_sup_x0,
        train_sup_x1,
        train_sup_y,
        train_unsup_x0,
        train_unsup_x1,
        train_unsup_y,
        test_sup_x0,
        test_sup_x1,
        test_sup_y,
        test_unsup_x0,
        test_unsup_x1,
        test_unsup_y,
        verbose=True,
        device=device,
        logger=logger,
        **train_kwargs,
    )
    cur_probe = fit_result["best_probe"]

    # Make copies of the data that is modified in each round.
    cur_train_sup_x0 = train_sup_x0.copy()
    cur_train_sup_x1 = train_sup_x1.copy()
    cur_train_sup_y = train_sup_y.copy()
    cur_train_unsup_x0 = train_unsup_x0.copy()
    cur_train_unsup_x1 = train_unsup_x1.copy()
    cur_train_unsup_y = train_unsup_y.copy()

    # Make tensors with original data for evaluation.
    train_sup_x0 = torch.tensor(train_sup_x0, dtype=torch.float32, device=device)
    train_sup_x1 = torch.tensor(train_sup_x1, dtype=torch.float32, device=device)
    train_sup_y = torch.tensor(train_sup_y, dtype=torch.float32, device=device)
    test_sup_x0 = torch.tensor(test_sup_x0, dtype=torch.float32, device=device)
    test_sup_x1 = torch.tensor(test_sup_x1, dtype=torch.float32, device=device)
    test_sup_y = torch.tensor(test_sup_y, dtype=torch.float32, device=device)
    train_unsup_x0 = torch.tensor(train_unsup_x0, dtype=torch.float32, device=device)
    train_unsup_x1 = torch.tensor(train_unsup_x1, dtype=torch.float32, device=device)
    train_unsup_y = torch.tensor(train_unsup_y, dtype=torch.float32, device=device)
    test_unsup_x0 = torch.tensor(test_unsup_x0, dtype=torch.float32, device=device)
    test_unsup_x1 = torch.tensor(test_unsup_x1, dtype=torch.float32, device=device)
    test_unsup_y = torch.tensor(test_unsup_y, dtype=torch.float32, device=device)

    # Log accuracies of the initial model.
    train_sup_acc, train_sup_p0, train_sup_p1, train_sup_probs = (
        cur_probe.evaluate_accuracy(train_sup_x0, train_sup_x1, train_sup_y)
    )
    test_sup_acc, test_sup_p0, test_sup_p1, test_sup_probs = (
        cur_probe.evaluate_accuracy(test_sup_x0, test_sup_x1, test_sup_y)
    )
    train_unsup_acc, train_unsup_p0, train_unsup_p1, train_unsup_probs = (
        cur_probe.evaluate_accuracy(train_unsup_x0, train_unsup_x1, train_unsup_y)
    )
    test_unsup_acc, test_unsup_p0, test_unsup_p1, test_unsup_probs = (
        cur_probe.evaluate_accuracy(test_unsup_x0, test_unsup_x1, test_unsup_y)
    )
    fit_result["all_train_sup_acc"] = train_sup_acc
    fit_result["all_test_sup_acc"] = test_sup_acc
    fit_result["all_train_unsup_acc"] = train_unsup_acc
    fit_result["all_test_unsup_acc"] = test_unsup_acc
    if logger is not None:
        logger.info(f"Train supervised accuracy: {train_sup_acc}")
        logger.info(f"Test supervised accuracy: {test_sup_acc}")
        logger.info(f"Train unsupervised accuracy: {train_unsup_acc}")
        logger.info(f"Test unsupervised accuracy: {test_unsup_acc}")

    fit_results = [fit_result]
    probes = [cur_probe]
    n_rounds = pseudo_label_config["n_rounds"]

    for i in range(n_rounds):
        if logger is not None:
            logger.info(f"Round {i + 1}/{n_rounds}")
        else:
            print(f"Round {i + 1}/{n_rounds}")

        train_unsup_acc, train_unsup_p0, train_unsup_p1, train_unsup_probs = (
            cur_probe.evaluate_accuracy(
                torch.tensor(cur_train_unsup_x0, dtype=torch.float32, device=device),
                torch.tensor(cur_train_unsup_x1, dtype=torch.float32, device=device),
                torch.tensor(cur_train_unsup_y, dtype=torch.float32, device=device),
            )
        )
        train_unsup_p0 = train_unsup_p0.cpu().numpy()
        train_unsup_p1 = train_unsup_p1.cpu().numpy()

        # Select probabilities for pseudo-labeling.
        pseudolabel_mask = make_pseudolabel_mask(
            pseudo_label_config, train_unsup_p0, train_unsup_p1
        )
        assert (
            pseudolabel_mask.ndim == 2 and pseudolabel_mask.shape[1] == 1
        ), pseudolabel_mask.shape
        train_unsup_p0_with_pseudolabels = train_unsup_p0[pseudolabel_mask]
        train_unsup_p1_with_pseudolabels = train_unsup_p1[pseudolabel_mask]

        # Make pseudo-labels.
        train_unsup_pseudolabels = make_pseudolabels(
            pseudo_label_config,
            train_unsup_p0_with_pseudolabels,
            train_unsup_p1_with_pseudolabels,
        )

        # Get pseudo-labeled x data.
        n_pairs_with_pseudolabels = pseudolabel_mask.sum()
        broadcast_pseudolabel_mask = np.broadcast_to(
            pseudolabel_mask, cur_train_unsup_x0.shape
        )
        cur_train_unsup_x0_with_pseudolabels = cur_train_unsup_x0[
            broadcast_pseudolabel_mask
        ].reshape((n_pairs_with_pseudolabels, -1))
        cur_train_unsup_x1_with_pseudolabels = cur_train_unsup_x1[
            broadcast_pseudolabel_mask
        ].reshape((n_pairs_with_pseudolabels, -1))

        # Concatenate original labeled data with the pseudo-labeled data.
        cur_train_sup_x0 = np.concatenate(
            [cur_train_sup_x0, cur_train_unsup_x0_with_pseudolabels], axis=0
        )
        cur_train_sup_x1 = np.concatenate(
            [cur_train_sup_x1, cur_train_unsup_x1_with_pseudolabels], axis=0
        )
        cur_train_sup_y = np.concatenate(
            [cur_train_sup_y, train_unsup_pseudolabels], axis=0
        )

        # Update unlabeled data to be the remaining un-pseudo-labeled data.
        # This is the unlabeled data on which the accuracy is evaluated on this
        # round.
        n_pairs_without_pseudolabels = len(pseudolabel_mask) - n_pairs_with_pseudolabels
        # If all pairs are pseudo-labeled, keep all of the unsupervised data
        # for evaluation.
        if n_pairs_without_pseudolabels != 0:
            cur_train_unsup_x0 = cur_train_unsup_x0[
                ~broadcast_pseudolabel_mask
            ].reshape((n_pairs_without_pseudolabels, -1))
            cur_train_unsup_x1 = cur_train_unsup_x1[
                ~broadcast_pseudolabel_mask
            ].reshape((n_pairs_without_pseudolabels, -1))
            cur_train_unsup_y = cur_train_unsup_y[~pseudolabel_mask.squeeze(1)]

        # Fit the model to the pseudo-labeled data and original labeled data.
        fit_result = fit_ccs_lr(
            cur_train_sup_x0,
            cur_train_sup_x1,
            cur_train_sup_y,
            cur_train_unsup_x0,  # Only for eval, not for training.
            cur_train_unsup_x1,  # Only for eval, not for training.
            cur_train_unsup_y,  # Only for eval, not for training.
            test_sup_x0,
            test_sup_x1,
            test_sup_y,
            test_unsup_x0,
            test_unsup_x1,
            test_unsup_y,
            verbose=True,
            device=device,
            logger=logger,
            **train_kwargs,
        )
        cur_probe = fit_result["best_probe"]

        # Add number of pseudo-labeled pairs to the fit result.
        # Cast to int to make it JSON serializable.
        fit_result["n_pairs_with_pseudolabels"] = int(n_pairs_with_pseudolabels)
        fit_result["n_pairs_without_pseudolabels"] = int(n_pairs_without_pseudolabels)

        # Add accuracy on the original data to the fit result.
        train_sup_acc, train_sup_p0, train_sup_p1, train_sup_probs = (
            cur_probe.evaluate_accuracy(train_sup_x0, train_sup_x1, train_sup_y)
        )
        test_sup_acc, test_sup_p0, test_sup_p1, test_sup_probs = (
            cur_probe.evaluate_accuracy(test_sup_x0, test_sup_x1, test_sup_y)
        )
        train_unsup_acc, train_unsup_p0, train_unsup_p1, train_unsup_probs = (
            cur_probe.evaluate_accuracy(train_unsup_x0, train_unsup_x1, train_unsup_y)
        )
        test_unsup_acc, test_unsup_p0, test_unsup_p1, test_unsup_probs = (
            cur_probe.evaluate_accuracy(test_unsup_x0, test_unsup_x1, test_unsup_y)
        )
        fit_result["all_train_sup_acc"] = train_sup_acc
        fit_result["all_test_sup_acc"] = test_sup_acc
        fit_result["all_train_unsup_acc"] = train_unsup_acc
        fit_result["all_test_unsup_acc"] = test_unsup_acc

        # Log accuracies.
        if logger is not None:
            logger.info(f"Train supervised accuracy: {train_sup_acc}")
            logger.info(f"Test supervised accuracy: {test_sup_acc}")
            logger.info(f"Train unsupervised accuracy: {train_unsup_acc}")
            logger.info(f"Test unsupervised accuracy: {test_unsup_acc}")

        fit_results.append(fit_result)
        probes.append(cur_probe)

        # If all pairs are pseudo-labeled, stop training.
        if n_pairs_without_pseudolabels == 0:
            logger.info("All pairs are pseudo-labeled. Stopping training.")
            break

    return probes, fit_results