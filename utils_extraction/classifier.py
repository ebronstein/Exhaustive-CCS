import json
import os
import typing
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from copy import copy
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.types import (
    DataDictType,
    Mode,
    PermutationDictType,
    PrefixDataDictType,
    PromptIndicesDictType,
)
from utils_extraction import load_utils
from utils_extraction.data_utils import getPair
from utils_extraction.logistic_reg import LogisticRegressionClassifier
from utils_extraction.projection import myReduction

Tensor = Union[torch.Tensor, np.ndarray]
ContrastPairNp = tuple[np.ndarray, np.ndarray]
ContrastPair = tuple[torch.Tensor, torch.Tensor]
SpanDirsCombination = Literal["linear", "affine", "convex"]

# Keyword arguments for ContrastPairClassifier.train_model.
CCS_LR_KWARGS_NAMES = [
    "n_epochs",
    "lr",
    "unsup_weight",
    "sup_weight",
    "consistency_weight",
    "confidence_weight",
    "weight_decay",
    "opt",
]

# Keyword arguments for the fit_ccs_lr function.
FIT_CCS_LR_KWARGS_NAMES = CCS_LR_KWARGS_NAMES + [
    "n_tries",
    "include_bias",
    "span_dirs_combination",
]


def normalize(directions):
    """Normalize the directions.

    Args:
        directions: The directions to normalize. Shape: [n_directions, input_dim].

    Returns:
        The normalized directions.
    """
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def assert_close_to_orthonormal(directions, atol=1e-3):
    """Assert that the directions are close to orthonormal.

    Args:
        directions: The directions to check. Shape: [input_dim, n_directions].
        atol: The absolute tolerance for the check.

    Raises:
        AssertionError: If the directions are not close to orthonormal.
    """
    assert np.allclose(
        directions @ directions.T, np.eye(directions.shape[0]), atol=atol
    ), f"{directions} not orthonormal"


def project(x, along_directions):
    """Project x along the along_directions.

    x of shape (..., d) and along_directions of shape (n_directions, d)"""
    if isinstance(x, torch.Tensor) and isinstance(along_directions, torch.Tensor):
        inner_products = torch.einsum("...d,nd->...n", x, along_directions)
        return x - torch.einsum("...n,nd->...d", inner_products, along_directions)
    elif isinstance(x, np.ndarray) and isinstance(along_directions, np.ndarray):
        inner_products = np.einsum("...d,nd->...n", x, along_directions)
        return x - np.einsum("...n,nd->...d", inner_products, along_directions)
    else:
        raise ValueError(
            "x and along_directions should be both torch.Tensor or np.ndarray"
            f"Found {type(x)} and {type(along_directions)}"
        )


def project_coeff(coef_and_bias, along_directions):
    if along_directions is None:
        return coef_and_bias

    new_coef = project(coef_and_bias[:, :-1], along_directions)
    bias = coef_and_bias[:, -1]
    if isinstance(coef_and_bias, torch.Tensor):
        return torch.cat([new_coef, bias.unsqueeze(-1)], dim=-1)
    elif isinstance(coef_and_bias, np.ndarray):
        return np.concatenate([new_coef, bias[:, None]], axis=-1)
    else:
        raise ValueError("coef_and_bias should be either torch.Tensor or np.ndarray")


def project_data_along_axis(data, labels):
    # data: (n_samples, n_features)
    assert data.shape[0] == labels.shape[0]
    assert len(data.shape) == 2
    mean0 = np.mean(data[labels == 0], axis=0)
    mean1 = np.mean(data[labels == 1], axis=0)
    mean_diff = mean1 - mean0
    mean_diff /= np.linalg.norm(mean_diff)
    mean_diff = mean_diff.reshape(1, -1)
    return project(data, mean_diff)


def process_directions(directions: np.ndarray, input_dim: int) -> np.ndarray:
    if directions.ndim != 2:
        raise ValueError("Directions should have 2 dimensions.")

    if directions.shape[1] != input_dim:
        raise ValueError(
            "Directions should have the same number of columns as input_dim. "
            f"Directions shape: {directions.shape}, input_dim: {input_dim}"
        )

    directions = normalize(directions)
    assert_close_to_orthonormal(directions)
    return directions


def make_coef_bias(
    input_dim: int, include_bias: bool, device: str
) -> tuple[nn.Parameter, Optional[nn.Parameter]]:
    """Make the coefficients and bias for the classifier.

    Args:
        input_dim: The input dimension.
        include_bias: Whether to include a bias term.
        device: The device to use.

    Returns:
        coef: The coefficients with shape [1, input_dim].
        bias: The bias term with shape [1] if `include_bias`, otherwise None.
    """
    init_min = -1 / np.sqrt(input_dim)
    coef = torch.FloatTensor(1, input_dim).uniform_(init_min, -init_min)
    coef = nn.Parameter(coef.to(device))
    if include_bias:
        bias = torch.FloatTensor(1).uniform_(init_min, -init_min)
        bias = nn.Parameter(bias.to(device))
    else:
        bias = None
    return coef, bias


class ContrastPairClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        orthogonal_dirs: Optional[np.ndarray] = None,
        span_dirs: Optional[np.ndarray] = None,
        span_dirs_combination: SpanDirsCombination = "linear",
        include_bias=True,
        device="cuda",
        verbose=False,
    ):
        """CCS and LR classifier.

        TODO: complete docstring.

        Args:
            input_dim: Input dimension.
            orthogonal_dirs: [n_directions, input_dim]
            span_dirs: [n_directions, input_dim]

        Attributes:
            coef: [1, input_dim] if orthogonal_dirs and span_dirs is None,
                otherwise [1, n_directions].
            bias: [1] or None
        """
        super(ContrastPairClassifier, self).__init__()
        self.input_dim = input_dim
        self.span_dirs_combination = span_dirs_combination
        self.include_bias = include_bias
        self.device = device
        self.verbose = verbose

        if span_dirs_combination not in typing.get_args(SpanDirsCombination):
            raise ValueError(
                f"span_dirs_combination should be one of {SpanDirsCombination}"
            )

        if orthogonal_dirs is not None and span_dirs is not None:
            raise ValueError(
                "Only one of orthogonal_dirs and span_dirs should be provided."
            )

        if orthogonal_dirs is not None:
            orthogonal_dirs = process_directions(orthogonal_dirs, input_dim)
            self.register_buffer(
                "orthogonal_dirs",
                torch.tensor(orthogonal_dirs, dtype=torch.float32).to(device),
            )
            self.coef, self.bias = make_coef_bias(input_dim, include_bias, device)
            self.project_params(self.orthogonal_dirs)
            self.span_dirs = None
        elif span_dirs is not None:
            span_dirs = process_directions(span_dirs, input_dim)
            self.register_buffer(
                "span_dirs",
                torch.tensor(span_dirs, dtype=torch.float32).to(device),
            )
            self.coef, self.bias = make_coef_bias(
                span_dirs.shape[0], include_bias, device
            )
            self.orthogonal_dirs = None
        else:
            self.coef, self.bias = make_coef_bias(input_dim, include_bias, device)
            self.orthogonal_dirs = None
            self.span_dirs = None

    def set_params(self, coef: np.ndarray, bias: Optional[np.ndarray]):
        if bias is not None and self.bias is None:
            raise ValueError("Bias is not included in the model.")

        coef = torch.tensor(coef, dtype=torch.float32, device=self.device).reshape(
            1, -1
        )
        if coef.shape != self.coef.shape:
            raise ValueError(f"Expected coef shape {self.coef.shape}, got {coef.shape}")
        self.coef = nn.Parameter(coef)
        if bias is not None:
            bias = torch.tensor(bias, dtype=torch.float32, device=self.device).reshape(
                1
            )
            self.bias = nn.Parameter(bias)

    def project_params(self, directions: torch.Tensor):
        """Project `directions` out of the model parameters (coefficients).

        Args:
            directions: The directions to project out. Shape: [n_directions, input_dim].
        """
        with torch.no_grad():
            params = self.coef  # [1, input_dim]
            inner_products = torch.matmul(params, directions.t())  # [1, n_directions]
            # [1, input_dim]
            proj_params = params - torch.matmul(inner_products, directions)
            self.coef = nn.Parameter(proj_params)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)  # [N, input_dim]
        if self.span_dirs is not None:
            if self.span_dirs_combination == "affine":
                coef = self.coef / self.coef.sum(dim=1)
            elif self.span_dirs_combination == "convex":
                coef = F.softmax(self.coef, dim=1)
            else:
                coef = self.coef
            lin_combo = torch.matmul(coef, self.span_dirs)  # [1, input_dim]
            logits = torch.matmul(x, lin_combo.t())  # [N, 1]
        else:
            logits = torch.matmul(x, self.coef.t())  # [N, 1]
        if self.include_bias:
            logits += self.bias
        probs = torch.sigmoid(logits)
        return probs

    def compute_supervised_loss(
        self, sup_x0: torch.Tensor, sup_x1: torch.Tensor, sup_y: torch.Tensor
    ) -> OrderedDict:
        bce_loss_0 = F.binary_cross_entropy(self.predict(sup_x0), 1 - sup_y)
        bce_loss_1 = F.binary_cross_entropy(self.predict(sup_x1), sup_y)
        supervised_loss = 0.5 * (bce_loss_0 + bce_loss_1)
        return OrderedDict(
            [
                ("supervised_loss", supervised_loss),
                ("bce_loss_0", bce_loss_0),
                ("bce_loss_1", bce_loss_1),
            ]
        )

    def compute_unsupervised_loss(
        self,
        unsup_x0: torch.Tensor,
        unsup_x1: torch.Tensor,
        consistency_weight: float = 1.0,
        confidence_weight: float = 1.0,
    ) -> OrderedDict:
        unsup_prob_0 = self.predict(unsup_x0)
        unsup_prob_1 = self.predict(unsup_x1)
        consistency_loss = ((unsup_prob_0 - (1 - unsup_prob_1)) ** 2).mean()
        confidence_loss = torch.min(unsup_prob_0, unsup_prob_1).pow(2).mean()
        unsupervised_loss = (
            consistency_weight * consistency_loss + confidence_weight * confidence_loss
        )

        return OrderedDict(
            [
                ("unsupervised_loss", unsupervised_loss),
                ("consistency_loss", consistency_loss),
                ("confidence_loss", confidence_loss),
            ]
        )

    def compute_loss(
        self,
        sup_x0: torch.Tensor,
        sup_x1: torch.Tensor,
        sup_y: torch.Tensor,
        unsup_x0: torch.Tensor,
        unsup_x1: torch.Tensor,
        unsup_weight=1.0,
        sup_weight=1.0,
        consistency_weight=1.0,
        confidence_weight=1.0,
        # l2_weight=0.0,
    ) -> OrderedDict:
        supervised_loss_dict = self.compute_supervised_loss(sup_x0, sup_x1, sup_y)
        unsupervised_loss_dict = self.compute_unsupervised_loss(
            unsup_x0,
            unsup_x1,
            consistency_weight=consistency_weight,
            confidence_weight=confidence_weight,
        )

        # L2 regularization loss.
        # l2_reg_loss = torch.norm(self.coef) ** 2
        # if self.bias is not None:
        #     l2_reg_loss += torch.norm(self.bias) ** 2

        total_loss = (
            sup_weight * supervised_loss_dict["supervised_loss"]
            + unsup_weight * unsupervised_loss_dict["unsupervised_loss"]
            # + l2_weight * l2_reg_loss
        )

        loss_dict = OrderedDict([("total_loss", total_loss)])
        loss_dict.update(supervised_loss_dict)
        loss_dict.update(unsupervised_loss_dict)

        return loss_dict

    def train_model(
        self,
        train_sup_x0: torch.Tensor,
        train_sup_x1: torch.Tensor,
        train_sup_y: torch.Tensor,
        train_unsup_x0: torch.Tensor,
        train_unsup_x1: torch.Tensor,
        train_unsup_y: torch.Tensor,
        test_sup_x0: torch.Tensor,
        test_sup_x1: torch.Tensor,
        test_sup_y: torch.Tensor,
        test_unsup_x0: torch.Tensor,
        test_unsup_x1: torch.Tensor,
        test_unsup_y: torch.Tensor,
        n_epochs=1000,
        lr=1e-2,
        unsup_weight=1.0,
        sup_weight=1.0,
        consistency_weight=1.0,
        confidence_weight=1.0,
        weight_decay: float = 0.0,
        opt: str = "sgd",
        eval_freq: int = 20,
        logger=None,
    ) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        if opt == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt == "adam":
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt}")

        train_history = defaultdict(list)
        eval_history = defaultdict(list)
        self.train()

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            train_losses = self.compute_loss(
                train_sup_x0,
                train_sup_x1,
                train_sup_y,
                train_unsup_x0,
                train_unsup_x1,
                unsup_weight=unsup_weight,
                sup_weight=sup_weight,
                consistency_weight=consistency_weight,
                confidence_weight=confidence_weight,
            )
            train_losses["total_loss"].backward()
            optimizer.step()

            # Optionally project parameters.
            if self.orthogonal_dirs is not None:
                self.project_params(self.orthogonal_dirs)

            for key, loss in train_losses.items():
                train_history[key].append(loss.item())

            # Eval
            if epoch % eval_freq == 0 or epoch == n_epochs - 1:
                with torch.no_grad():
                    self.eval()
                    test_losses = self.compute_loss(
                        test_sup_x0,
                        test_sup_x1,
                        test_sup_y,
                        test_unsup_x0,
                        test_unsup_x1,
                        unsup_weight=unsup_weight,
                        sup_weight=sup_weight,
                    )
                train_sup_acc = self.evaluate_accuracy(
                    train_sup_x0, train_sup_x1, train_sup_y
                )[0]
                train_unsup_acc = self.evaluate_accuracy(
                    train_unsup_x0, train_unsup_x1, train_unsup_y
                )[0]
                test_sup_acc = self.evaluate_accuracy(
                    test_sup_x0, test_sup_x1, test_sup_y
                )[0]
                test_unsup_acc = self.evaluate_accuracy(
                    test_unsup_x0, test_unsup_x1, test_unsup_y
                )[0]

                eval_history["epoch"].append(epoch)
                eval_history["train_sup_acc"].append(train_sup_acc)
                eval_history["train_unsup_acc"].append(train_unsup_acc)
                eval_history["test_sup_acc"].append(test_sup_acc)
                eval_history["test_unsup_acc"].append(test_unsup_acc)
                for key, loss in test_losses.items():
                    eval_history[key].append(loss.item())

            if self.verbose and (epoch + 1) % 100 == 0:
                msg = f"Epoch {epoch+1}/{n_epochs}, Loss: {train_losses['total_loss'].item()}"
                if logger is not None:
                    logger.info(msg)
                else:
                    print(msg)

            train_history["coef_norm"].append(self.coef.norm().item())
            if self.include_bias:
                train_history["bias"].append(self.bias.item())

        return train_history, eval_history

    def evaluate_accuracy(
        self, x0: torch.Tensor, x1: torch.Tensor, y: torch.Tensor
    ) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the accuracy of the classifier on the given data.

        Args:
            x0: Class 0 contrast pair examples.
            x1: Class 1 contrast pair examples.
            y: Labels.

        Returns:
            accuracy: The accuracy of the classifier.
            p0: The predicted probabilities for the class 0 examples.
            p1: The predicted probabilities for the class 1 examples.
            probs: The predicted probabilities.
        """
        with torch.no_grad():
            self.eval()
            p0 = self.predict(x0)  # [N, 1]
            p1 = self.predict(x1)  # [N, 1]
            probs = ((p1 + 1 - p0) / 2).float()  # [N, 1]
            predictions = (probs >= 0.5).float().squeeze(-1)  # [N]
            y = y.to(self.device).float().view(-1)  # [N, 1]
            accuracy = (predictions == y).float().mean().item()

        return accuracy, p0, p1, probs


def fit_ccs_lr(
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
    orthogonal_dirs: Optional[np.ndarray] = None,
    span_dirs: Optional[np.ndarray] = None,
    span_dirs_combination: SpanDirsCombination = "linear",
    n_tries=10,
    n_epochs=1000,
    lr=1e-2,
    unsup_weight=1.0,
    sup_weight=1.0,
    consistency_weight=1.0,
    confidence_weight=1.0,
    weight_decay: float = 0.0,
    opt: str = "sgd",
    include_bias: bool = True,
    verbose=False,
    device="cuda",
    logger=None,
):
    best_loss = {"total_loss": float("inf")}
    best_probe = None
    all_probes = []
    final_losses = []
    train_histories = []
    eval_histories = []

    # Convert data to tensors.
    train_sup_x0 = torch.tensor(train_sup_x0, dtype=torch.float, device=device)
    train_sup_x1 = torch.tensor(train_sup_x1, dtype=torch.float, device=device)
    train_sup_y = torch.tensor(train_sup_y, dtype=torch.float, device=device).view(
        -1, 1
    )
    train_unsup_x0 = torch.tensor(train_unsup_x0, dtype=torch.float, device=device)
    train_unsup_x1 = torch.tensor(train_unsup_x1, dtype=torch.float, device=device)
    train_unsup_y = torch.tensor(train_unsup_y, dtype=torch.float, device=device).view(
        -1, 1
    )
    test_sup_x0 = torch.tensor(test_sup_x0, dtype=torch.float, device=device)
    test_sup_x1 = torch.tensor(test_sup_x1, dtype=torch.float, device=device)
    test_sup_y = torch.tensor(test_sup_y, dtype=torch.float, device=device).view(-1, 1)
    test_unsup_x0 = torch.tensor(test_unsup_x0, dtype=torch.float, device=device)
    test_unsup_x1 = torch.tensor(test_unsup_x1, dtype=torch.float, device=device)
    test_unsup_y = torch.tensor(test_unsup_y, dtype=torch.float, device=device).view(
        -1, 1
    )

    for _ in range(n_tries):
        classifier = ContrastPairClassifier(
            input_dim=train_sup_x1.shape[1],
            orthogonal_dirs=orthogonal_dirs,
            span_dirs=span_dirs,
            span_dirs_combination=span_dirs_combination,
            include_bias=include_bias,
            device=device,
            verbose=verbose,
        )
        train_history, eval_history = classifier.train_model(
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
            n_epochs,
            lr,
            unsup_weight=unsup_weight,
            sup_weight=sup_weight,
            consistency_weight=consistency_weight,
            confidence_weight=confidence_weight,
            weight_decay=weight_decay,
            opt=opt,
            logger=logger,
        )
        final_loss = {k: losses[-1] for k, losses in train_history.items()}

        all_probes.append(classifier)
        train_histories.append(train_history)
        eval_histories.append(eval_history)
        final_losses.append(final_loss)

        if final_loss["total_loss"] < best_loss["total_loss"]:
            best_loss = final_loss
            best_probe = classifier

    return {
        "best_probe": best_probe,
        "best_loss": best_loss,
        "all_probes": all_probes,
        "final_losses": final_losses,
        "train_histories": train_histories,
        "eval_histories": eval_histories,
    }


def make_contrast_pair_data(
    target_dict: PromptIndicesDictType,
    data_dict: DataDictType,
    permutation_dict: PermutationDictType,
    projection_model: myReduction,
    split: str,
    project_along_mean_diff=False,
    split_pair: bool = True,
) -> tuple[Union[tuple[np.ndarray, np.ndarray], np.ndarray], np.ndarray]:
    x_pair, y = getPair(
        target_dict=target_dict,
        data_dict=data_dict,
        permutation_dict=permutation_dict,
        projection_model=projection_model,
        split=split,
    )
    assert len(x_pair.shape) == 2
    if project_along_mean_diff:
        x_pair = project_data_along_axis(x_pair, y)

    # Convert to float32 since x_pair can be float16 when loaded.
    x_pair = x_pair.astype(np.float32)

    if split_pair:
        if x_pair.shape[1] % 2 != 0:
            raise ValueError(
                "Expected the same number of hidden states for "
                "class '0' and class '1' to be concatenated, got "
                f"{x_pair.shape[1]} total features, which is odd."
            )
        # Hidden states are concatenated, first class 0 then class 1.
        x0 = x_pair[:, : x_pair.shape[1] // 2]
        x1 = x_pair[:, x_pair.shape[1] // 2 :]
        x = (x0, x1)
    else:
        x = x_pair

    return x, y


def train_orthogonal_lr_probes(
    train_sup_x0: np.ndarray,
    train_sup_x1: np.ndarray,
    train_sup_y: np.ndarray,
    test_sup_x0: np.ndarray,
    test_sup_x1: np.ndarray,
    test_sup_y: np.ndarray,
    train_unsup_x0: np.ndarray,
    train_unsup_x1: np.ndarray,
    train_unsup_y: np.ndarray,
    test_unsup_x0: np.ndarray,
    test_unsup_x1: np.ndarray,
    test_unsup_y: np.ndarray,
    num_orthogonal_directions: int,
    mode: Mode,
    train_kwargs=None,
    logger=None,
):
    """Trains orthogonal directions using logistic regression.

    Args:
        TODO

    Returns:
        orthogonal_dirs: The orthogonal directions found by LR with shape
            [n_directions, hidden_dim].
        intercepts: The intercepts found by LR with shape [n_directions].
        lr_fit_results: The fit results of the LR probes.
    """
    train_kwargs = train_kwargs or {}
    cur_train_sup_x0 = train_sup_x0.copy()
    cur_train_sup_x1 = train_sup_x1.copy()
    orthogonal_dirs = []
    intercepts = []
    lr_fit_results = []

    for i in range(num_orthogonal_directions):
        if logger is not None:
            logger.info(f"Direction {i+1}/{num_orthogonal_directions}")

        lr_model = LogisticRegressionClassifier(n_jobs=1, solver="saga", **train_kwargs)
        lr_model.fit((cur_train_sup_x0, cur_train_sup_x1), train_sup_y, mode)
        orth_dir = lr_model.coef_ / np.linalg.norm(lr_model.coef_)
        orth_dir = orth_dir.squeeze(0)
        orthogonal_dirs.append(orth_dir)
        intercepts.append(lr_model.intercept_)

        # Eval
        fit_result = {}
        fit_result["sup_train_acc"] = lr_model.score(
            (train_sup_x0, train_sup_x1), train_sup_y, mode
        )[0]
        fit_result["sup_test_acc"] = lr_model.score(
            (test_sup_x0, test_sup_x1), test_sup_y, mode
        )[0]
        fit_result["unsup_train_acc"] = lr_model.score(
            (train_unsup_x0, train_unsup_x1), train_unsup_y, mode
        )[0]
        fit_result["unsup_test_acc"] = lr_model.score(
            (test_unsup_x0, test_unsup_x1), test_unsup_y, mode
        )[0]
        lr_fit_results.append(fit_result)

        if logger is not None:
            logger.info(f"Sup train acc: {fit_result['sup_train_acc']:.4f}")
            logger.info(f"Sup test acc: {fit_result['sup_test_acc']:.4f}")
            logger.info(f"Unsup train acc: {fit_result['unsup_train_acc']:.4f}")
            logger.info(f"Unsup test acc: {fit_result['unsup_test_acc']:.4f}")

        # Project away the direction.
        cur_train_sup_x0 -= (cur_train_sup_x0 @ orth_dir)[:, None] * orth_dir
        cur_train_sup_x1 -= (cur_train_sup_x1 @ orth_dir)[:, None] * orth_dir
        assert np.abs(cur_train_sup_x0 @ orth_dir).max() < 1e-4
        assert np.abs(cur_train_sup_x1 @ orth_dir).max() < 1e-4

    orthogonal_dirs = np.array(orthogonal_dirs)

    return orthogonal_dirs, intercepts, lr_fit_results


def train_orthogonal_probes(
    train_sup_x0: np.ndarray,
    train_sup_x1: np.ndarray,
    train_sup_y: np.ndarray,
    test_sup_x0: np.ndarray,
    test_sup_x1: np.ndarray,
    test_sup_y: np.ndarray,
    train_unsup_x0: np.ndarray,
    train_unsup_x1: np.ndarray,
    train_unsup_y: np.ndarray,
    test_unsup_x0: np.ndarray,
    test_unsup_x1: np.ndarray,
    test_unsup_y: np.ndarray,
    num_orthogonal_directions: int,
    projected_sgd: bool = True,
    include_bias: bool = True,
    train_kwargs=None,
    device="cuda",
    logger=None,
):
    """Trains orthogonal directions using CCS and logistic regression.

    Args:
        TODO
        projected_sgd: Whether to use projected SGD. If True, the orthogonal
            directions are projected out of the parameters after each update.
        train_kwargs: Keyword arguments passed to
            ContrastPairClassifier.train_model.

    Returns:
        orthogonal_dirs: The orthogonal directions found by LR with shape
            [n_directions, hidden_dim].
        intercepts: The intercepts found by LR with shape [n_directions].
        lr_fit_results: The fit results of the LR probes.
    """
    train_kwargs = train_kwargs or {}
    orthogonal_dirs = []
    intercepts = []
    fit_results = []

    # Convert data to tensors.
    train_sup_x0 = torch.tensor(train_sup_x0, dtype=torch.float, device=device)
    train_sup_x1 = torch.tensor(train_sup_x1, dtype=torch.float, device=device)
    train_sup_y = torch.tensor(train_sup_y, dtype=torch.float, device=device).view(
        -1, 1
    )
    train_unsup_x0 = torch.tensor(train_unsup_x0, dtype=torch.float, device=device)
    train_unsup_x1 = torch.tensor(train_unsup_x1, dtype=torch.float, device=device)
    train_unsup_y = torch.tensor(train_unsup_y, dtype=torch.float, device=device).view(
        -1, 1
    )
    test_sup_x0 = torch.tensor(test_sup_x0, dtype=torch.float, device=device)
    test_sup_x1 = torch.tensor(test_sup_x1, dtype=torch.float, device=device)
    test_sup_y = torch.tensor(test_sup_y, dtype=torch.float, device=device).view(-1, 1)
    test_unsup_x0 = torch.tensor(test_unsup_x0, dtype=torch.float, device=device)
    test_unsup_x1 = torch.tensor(test_unsup_x1, dtype=torch.float, device=device)
    test_unsup_y = torch.tensor(test_unsup_y, dtype=torch.float, device=device).view(
        -1, 1
    )

    for i in range(num_orthogonal_directions):
        if logger is not None:
            logger.info(f"Direction {i+1}/{num_orthogonal_directions}")

        # Project away the previous orthogonal directions if they exist and
        # projected_sgd.
        cur_orthogonal_dirs = (
            np.array(orthogonal_dirs) if projected_sgd and orthogonal_dirs else None
        )
        model = ContrastPairClassifier(
            train_sup_x1.shape[1],
            orthogonal_dirs=cur_orthogonal_dirs,
            include_bias=include_bias,
            device=device,
        )
        train_history, eval_history = model.train_model(
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
            logger=logger,
            **train_kwargs,
        )
        orth_dir = model.coef.detach().squeeze(0)
        # Normalize the orthogonal direction.
        orth_dir /= torch.norm(orth_dir)
        orthogonal_dirs.append(orth_dir.cpu().numpy())
        if model.bias is not None:
            intercepts.append(model.bias.detach().cpu().numpy())

        # Eval
        fit_result = {
            "sup_train_acc": eval_history["train_sup_acc"][-1],
            "sup_test_acc": eval_history["test_sup_acc"][-1],
            "unsup_train_acc": eval_history["train_unsup_acc"][-1],
            "unsup_test_acc": eval_history["test_unsup_acc"][-1],
        }
        fit_results.append(fit_result)

        if logger is not None:
            logger.info(f"Sup train acc: {fit_result['sup_train_acc']:.4f}")
            logger.info(f"Sup test acc: {fit_result['sup_test_acc']:.4f}")
            logger.info(f"Unsup train acc: {fit_result['unsup_train_acc']:.4f}")
            logger.info(f"Unsup test acc: {fit_result['unsup_test_acc']:.4f}")

        # Project away the direction.
        train_sup_x0 -= (train_sup_x0 @ orth_dir)[:, None] * orth_dir
        train_sup_x1 -= (train_sup_x1 @ orth_dir)[:, None] * orth_dir
        assert (train_sup_x0 @ orth_dir).abs().max() < 1e-4
        assert (train_sup_x1 @ orth_dir).abs().max() < 1e-4

    orthogonal_dirs = np.array(orthogonal_dirs)

    return orthogonal_dirs, intercepts, fit_results


def train_ccs_lr_in_span(
    data_dict: PrefixDataDictType,
    permutation_dict: PermutationDictType,
    unlabeled_train_data_dict: PromptIndicesDictType,
    labeled_train_data_dict: PromptIndicesDictType,
    projection_model: myReduction,
    labeled_prefix: str,
    unlabeled_prefix: str,
    num_orthogonal_directions: int,
    load_orthogonal_directions_run_dir: Optional[str] = None,
    projected_sgd: bool = True,
    train_kwargs={},
    project_along_mean_diff=False,
    device="cuda",
    logger=None,
) -> tuple[ContrastPairClassifier, dict, np.ndarray, np.ndarray]:
    """Train CCS+LR in the span of given orthogonal directions.

    Args:
        TODO
        load_orthogonal_directions_run_dir: Run directory from which to load the orthogonal directions. If provided, expects the file
        {load_orthogonal_directions_run_dir}/train/orthogonal_directions.npy
        to exist.
        projected_sgd: Whether to use projected SGD. If True, the orthogonal
            directions are projected out of the parameters after each update.

    Returns:
        best_probe: The best probe.
        final_fit_result: The fit result of the CCS and LR probes.
        orthogonal_dirs: The orthogonal directions found by LR with shape
            [n_directions, hidden_dim].
    """
    # Labeled data.
    sup_data_kwargs = dict(
        target_dict=labeled_train_data_dict,
        data_dict=data_dict[labeled_prefix],
        permutation_dict=permutation_dict,
        projection_model=projection_model,
        project_along_mean_diff=project_along_mean_diff,
    )
    (train_sup_x0, train_sup_x1), train_sup_y = make_contrast_pair_data(
        split="train",
        **sup_data_kwargs,
    )
    (test_sup_x0, test_sup_x1), test_sup_y = make_contrast_pair_data(
        split="test",
        **sup_data_kwargs,
    )
    # Unlabeled data.
    unsup_data_kwargs = dict(
        target_dict=unlabeled_train_data_dict,
        data_dict=data_dict[unlabeled_prefix],
        permutation_dict=permutation_dict,
        projection_model=projection_model,
        project_along_mean_diff=project_along_mean_diff,
    )
    (train_unsup_x0, train_unsup_x1), train_unsup_y = make_contrast_pair_data(
        split="train", **unsup_data_kwargs
    )
    (test_unsup_x0, test_unsup_x1), test_unsup_y = make_contrast_pair_data(
        split="test", **unsup_data_kwargs
    )

    if load_orthogonal_directions_run_dir is None:
        include_bias = train_kwargs["include_bias"]
        ccs_lr_train_kwargs = {
            k: v for k, v in train_kwargs.items() if k in CCS_LR_KWARGS_NAMES
        }
        orthogonal_dirs, intercepts, _ = train_orthogonal_probes(
            train_sup_x0,
            train_sup_x1,
            train_sup_y,
            test_sup_x0,
            test_sup_x1,
            test_sup_y,
            train_unsup_x0,
            train_unsup_x1,
            train_unsup_y,
            test_unsup_x0,
            test_unsup_x1,
            test_unsup_y,
            num_orthogonal_directions,
            projected_sgd=projected_sgd,
            include_bias=include_bias,
            train_kwargs=ccs_lr_train_kwargs,
            device=device,
            logger=logger,
        )
    else:
        # Load the orthogonal directions.
        orthogonal_dirs, intercepts = load_utils.load_orthogonal_directions(
            load_orthogonal_directions_run_dir, num_directions=num_orthogonal_directions
        )

    fit_train_kwargs = {
        k: v for k, v in train_kwargs.items() if k in FIT_CCS_LR_KWARGS_NAMES
    }
    final_fit_result = fit_ccs_lr(
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
        span_dirs=orthogonal_dirs,
        verbose=True,
        device=device,
        logger=logger,
        **fit_train_kwargs,
    )
    best_probe = final_fit_result["best_probe"]

    return best_probe, final_fit_result, orthogonal_dirs, intercepts


def train_ccs_in_lr_span(
    data_dict: PrefixDataDictType,
    permutation_dict: PermutationDictType,
    unlabeled_train_data_dict: PromptIndicesDictType,
    labeled_train_data_dict: PromptIndicesDictType,
    projection_model: myReduction,
    labeled_prefix: str,
    unlabeled_prefix: str,
    num_orthogonal_directions: int,
    mode: Mode,
    load_orthogonal_directions_run_dir: Optional[str] = None,
    train_kwargs={},
    project_along_mean_diff=False,
    device="cuda",
    logger=None,
) -> tuple[ContrastPairClassifier, dict, np.ndarray, np.ndarray]:
    """Train CCS in span of LR directions.

    Args:
        TODO
        load_orthogonal_directions_run_dir: Run directory from which to load the
            orthogonal directions. If provided, expects the file
            "{load_orthogonal_directions_run_dir}/train/orthogonal_directions.npy"
            to exist.

    Returns:
        best_probe: The best probe.
        final_fit_result: The fit result of the CCS and LR probes.
        orthogonal_dirs: The orthogonal directions found by LR with shape
            [n_directions, hidden_dim].
    """
    # Labeled data.
    sup_data_kwargs = dict(
        target_dict=labeled_train_data_dict,
        data_dict=data_dict[labeled_prefix],
        permutation_dict=permutation_dict,
        projection_model=projection_model,
        project_along_mean_diff=project_along_mean_diff,
    )
    (train_sup_x0, train_sup_x1), train_sup_y = make_contrast_pair_data(
        split="train",
        **sup_data_kwargs,
    )
    (test_sup_x0, test_sup_x1), test_sup_y = make_contrast_pair_data(
        split="test",
        **sup_data_kwargs,
    )
    # Unlabeled data.
    unsup_data_kwargs = dict(
        target_dict=unlabeled_train_data_dict,
        data_dict=data_dict[unlabeled_prefix],
        permutation_dict=permutation_dict,
        projection_model=projection_model,
        project_along_mean_diff=project_along_mean_diff,
    )
    (train_unsup_x0, train_unsup_x1), train_unsup_y = make_contrast_pair_data(
        split="train", **unsup_data_kwargs
    )
    (test_unsup_x0, test_unsup_x1), test_unsup_y = make_contrast_pair_data(
        split="test", **unsup_data_kwargs
    )

    lr_train_kwargs = train_kwargs.pop("log_reg", {})

    # Load the orthogonal directions if provided. Otherwise, train them.
    if load_orthogonal_directions_run_dir is not None:
        orthogonal_dirs, intercepts = load_utils.load_orthogonal_directions(
            load_orthogonal_directions_run_dir, num_directions=num_orthogonal_directions
        )
        lr_fit_results = []
    else:
        orthogonal_dirs, intercepts, lr_fit_results = train_orthogonal_lr_probes(
            train_sup_x0,
            train_sup_x1,
            train_sup_y,
            test_sup_x0,
            test_sup_x1,
            test_sup_y,
            train_unsup_x0,
            train_unsup_x1,
            train_unsup_y,
            test_unsup_x0,
            test_unsup_x1,
            test_unsup_y,
            num_orthogonal_directions,
            mode,
            train_kwargs=lr_train_kwargs,
            logger=logger,
        )

    ccs_train_kwargs = {
        k: v for k, v in train_kwargs.items() if k in FIT_CCS_LR_KWARGS_NAMES
    }
    ccs_train_kwargs.update({"sup_weight": 0.0, "unsup_weight": 1.0})
    final_fit_result = fit_ccs_lr(
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
        span_dirs=orthogonal_dirs,
        verbose=True,
        device=device,
        logger=logger,
        **ccs_train_kwargs,
    )
    best_probe = final_fit_result["best_probe"]

    # Add the orthogonal directions to the final fit result if they were trained
    # in this run.
    if load_orthogonal_directions_run_dir is None:
        final_fit_result["lr_fit_results"] = lr_fit_results

    return best_probe, final_fit_result, orthogonal_dirs, intercepts


def train_ccs_select_lr(
    data_dict: PrefixDataDictType,
    permutation_dict: PermutationDictType,
    unlabeled_train_data_dict: PromptIndicesDictType,
    labeled_train_data_dict: PromptIndicesDictType,
    projection_model: myReduction,
    labeled_prefix: str,
    unlabeled_prefix: str,
    num_orthogonal_directions: int,
    mode: Mode,
    load_orthogonal_directions_run_dir: Optional[str] = None,
    train_kwargs={},
    project_along_mean_diff=False,
    device="cuda",
    logger=None,
) -> tuple[ContrastPairClassifier, dict, np.ndarray, np.ndarray]:
    """Choose LR direction with the lowest CCS loss.

    Args:
        TODO
        num_orthogonal_directions: Number of orthogonal directions to use. If
            load_orthogonal_directions_run_dir is provided, the first
            num_orthogonal_directions of the loaded directions are used.
            Otherwise, exactly num_orthogonal_directions are generated.
        load_orthogonal_directions_run_dir: Run directory from which to load the
            orthogonal directions. If provided, expects the file
            "{load_orthogonal_directions_run_dir}/train/orthogonal_directions.npy"
            to exist.

    Returns:
        best_probe: The best probe.
        final_fit_result: The fit result of the CCS and LR probes.
        orthogonal_dirs: The orthogonal directions found by LR with shape
            [n_directions, hidden_dim].
    """
    # Labeled data.
    sup_data_kwargs = dict(
        target_dict=labeled_train_data_dict,
        data_dict=data_dict[labeled_prefix],
        permutation_dict=permutation_dict,
        projection_model=projection_model,
        project_along_mean_diff=project_along_mean_diff,
    )
    (train_sup_x0, train_sup_x1), train_sup_y = make_contrast_pair_data(
        split="train",
        **sup_data_kwargs,
    )
    (test_sup_x0, test_sup_x1), test_sup_y = make_contrast_pair_data(
        split="test",
        **sup_data_kwargs,
    )
    # Unlabeled data.
    unsup_data_kwargs = dict(
        target_dict=unlabeled_train_data_dict,
        data_dict=data_dict[unlabeled_prefix],
        permutation_dict=permutation_dict,
        projection_model=projection_model,
        project_along_mean_diff=project_along_mean_diff,
    )
    (train_unsup_x0, train_unsup_x1), train_unsup_y = make_contrast_pair_data(
        split="train", **unsup_data_kwargs
    )
    (test_unsup_x0, test_unsup_x1), test_unsup_y = make_contrast_pair_data(
        split="test", **unsup_data_kwargs
    )

    lr_train_kwargs = train_kwargs.pop("log_reg", {})

    # Load the orthogonal directions if provided. Otherwise, train them.
    if load_orthogonal_directions_run_dir is not None:
        # orthogonal_dirs shape: [hidden_dim, n_directions]
        orthogonal_dirs, intercepts = load_utils.load_orthogonal_directions(
            load_orthogonal_directions_run_dir, num_directions=num_orthogonal_directions
        )
        lr_fit_results = []
    else:
        orthogonal_dirs, intercepts, lr_fit_results = train_orthogonal_lr_probes(
            train_sup_x0,
            train_sup_x1,
            train_sup_y,
            test_sup_x0,
            test_sup_x1,
            test_sup_y,
            train_unsup_x0,
            train_unsup_x1,
            train_unsup_y,
            test_unsup_x0,
            test_unsup_x1,
            test_unsup_y,
            num_orthogonal_directions,
            mode,
            train_kwargs=lr_train_kwargs,
            logger=logger,
        )

    # Convert to tensors for CCS loss computation.
    train_sup_x0 = torch.tensor(train_sup_x0, dtype=torch.float, device=device)
    train_sup_x1 = torch.tensor(train_sup_x1, dtype=torch.float, device=device)
    train_sup_y = torch.tensor(train_sup_y, dtype=torch.float, device=device).view(
        -1, 1
    )
    train_unsup_x0 = torch.tensor(train_unsup_x0, dtype=torch.float, device=device)
    train_unsup_x1 = torch.tensor(train_unsup_x1, dtype=torch.float, device=device)
    train_unsup_y = torch.tensor(train_unsup_y, dtype=torch.float, device=device).view(
        -1, 1
    )
    test_sup_x0 = torch.tensor(test_sup_x0, dtype=torch.float, device=device)
    test_sup_x1 = torch.tensor(test_sup_x1, dtype=torch.float, device=device)
    test_sup_y = torch.tensor(test_sup_y, dtype=torch.float, device=device).view(-1, 1)
    test_unsup_x0 = torch.tensor(test_unsup_x0, dtype=torch.float, device=device)
    test_unsup_x1 = torch.tensor(test_unsup_x1, dtype=torch.float, device=device)
    test_unsup_y = torch.tensor(test_unsup_y, dtype=torch.float, device=device).view(
        -1, 1
    )

    train_unlabeled_losses = defaultdict(list)
    test_unlabeled_losses = defaultdict(list)
    train_labeled_losses = defaultdict(list)
    test_labeled_losses = defaultdict(list)
    for orthogonal_dir, intercept in zip(orthogonal_dirs, intercepts):
        classifier = ContrastPairClassifier(
            input_dim=train_sup_x1.shape[1],
            include_bias=True,
            device=device,
            verbose=True,
        )
        classifier.set_params(orthogonal_dir, intercept)
        # CCS loss on the unlabeled train set.
        train_unlabeled_loss_dict = classifier.compute_unsupervised_loss(
            train_unsup_x0,
            train_unsup_x1,
        )
        for k, v in train_unlabeled_loss_dict.items():
            train_unlabeled_losses[k].append(v.item())
        # CCS loss on the unlabeled test set.
        test_unlabeled_loss_dict = classifier.compute_unsupervised_loss(
            test_unsup_x0,
            test_unsup_x1,
        )
        for k, v in test_unlabeled_loss_dict.items():
            test_unlabeled_losses[k].append(v.item())
        # CCS loss on the labeled train set.
        train_labeled_loss_dict = classifier.compute_unsupervised_loss(
            train_sup_x0,
            train_sup_x1,
        )
        for k, v in train_labeled_loss_dict.items():
            train_labeled_losses[k].append(v.item())
        # CCS loss on the labeled test set.
        test_labeled_loss_dict = classifier.compute_unsupervised_loss(
            test_sup_x0,
            test_sup_x1,
        )
        for k, v in test_labeled_loss_dict.items():
            test_labeled_losses[k].append(v.item())

    # Choose the direction with the lowest CCS loss on the unlabeled combined
    # train and test sets. Scale the train and test losses by the number of
    # examples in each set.
    unlabeled_losses = train_unsup_y.shape[0] * np.array(
        train_unlabeled_losses["unsupervised_loss"]
    ) + test_unsup_y.shape[0] * np.array(test_unlabeled_losses["unsupervised_loss"])
    best_orthogonal_dir_idx = int(np.argmin(unlabeled_losses))
    best_orthogonal_dir = orthogonal_dirs[best_orthogonal_dir_idx]
    best_intercept = float(intercepts[best_orthogonal_dir_idx])
    best_classifier = ContrastPairClassifier(
        input_dim=train_sup_x1.shape[1],
        include_bias=True,
        device=device,
        verbose=True,
    )
    best_classifier.set_params(best_orthogonal_dir, best_intercept)

    # Make the fit result.
    fit_result = {
        "best_idx": best_orthogonal_dir_idx,
        "train_unlabeled_losses": train_unlabeled_losses,
        "test_unlabeled_losses": test_unlabeled_losses,
        "train_labeled_losses": train_labeled_losses,
        "test_labeled_losses": test_labeled_losses,
    }

    # Add the orthogonal directions to the final fit result if they were trained
    # in this run.
    if load_orthogonal_directions_run_dir is None:
        fit_result["lr_fit_results"] = lr_fit_results

    return best_classifier, fit_result, orthogonal_dirs, intercepts


def train_ccs_lr(
    data_dict: PrefixDataDictType,
    permutation_dict: PermutationDictType,
    unlabeled_train_data_dict: PromptIndicesDictType,
    labeled_train_data_dict: PromptIndicesDictType,
    projection_model: myReduction,
    labeled_prefix: str,
    unlabeled_prefix: str,
    train_kwargs={},
    project_along_mean_diff=False,
    device="cuda",
    logger=None,
) -> tuple[ContrastPairClassifier, dict]:
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

    train_kwargs = {
        k: v for k, v in train_kwargs.items() if k in FIT_CCS_LR_KWARGS_NAMES
    }

    # Train the model.
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
    classify_model = fit_result["best_probe"]

    return classify_model, fit_result


# def train_ccs_in_lr_span(
#     data_dict: PrefixDataDictType,
#     permutation_dict: PermutationDictType,
#     unlabeled_train_data_dict: PromptIndicesDictType,
#     labeled_train_data_dict: PromptIndicesDictType,
#     projection_model: myReduction,
#     labeled_prefix: str,
#     unlabeled_prefix: str,
#     num_orthogonal_directions: int,
#     train_kwargs={},
#     project_along_mean_diff=False,
#     device="cuda",
#     logger=None,
# ) -> tuple[ContrastPairClassifier, dict]:
#     # Labeled data.
#     (train_sup_x0, train_sup_x1), train_sup_y = make_contrast_pair_data(
#         target_dict=labeled_train_data_dict,
#         data_dict=data_dict[labeled_prefix],
#         permutation_dict=permutation_dict,
#         projection_model=projection_model,
#         split="train",
#         project_along_mean_diff=project_along_mean_diff,
#     )
#     (test_sup_x0, test_sup_x1), test_sup_y = make_contrast_pair_data(
#         target_dict=labeled_train_data_dict,
#         data_dict=data_dict[labeled_prefix],
#         permutation_dict=permutation_dict,
#         projection_model=projection_model,
#         split="test",
#         project_along_mean_diff=project_along_mean_diff,
#     )
#     # Unlabeled data.
#     (train_unsup_x0, train_unsup_x1), train_unsup_y = make_contrast_pair_data(
#         target_dict=unlabeled_train_data_dict,
#         data_dict=data_dict[unlabeled_prefix],
#         permutation_dict=permutation_dict,
#         projection_model=projection_model,
#         split="train",
#         project_along_mean_diff=project_along_mean_diff,
#     )
#     (test_unsup_x0, test_unsup_x1), test_unsup_y = make_contrast_pair_data(
#         target_dict=unlabeled_train_data_dict,
#         data_dict=data_dict[unlabeled_prefix],
#         permutation_dict=permutation_dict,
#         projection_model=projection_model,
#         split="test",
#         project_along_mean_diff=project_along_mean_diff,
#     )

#     train_kwargs_names = [
#         "n_tries",
#         "n_epochs",
#         "lr",
#         "opt",
#     ]
#     train_kwargs = {
#         k: v for k, v in train_kwargs.items() if k in train_kwargs_names
#     }
#     lr_train_kwargs = copy(train_kwargs)
#     lr_train_kwargs.update({"sup_weight": 1.0, "unsup_weight": 0.0})

#     orthogonal_dirs = None
#     lr_fit_results = []
#     for i in range(num_orthogonal_directions):
#         logger.info(f"Direction {i+1}/{num_orthogonal_directions}.")
#         fit_result = fit(
#             train_sup_x0,
#             train_sup_x1,
#             train_sup_y,
#             train_unsup_x0,
#             train_unsup_x1,
#             train_unsup_y,
#             test_sup_x0,
#             test_sup_x1,
#             test_sup_y,
#             test_unsup_x0,
#             test_unsup_x1,
#             test_unsup_y,
#             orthogonal_dirs=orthogonal_dirs,
#             include_bias=True,
#             verbose=True,
#             device=device,
#             logger=logger,
#             **lr_train_kwargs,
#         )

#         # [input_dim, 1]
#         new_orthogonal_dir = (
#             fit_result["best_probe"].linear.weight.detach().cpu().numpy()
#         ).T
#         new_orthogonal_dir = normalize(new_orthogonal_dir)
#         if orthogonal_dirs is None:
#             orthogonal_dirs = new_orthogonal_dir
#         else:
#             orthogonal_dirs = np.hstack([orthogonal_dirs, new_orthogonal_dir])

#         # Remove elements that are not JSON-serializable.
#         del fit_result["best_probe"]
#         del fit_result["all_probes"]
#         lr_fit_results.append(fit_result)

#     ccs_train_kwargs = copy(train_kwargs)
#     ccs_train_kwargs.update({"sup_weight": 0.0, "unsup_weight": 1.0})
#     final_fit_result = fit(
#         train_sup_x0,
#         train_sup_x1,
#         train_sup_y,
#         train_unsup_x0,
#         train_unsup_x1,
#         train_unsup_y,
#         test_sup_x0,
#         test_sup_x1,
#         test_sup_y,
#         test_unsup_x0,
#         test_unsup_x1,
#         test_unsup_y,
#         span_dirs=orthogonal_dirs,
#         include_bias=False,
#         verbose=True,
#         device=device,
#         logger=logger,
#         **ccs_train_kwargs,
#     )
#     final_fit_result["lr_fit_results"] = lr_fit_results
#     final_fit_result["orthogonal_dirs"] = orthogonal_dirs.tolist()

#     best_probe = final_fit_result["best_probe"]
#     final_fit_result["best_probe_weight"] = (
#         best_probe.linear.weight.detach().cpu().numpy().tolist()
#     )
#     if best_probe.linear.bias is not None:
#         final_fit_result["best_probe_bias"] = (
#             best_probe.linear.bias.detach().cpu().numpy().tolist()
#         )

#     return best_probe, final_fit_result
