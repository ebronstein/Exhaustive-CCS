from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from copy import copy
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.types import (
    DataDictType,
    Mode,
    PermutationDictType,
    PrefixDataDictType,
    PromptIndicesDictType,
)
from utils_extraction.data_utils import getPair
from utils_extraction.logistic_reg import LogisticRegressionClassifier
from utils_extraction.projection import myReduction

Tensor = Union[torch.Tensor, np.ndarray]
ContrastPairNp = tuple[np.ndarray, np.ndarray]
ContrastPair = tuple[torch.Tensor, torch.Tensor]


def normalize(directions):
    # directions shape: [input_dim, n_directions]
    return directions / np.linalg.norm(directions, axis=0, keepdims=True)


def assert_close_to_orthonormal(directions, atol=1e-3):
    # directions shape: [input_dim, n_directions]
    assert np.allclose(
        directions.T @ directions, np.eye(directions.shape[1]), atol=atol
    ), f"{directions} not orthonormal"


def project(x, along_directions):
    """Project x along the along_directions.

    x of shape (..., d) and along_directions of shape (n_directions, d)"""
    if isinstance(x, torch.Tensor) and isinstance(
        along_directions, torch.Tensor
    ):
        inner_products = torch.einsum("...d,nd->...n", x, along_directions)
        return x - torch.einsum(
            "...n,nd->...d", inner_products, along_directions
        )
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
        raise ValueError(
            "coef_and_bias should be either torch.Tensor or np.ndarray"
        )


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
    if directions.ndim == 1:
        directions = directions[:, None]
    elif directions.ndim != 2:
        raise ValueError("Directions should have 2 dimensions.")

    if directions.shape[0] != input_dim:
        raise ValueError(
            "Directions should have the same number of columns as input_dim."
        )

    directions = normalize(directions)
    assert_close_to_orthonormal(directions)
    return directions


class ContrastPairClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        orthogonal_dirs: Optional[np.ndarray] = None,
        span_dirs: Optional[np.ndarray] = None,
        include_bias=True,
        device="cuda",
        verbose=False,
    ):
        """CCS and LR classifier.

        Args:
            input_dim: Input dimension.
            orthogonal_dirs: [input_dim, n_directions]
            span_dirs: [input_dim, n_directions]
        """
        super(ContrastPairClassifier, self).__init__()
        self.input_dim = input_dim
        self.include_bias = include_bias
        self.device = device
        self.verbose = verbose

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
            self.linear = nn.Linear(input_dim, 1, bias=include_bias).to(device)
            self.project_params(self.orthogonal_dirs)

            self.span_dirs = None
        elif span_dirs is not None:
            if self.include_bias:
                raise ValueError(
                    "Cannot include bias if span_dirs is provided."
                )

            span_dirs = process_directions(span_dirs, input_dim)
            self.register_buffer(
                "span_dirs",
                torch.tensor(span_dirs, dtype=torch.float32).to(device),
            )
            self.linear = nn.Linear(span_dirs.shape[1], 1, bias=False).to(
                device
            )

            self.orthogonal_dirs = None
        else:
            self.linear = nn.Linear(input_dim, 1, bias=include_bias).to(device)
            self.orthogonal_dirs = None
            self.span_dirs = None

    def project_params(self, directions: torch.Tensor):
        # directions shape: [input_dim, n_directions]
        with torch.no_grad():
            params = self.linear.weight  # [1, input_dim]
            # [1, n_directions]
            inner_products = torch.matmul(params, directions)
            # [1, input_dim]
            proj_params = params - torch.matmul(
                inner_products, directions.permute(1, 0)
            )
            self.linear.weight = nn.Parameter(proj_params)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        if self.span_dirs is not None:
            logits = torch.matmul(x, self.linear(self.span_dirs))
        else:
            logits = self.linear(x)
        probs = torch.sigmoid(logits)
        return probs

    def compute_loss(
        self,
        sup_x0: torch.Tensor,
        sup_x1: torch.Tensor,
        sup_y: torch.Tensor,
        unsup_x0: torch.Tensor,
        unsup_x1: torch.Tensor,
        unsup_weight=1.0,
        sup_weight=1.0,
        # l2_weight=0.0,
    ) -> OrderedDict:
        bce_loss_0 = nn.BCELoss()(self.predict(sup_x0), 1 - sup_y)
        bce_loss_1 = nn.BCELoss()(self.predict(sup_x1), sup_y)
        supervised_loss = 0.5 * (bce_loss_0 + bce_loss_1)

        unsup_prob_0 = self.predict(unsup_x0)
        unsup_prob_1 = self.predict(unsup_x1)
        consistency_loss = ((unsup_prob_0 - (1 - unsup_prob_1)) ** 2).mean()
        confidence_loss = torch.min(unsup_prob_0, unsup_prob_1).pow(2).mean()
        unsupervised_loss = consistency_loss + confidence_loss

        # L2 regularization loss.
        # l2_reg_loss = torch.tensor(0.0).to(self.device)
        # for param in self.linear.parameters():
        #     l2_reg_loss += torch.norm(param) ** 2

        total_loss = (
            sup_weight * supervised_loss
            + unsup_weight * unsupervised_loss
            # + l2_weight * l2_reg_loss
        )

        return OrderedDict(
            [
                ("total_loss", total_loss),
                ("supervised_loss", supervised_loss),
                ("unsupervised_loss", unsupervised_loss),
                # ("l2_reg_loss", l2_reg_loss),
                ("bce_loss_0", bce_loss_0),
                ("bce_loss_1", bce_loss_1),
                ("consistency_loss", consistency_loss),
                ("confidence_loss", confidence_loss),
            ]
        )

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
        opt: str = "sgd",
        eval_freq: int = 20,
        logger=None,
    ) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        if opt == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=0)
        elif opt == "adam":
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0)
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

            if self.verbose and (epoch + 1) % 100 == 0 and logger is not None:
                logger.info(
                    f"Epoch {epoch+1}/{n_epochs}, Loss: {train_losses['total_loss'].item()}"
                )

            train_history["weight_norm"].append(
                self.linear.weight.norm().item()
            )
            if self.include_bias:
                train_history["bias"].append(self.linear.bias.item())

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


def fit(
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
    n_tries=10,
    n_epochs=1000,
    lr=1e-2,
    unsup_weight=1.0,
    sup_weight=1.0,
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
    train_sup_y = torch.tensor(
        train_sup_y, dtype=torch.float, device=device
    ).view(-1, 1)
    train_unsup_x0 = torch.tensor(
        train_unsup_x0, dtype=torch.float, device=device
    )
    train_unsup_x1 = torch.tensor(
        train_unsup_x1, dtype=torch.float, device=device
    )
    train_unsup_y = torch.tensor(
        train_unsup_y, dtype=torch.float, device=device
    ).view(-1, 1)
    test_sup_x0 = torch.tensor(test_sup_x0, dtype=torch.float, device=device)
    test_sup_x1 = torch.tensor(test_sup_x1, dtype=torch.float, device=device)
    test_sup_y = torch.tensor(
        test_sup_y, dtype=torch.float, device=device
    ).view(-1, 1)
    test_unsup_x0 = torch.tensor(
        test_unsup_x0, dtype=torch.float, device=device
    )
    test_unsup_x1 = torch.tensor(
        test_unsup_x1, dtype=torch.float, device=device
    )
    test_unsup_y = torch.tensor(
        test_unsup_y, dtype=torch.float, device=device
    ).view(-1, 1)

    for _ in range(n_tries):
        classifier = ContrastPairClassifier(
            input_dim=train_sup_x1.shape[1],
            orthogonal_dirs=orthogonal_dirs,
            span_dirs=span_dirs,
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


def train_ccs_in_lr_span(
    data_dict: PrefixDataDictType,
    permutation_dict: PermutationDictType,
    unlabeled_train_data_dict: PromptIndicesDictType,
    labeled_train_data_dict: PromptIndicesDictType,
    projection_model: myReduction,
    labeled_prefix: str,
    unlabeled_prefix: str,
    num_orthogonal_dirs: int,
    mode: Mode,
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

    train_kwargs_names = [
        "n_tries",
        "n_epochs",
        "lr",
        "opt",
    ]
    train_kwargs = {
        k: v for k, v in train_kwargs.items() if k in train_kwargs_names
    }

    lr_train_kwargs = train_kwargs.pop("log_reg", {})
    lr_model = LogisticRegressionClassifier(
        n_jobs=1, **lr_train_kwargs
    )

    cur_train_sup_x0 = train_sup_x0.copy()
    cur_train_sup_x1 = train_sup_x1.copy()

    orthogonal_dirs = []
    lr_fit_results = []
    for i in range(num_orthogonal_dirs):
        logger.info(f"Direction {i+1}/{num_orthogonal_dirs}.")
        breakpoint()

        lr_model.fit((cur_train_sup_x0, cur_train_sup_x1), train_sup_y, mode)
        orth_dir = lr_model.coef_ / np.linalg.norm(lr_model.coef_)
        orth_dir = orth_dir.squeeze(0)
        orthogonal_dirs.append(orth_dir)

        # Eval
        fit_result = []
        fit_result["proj_train_acc"] = lr_model.score((cur_train_sup_x0, cur_train_sup_x1), train_sup_y, mode)[0]
        fit_result["train_acc"] = lr_model.score((train_sup_x0, train_sup_x1), train_sup_y, mode)[0]
        fit_result["test_acc"] = lr_model.score((test_sup_x0, test_sup_x1), test_sup_y, mode)[0]
        lr_fit_results.append(fit_result)

        # Project away the direction.
        cur_train_sup_x0 -= ((cur_train_sup_x0 @ orth_dir)[:, None] * orth_dir)
        cur_train_sup_x1 -= ((cur_train_sup_x0 @ orth_dir)[:, None] * orth_dir)
        assert np.allclose(cur_train_sup_x0 @ orth_dir, 0)
        assert np.allclose(cur_train_sup_x1 @ orth_dir, 0)


    ccs_train_kwargs = copy(train_kwargs)
    ccs_train_kwargs.update({"sup_weight": 0.0, "unsup_weight": 1.0})
    final_fit_result = fit(
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
        include_bias=False,
        verbose=True,
        device=device,
        logger=logger,
        **ccs_train_kwargs,
    )
    final_fit_result["lr_fit_results"] = lr_fit_results
    final_fit_result["orthogonal_dirs"] = orthogonal_dirs.tolist()

    best_probe = final_fit_result["best_probe"]
    final_fit_result["best_probe_weight"] = (
        best_probe.linear.weight.detach().cpu().numpy().tolist()
    )
    if best_probe.linear.bias is not None:
        final_fit_result["best_probe_bias"] = (
            best_probe.linear.bias.detach().cpu().numpy().tolist()
        )

    return best_probe, final_fit_result

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

    train_kwargs_names = [
        "n_tries",
        "n_epochs",
        "lr",
        "unsup_weight",
        "sup_weight",
        "opt",
    ]
    train_kwargs = {
        k: v for k, v in train_kwargs.items() if k in train_kwargs_names
    }

    # Train the model.
    fit_result = fit(
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
#     num_orthogonal_dirs: int,
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
#     for i in range(num_orthogonal_dirs):
#         logger.info(f"Direction {i+1}/{num_orthogonal_dirs}.")
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
