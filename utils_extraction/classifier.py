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
    PermutationDictType,
    PrefixDataDictType,
    PromptIndicesDictType,
)
from utils_extraction.data_utils import getPair
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


def train_ccs_in_lr_span(
    data_dict: PrefixDataDictType,
    permutation_dict: PermutationDictType,
    unlabeled_train_data_dict: PromptIndicesDictType,
    labeled_train_data_dict: PromptIndicesDictType,
    projection_model: myReduction,
    labeled_prefix: str,
    unlabeled_prefix: str,
    num_orthogonal_dirs: int,
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
    lr_train_kwargs = copy(train_kwargs)
    lr_train_kwargs.update({"sup_weight": 1.0, "unsup_weight": 0.0})

    orthogonal_dirs = None
    lr_fit_results = []
    for i in range(num_orthogonal_dirs):
        logger.info(f"Finding {i}-th direction.")
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
            orthogonal_dirs=orthogonal_dirs,
            include_bias=True,
            verbose=True,
            device=device,
            logger=logger,
            **lr_train_kwargs,
        )

        # [input_dim, 1]
        new_orthogonal_dir = (
            fit_result["best_probe"].linear.weight.detach().cpu().numpy()
        ).T
        new_orthogonal_dir = normalize(new_orthogonal_dir)
        if orthogonal_dirs is None:
            orthogonal_dirs = new_orthogonal_dir
        else:
            orthogonal_dirs = np.hstack([orthogonal_dirs, new_orthogonal_dir])

        # Remove elements that are not JSON-serializable.
        del fit_result["best_probe"]
        del fit_result["all_probes"]
        lr_fit_results.append(fit_result)

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

    classify_model = final_fit_result["best_probe"]

    return classify_model, final_fit_result


# class ContrastPairClassifier(ABC):
#     def __init__(
#         self,
#         input_dim: int,
#         verbose=False,
#         include_bias=True,
#         no_train=False,
#         constraints=None,
#         logger=None,
#     ):
#         """Consistency classifier base class.

#         Args:
#             input_dim: dimension of the input data, not including the bias term.
#             verbose: whether to be verbose in train
#             include_bias: whether to include bias in the linear model
#             no_train: whether to train the linear model (otherwise just return randomly initialized weights)
#             constraints: an optional matrix of shape (n_directions, n_features)*
#                 of unnormalized but orthogonal directions which the linear model should be orthogonal to
#         """
#         self.input_dim = input_dim
#         self.theta_dim = input_dim + int(include_bias)
#         self.include_bias = include_bias
#         self.verbose = verbose
#         self.no_train = no_train
#         self.constraints = constraints
#         if self.constraints is not None:
#             self.constraints = normalize(self.constraints)
#             assert_close_to_orthonormal(self.constraints)
#         self.logger = logger

#         self.best_theta = None
#         self.best_loss = None

#     @abstractmethod
#     def validate_data(
#         self,
#         x1: ContrastPairNp,
#         y1: np.ndarray,
#         x2: Optional[ContrastPairNp] = None,
#         y2: Optional[np.ndarray] = None,
#     ):
#         pass

#     @classmethod
#     def from_coef_and_bias(cls, coef, bias=None, **kwargs):
#         coef = np.asarray(coef)
#         if coef.ndim == 1:
#             coef = coef[None, :]
#         elif coef.ndim > 2:
#             raise ValueError(
#                 f"coef should have at most 2 dimensions, found {coef.ndim}"
#             )

#         if bias is not None:
#             bias = np.asarray(bias)
#             if not (np.isscalar(bias) or bias.shape == (1,)):
#                 raise ValueError(f"bias should be a scalar, found {bias}")

#         kwargs["include_bias"] = bias is not None
#         kwargs["no_train"] = False
#         instance = cls(**kwargs)

#         if bias is None:
#             theta = coef
#         else:
#             theta = np.concatenate([coef, bias[:, None]], axis=-1)
#         instance.best_theta = theta
#         return instance

#     @property
#     def coef(self) -> Optional[np.ndarray]:
#         return self.best_theta[:, :-1] if self.best_theta is not None else None

#     @property
#     def bias(self) -> Optional[float]:
#         if self.best_theta is None or not self.include_bias:
#             return None
#         return self.best_theta[:, -1]

#     def maybe_add_ones_dimension(self, h):
#         """Maybe add a ones column for the bias term."""
#         if self.include_bias and h.shape[-1] == self.input_dim:
#             return np.concatenate([h, np.ones(h.shape[0])[:, None]], axis=-1)
#         else:
#             return h

#     @abstractmethod
#     def get_loss(
#         self,
#         probs1: tuple[Tensor, Tensor],
#         y1: Tensor,
#         probs2: Optional[tuple[Tensor, Tensor]] = None,
#         y2: Optional[Tensor] = None,
#     ):
#         """Compute the loss."""
#         pass

#     @abstractmethod
#     def get_losses(
#         self,
#         probs1: tuple[Tensor, Tensor],
#         y1: Tensor,
#         probs2: Optional[tuple[Tensor, Tensor]] = None,
#         y2: Optional[Tensor] = None,
#     ) -> collections.OrderedDict[str, float]:
#         """Return one or more loss terms."""
#         pass

#     def predict(
#         self, x: Tensor, theta: Optional[Tensor] = None
#     ) -> torch.Tensor:
#         """Predict the probability for the given x."""
#         if theta is None:
#             theta = self.best_theta
#         logit = torch.tensor(x.dot(theta.T))
#         return torch.sigmoid(logit)

#     def predict_from_class_probs(
#         self, p0: np.ndarray, p1: np.ndarray
#     ) -> np.ndarray:
#         """Predict class 1 probability from both class probabilities."""
#         return 0.5 * (p1 + (1 - p0))

#     def get_acc(self, theta_np: np.ndarray, x: ContrastPairNp, label) -> float:
#         """Compute the accuracy of a given direction for the data."""
#         x = [self.maybe_add_ones_dimension(d) for d in x]
#         p0, p1 = [self.predict(d, theta_np) for d in x]
#         avg_confidence = self.predict_probs(p0, p1)

#         label = label.reshape(-1)
#         predictions = (avg_confidence >= 0.5).astype(int)[:, 0]
#         return (predictions == label).mean()

#     def train(
#         self,
#         x1: ContrastPairNp,
#         y1: np.ndarray,
#         x2: Optional[ContrastPairNp] = None,
#         y2: Optional[np.ndarray] = None,
#         init_theta: Optional[torch.Tensor] = None,
#         n_epochs: int = 1000,
#         lr: float = 1e-2,
#         device="cuda",
#     ):
#         """Perform a single training run."""
#         # Initialize parameters
#         if init_theta is None:
#             init_theta = np.random.randn(self.theta_dim).reshape(1, -1)
#             init_theta = init_theta / np.linalg.norm(init_theta)
#         else:
#             init_theta = init_theta

#         init_theta = project_coeff(init_theta, self.constraints)

#         if self.no_train:
#             return init_theta, 0

#         theta = torch.tensor(
#             init_theta,
#             dtype=torch.float,
#             requires_grad=True,
#             device=device,
#         )

#         x1 = tuple(
#             torch.tensor(
#                 x, dtype=torch.float, requires_grad=False, device=device
#             )
#             for x in x1
#         )
#         if x2 is not None:
#             x2 = tuple(
#                 torch.tensor(
#                     x, dtype=torch.float, requires_grad=False, device=device
#                 )
#                 for x in x2
#             )

#         if self.constraints is not None:
#             constraints_t = torch.tensor(
#                 self.constraints,
#                 dtype=torch.float,
#                 requires_grad=False,
#                 device=device,
#             )
#         else:
#             constraints_t = None

#         # set up optimizer
#         optimizer = torch.optim.AdamW([theta], lr=lr)

#         losses = []
#         for _ in range(n_epochs):
#             # project onto theta
#             # TODO: does this affect the gradients?
#             theta_ = project_coeff(theta, constraints_t)

#             probs1 = (self.predict(x1[0], theta_), self.predict(x1[1], theta_))

#             if x2 is not None:
#                 probs2 = (
#                     self.predict(x2[0], theta_),
#                     self.predict(x2[1], theta_),
#                 )
#             else:
#                 probs2 = None

#             # get the corresponding loss
#             loss = self.get_loss(probs1, y1, probs2=probs2, y2=y2)

#             # update the parameters
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             # no gradient manipulation here
#             # TODO: don't use .data. See https://stackoverflow.com/questions/51743214/is-data-still-useful-in-pytorch
#             theta.data = project_coeff(theta.data, constraints_t)

#             loss_np = loss.detach().cpu().item()
#             losses.append(loss_np)

#         theta_np = theta.cpu().detach().numpy().reshape(1, -1)

#         return theta_np, losses

#     def fit(
#         self,
#         x1: ContrastPairNp,
#         y1: np.ndarray,
#         x2: Optional[ContrastPairNp] = None,
#         y2: Optional[np.ndarray] = None,
#         nepochs=1000,
#         ntries=10,
#         lr=1e-2,
#         init_theta=None,
#         device="cuda",
#     ):
#         """Fit the classifier to the data.

#         Args:
#             x1: Contrast pair data.
#             y1: Labels for x1.
#             x2: Optional second set of data for supervision.
#             y2: Labels for x2.
#         """
#         self.validate_data(x1, y1, x2, y2)
#         if init_theta is not None:
#             ntries = 1

#         self.best_loss = np.inf
#         self.best_theta = init_theta

#         best_acc = 0.0
#         losses = []
#         accs = []
#         accs1 = [] if x2 is not None else None
#         accs2 = [] if x2 is not None else None

#         x1 = tuple(self.maybe_add_ones_dimension(x) for x in x1)
#         if x2 is not None:
#             x2 = tuple(self.maybe_add_ones_dimension(x) for x in x2)

#         for _ in range(ntries):
#             theta_np, losses = self.train(
#                 x1,
#                 y1,
#                 x2=x2,
#                 y2=y2,
#                 init_theta=init_theta,
#                 n_epochs=nepochs,
#                 lr=lr,
#                 device=device,
#             )
#             acc1 = self.get_acc(theta_np, x1, y1)
#             if x2 is not None:
#                 acc2 = self.get_acc(theta_np, x2, y2)
#                 # TODO: maybe these should be weighted differently.
#                 acc = np.average([acc1, acc2], weights=[len(y1), len(y2)])
#                 accs1.append(acc1)
#                 accs2.append(acc2)
#             else:
#                 acc2 = None
#                 acc = acc1

#             accs.append(acc)
#             losses.extend(losses)

#             loss = losses[-1]
#             if loss < self.best_loss:
#                 if self.verbose and self.logger is not None:
#                     self.logger.debug(
#                         f"Found a new best theta. New loss: {format:.4f}, "
#                         f"new acc: {acc:.4f}"
#                     )
#                 self.best_theta = theta_np
#                 self.best_loss = loss
#                 best_acc = acc

#         return self.best_theta, self.best_loss, best_acc

#     def score(
#         self, x: ContrastPairNp, y: np.ndarray, get_loss=True, get_probs=True
#     ) -> tuple[
#         float,
#         Optional[collections.OrderedDict[str, float]],
#         Optional[np.ndarray],
#         Optional[np.ndarray],
#     ]:
#         x = tuple(self.maybe_add_ones_dimension(d) for d in x)
#         acc = self.get_acc(self.best_theta, x, y)

#         if get_probs or get_loss:
#             p0, p1 = [self.predict(d, self.best_theta) for d in x]
#         else:
#             p0 = None
#             p1 = None

#         if get_loss:
#             # TODO: handle case where the full loss requires x1, y1, x2, and y2,
#             # but only x1 and y1 are provided.
#             losses = {
#                 key: loss.cpu().detach().item()
#                 for key, loss in self.get_losses(
#                     (torch.tensor(p0), torch.tensor(p1)), y
#                 ).items()
#             }
#         else:
#             losses = None

#         return acc, losses, p0, p1

#     def get_train_loss(self):
#         return self.best_loss
