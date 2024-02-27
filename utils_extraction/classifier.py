from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Tensor = Union[torch.Tensor, np.ndarray]
ContrastPairNp = tuple[np.ndarray, np.ndarray]
ContrastPair = tuple[torch.Tensor, torch.Tensor]


def normalize(directions):
    return directions / np.linalg.norm(directions, axis=-1, keepdims=True)


def assert_close_to_orthonormal(directions, atol=1e-3):
    assert np.allclose(
        directions @ directions.T, np.eye(directions.shape[0]), atol=atol
    ), "Not orthonormal"


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


class ContrastPairClassifier(nn.Module):
    def __init__(
        self, input_dim: int, include_bias=True, device="cuda", verbose=False
    ):
        super(ContrastPairClassifier, self).__init__()
        self.input_dim = input_dim
        self.include_bias = include_bias
        self.device = device
        self.verbose = verbose
        self.linear = nn.Linear(input_dim, 1, bias=include_bias).to(device)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        logits = self.linear(x)
        probs = torch.sigmoid(logits)
        return probs

    def compute_loss(
        self,
        x_pos1: torch.Tensor,
        x_neg1: torch.Tensor,
        y1: torch.Tensor,
        x_pos2: torch.Tensor,
        x_neg2: torch.Tensor,
        unsup_weight=1.0,
    ) -> OrderedDict:
        supervised_loss = nn.BCELoss()(self.predict(x_pos1), y1) + nn.BCELoss()(
            self.predict(x_neg1), 1 - y1
        )

        p_pos2 = self.predict(x_pos2)
        p_neg2 = self.predict(x_neg2)
        consistency_loss = ((p_neg2 - (1 - p_pos2)) ** 2).mean()
        confidence_loss = torch.min(p_neg2, p_pos2).pow(2).mean()
        unsupervised_loss = consistency_loss + confidence_loss

        total_loss = supervised_loss + unsup_weight * unsupervised_loss

        return OrderedDict(
            [
                ("total_loss", total_loss),
                ("supervised_loss", supervised_loss),
                ("unsupervised_loss", unsupervised_loss),
                ("consistency_loss", consistency_loss),
                ("confidence_loss", confidence_loss),
            ]
        )

    def train_model(
        self,
        x_pos1: torch.Tensor,
        x_neg1: torch.Tensor,
        y1: torch.Tensor,
        x_pos2: torch.Tensor,
        x_neg2: torch.Tensor,
        n_epochs=1000,
        lr=1e-2,
        unsup_weight=1.0,
    ) -> List[OrderedDict]:
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_history = []
        self.train()

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            losses = self.compute_loss(
                x_pos1, x_neg1, y1, x_pos2, x_neg2, unsup_weight
            )
            losses["total_loss"].backward()
            optimizer.step()

            if self.verbose and (epoch + 1) % 100 == 0:
                print(
                    f"Epoch {epoch+1}/{n_epochs}, Loss: {losses['total_loss'].item()}"
                )

            loss_history.append(losses)

        return loss_history

    def evaluate_accuracy(
        self, x_pos: torch.Tensor, x_neg: torch.Tensor, y: torch.Tensor
    ) -> float:
        with torch.no_grad():
            self.eval()
            p_pos = self.predict(x_pos)
            p_neg = self.predict(x_neg)
            predictions = ((p_pos + 1 - p_neg) / 2).float()
            y = y.to(self.device).float().view(-1)
            accuracy = (predictions == y).float().mean().item()
        return accuracy


def fit(
    x_pos1,
    x_neg1,
    y1,
    x_pos2,
    x_neg2,
    y2,
    n_tries=10,
    n_epochs=1000,
    lr=1e-2,
    unsup_weight=1.0,
    verbose=False,
    device="cuda",
):
    best_loss = {"total_loss": float("inf")}
    best_probe = None
    all_probes = []
    all_losses = []
    all_accuracies1 = []
    all_accuracies2 = []

    x_pos1 = torch.tensor(x_pos1, dtype=torch.float, device=device)
    x_neg1 = torch.tensor(x_neg1, dtype=torch.float, device=device)
    y1 = torch.tensor(y1, dtype=torch.float, device=device).view(-1, 1)
    x_pos2 = torch.tensor(x_pos2, dtype=torch.float, device=device)
    x_neg2 = torch.tensor(x_neg2, dtype=torch.float, device=device)
    y2 = torch.tensor(y2, dtype=torch.float, device=device).view(-1, 1)

    for _ in range(n_tries):
        classifier = ContrastPairClassifier(
            input_dim=x_pos1.shape[1], device=device, verbose=verbose
        )
        loss_history = classifier.train_model(
            x_pos1, x_neg1, y1, x_pos2, x_neg2, n_epochs, lr, unsup_weight
        )
        final_loss = {k: loss.item() for k, loss in loss_history[-1].items()}
        accuracy1 = classifier.evaluate_accuracy(x_pos1, x_neg1, y1)
        accuracy2 = classifier.evaluate_accuracy(x_pos2, x_neg2, y2)

        all_probes.append(classifier)
        all_losses.append(final_loss)
        all_accuracies1.append(accuracy1)
        all_accuracies2.append(accuracy2)

        if final_loss["total_loss"] < best_loss["total_loss"]:
            best_loss = final_loss
            best_probe = classifier

    return {
        "best_probe": best_probe,
        "best_loss": best_loss,
        "best_accuracy1": max(all_accuracies1),
        "best_accuracy2": max(all_accuracies2),
        "all_probes": all_probes,
        "all_losses": all_losses,
        "all_accuracies1": all_accuracies1,
        "all_accuracies2": all_accuracies2,
    }


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
