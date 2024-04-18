import os
import time
import typing
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from utils.types import (
    DataDictType,
    Mode,
    PermutationDictType,
    PrefixDataDictType,
    PromptIndicesDictType,
)
from utils_extraction import load_utils, metrics
from utils_extraction.classifier import (
    assert_close_to_orthonormal,
    make_contrast_pair_data,
    normalize,
    project_coeff,
    train_ccs_in_lr_span,
    train_ccs_lr,
    train_ccs_lr_in_span,
    train_ccs_select_lr,
)
from utils_extraction.data_utils import getConcat, getPair
from utils_extraction.logistic_reg import LogisticRegressionClassifier
from utils_extraction.projection import IdentityReduction, myReduction
from utils_extraction.pseudo_label import train_pseudo_label

UNSUPERVISED_METHODS = ("TPC", "KMeans", "BSS", "CCS", "Random")
SUPERVISED_METHODS = ("LR", "CCS+LR")

EvalClassificationMethodType = Literal[
    "LR",
    "BSS",
    "CCS",
    "CCS+LR",
    "CCS-in-LR-span",
    "CCS+LR-in-span",
    "CCS-select-LR",
    "pseudolabel",
]


def is_method_unsupervised(method):
    return method in UNSUPERVISED_METHODS or method.startswith("RCCS")


def getSingleLoss(x, verbose=False):
    # x: shape (n, 1)
    x1 = x[x < 0]
    x2 = x[x >= 0]

    if verbose:
        print(
            "var(x1) = {}, var(x2) = {}, var(x) = {}".format(
                x1.var(), x2.var(), x.var()
            )
        )
    return (x1.var() + x2.var()) / x.var()


def getLoss(z, weights, verbose=False):
    # weighted loss according to `weights`
    return sum([u * getSingleLoss(x, verbose) for u, x in zip(weights, z)])


def get_all_data(data_dict):
    all_data, all_labels = [], []
    for dataset in data_dict.keys():
        raw_data = np.concatenate([w[0] for w in data_dict[dataset]], axis=0)
        label = np.concatenate([w[1] for w in data_dict[dataset]])

        all_data.append(raw_data)
        all_labels.append(label)
    all_data, all_labels = np.concatenate(all_data), np.concatenate(all_labels)

    hs0, hs1 = (
        all_data[:, : all_data.shape[-1] // 2],
        all_data[:, all_data.shape[-1] // 2 :],
    )

    return hs0, hs1, all_labels


class ConsistencyMethod(object):
    def __init__(
        self, verbose=False, include_bias=True, no_train=False, constraints=None
    ):
        """The main CCS class
        verbose: whether to be verbose in train
        include_bias: whether to include bias in the linear model
        no_train: whether to train the linear model (otherwise just return randomly initialized weights)
        constraints: an optional matrix of shape (n_directions, n_features)*
            of unormalized but orthogonal directions which the linear model should be orthogonal to
        """
        self.include_bias = include_bias
        self.verbose = verbose
        self.no_train = no_train
        self.constraints = constraints
        if self.constraints is not None:
            self.constraints = normalize(self.constraints)
            assert_close_to_orthonormal(self.constraints)

        self.best_theta = None
        self.best_loss = None

    @classmethod
    def from_coef_and_bias(cls, coef, bias=None, **kwargs):
        coef = np.asarray(coef)
        if coef.ndim == 1:
            coef = coef[None, :]
        elif coef.ndim > 2:
            raise ValueError(
                f"coef should have at most 2 dimensions, found {coef.ndim}"
            )

        if bias is not None:
            bias = np.asarray(bias)
            if not (np.isscalar(bias) or bias.shape == (1,)):
                raise ValueError(f"bias should be a scalar, found {bias}")

        kwargs["include_bias"] = bias is not None
        kwargs["no_train"] = False
        instance = cls(**kwargs)

        if bias is None:
            theta = coef
        else:
            theta = np.concatenate([coef, bias[:, None]], axis=-1)
        instance.best_theta = theta
        return instance

    @property
    def coef(self) -> Optional[np.ndarray]:
        return self.best_theta[:, :-1] if self.best_theta is not None else None

    @property
    def bias(self) -> Optional[float]:
        if self.best_theta is None or not self.include_bias:
            return None
        return self.best_theta[:, -1]

    def add_ones_dimension(self, h):
        if self.include_bias:
            return np.concatenate([h, np.ones(h.shape[0])[:, None]], axis=-1)
        else:
            return h

    def get_confidence_loss(self, p0, p1):
        """
        Assumes p0 and p1 are each a tensor of probabilities of shape (n,1) or (n,)
        Assumes p0 is close to 1-p1
        Encourages p0 and p1 to be close to 0 or 1 (far from 0.5)
        """
        min_p = torch.min(p0, p1)
        return (min_p**2).mean(0)
        # return (min_p).mean(0)**2  # seems a bit worse

    def get_similarity_loss(self, p0, p1):
        """
        Assumes p0 and p1 are each a tensor of probabilities of shape (n,1) or (n,)
        Encourages p0 to be close to 1-p1 and vice versa
        """
        return ((p0 - (1 - p1)) ** 2).mean(0)

    def get_loss(self, p0, p1):
        """
        Returns the ConsistencyModel loss for two probabilities each of shape (n,1) or (n,)
        p0 and p1 correspond to the probabilities
        """
        similarity_loss = self.get_similarity_loss(p0, p1)
        confidence_loss = self.get_confidence_loss(p0, p1)

        return similarity_loss + confidence_loss

    def get_losses(self, p0, p1):
        """Returns loss, similarity_loss, confidence_loss"""
        similarity_loss = self.get_similarity_loss(p0, p1)
        confidence_loss = self.get_confidence_loss(p0, p1)

        return (
            similarity_loss + confidence_loss,
            similarity_loss,
            confidence_loss,
        )

    # return the probability tuple (p0, p1)
    def transform(self, data: list, theta_np=None) -> tuple[np.ndarray, np.ndarray]:
        if theta_np is None:
            theta_np = self.best_theta
        z0, z1 = torch.tensor(
            self.add_ones_dimension(data[0]).dot(theta_np.T)
        ), torch.tensor(self.add_ones_dimension(data[1]).dot(theta_np.T))
        p0, p1 = torch.sigmoid(z0).numpy(), torch.sigmoid(z1).numpy()

        return p0, p1

    def predict_probs(self, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        """Predicts class 1 probabilities for data."""
        return 0.5 * (p1 + (1 - p0))

    def get_acc(self, theta_np, data: list, label) -> float:
        """
        Computes the accuracy of a given direction theta_np represented as a numpy array
        """
        p0, p1 = self.transform(data, theta_np)
        avg_confidence = self.predict_probs(p0, p1)

        label = label.reshape(-1)
        predictions = (avg_confidence >= 0.5).astype(int)[:, 0]
        return (predictions == label).mean()

    def train(self):
        """
        Does a single training run of n_epochs epochs
        """

        # convert to tensors
        x0 = torch.tensor(
            self.x0, dtype=torch.float, requires_grad=False, device=self.device
        )
        x1 = torch.tensor(
            self.x1, dtype=torch.float, requires_grad=False, device=self.device
        )

        # initialize parameters
        if self.init_theta is None:
            init_theta = np.random.randn(self.d).reshape(1, -1)
            init_theta = init_theta / np.linalg.norm(init_theta)
        else:
            init_theta = self.init_theta

        init_theta = project_coeff(init_theta, self.constraints)

        if self.no_train:
            return init_theta, 0

        theta = torch.tensor(
            init_theta,
            dtype=torch.float,
            requires_grad=True,
            device=self.device,
        )

        if self.constraints is not None:
            constraints_t = torch.tensor(
                self.constraints,
                dtype=torch.float,
                requires_grad=False,
                device=self.device,
            )
        else:
            constraints_t = None

        # set up optimizer
        optimizer = torch.optim.AdamW([theta], lr=self.lr)

        # Start training (full batch)
        for _ in range(self.n_epochs):

            # project onto theta
            theta_ = project_coeff(theta, constraints_t)
            z0, z1 = x0.mm(theta_.T), x1.mm(theta_.T)

            # sigmoide to get probability
            p0, p1 = torch.sigmoid(z0), torch.sigmoid(z1)

            # get the corresponding loss
            loss = self.get_loss(p0, p1)

            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # with torch.no_grad():
            #     theta /= torch.norm(theta)

            # no gradient manipulation here
            theta.data = project_coeff(theta.data, constraints_t)

            theta_np = theta.cpu().detach().numpy().reshape(1, -1)
            # print("Norm of theta is " + str(np.linalg.norm(theta_np)))
            loss_np = loss.detach().cpu().item()

        return theta_np, loss_np

    def validate_data(self, data):
        assert len(data) == 2 and data[0].shape == data[1].shape

    def get_train_loss(self):
        return self.best_loss

    def visualize(self, losses, accs):
        plt.scatter(losses, accs)
        plt.xlabel("Loss")
        plt.ylabel("Accuracy")
        plt.show()

    # seems 50, 20 can significantly reduce overfitting than 1000, 10
    # switch back to 1000 + 10
    def fit(
        self,
        data: tuple[np.ndarray, np.ndarray],
        label: np.ndarray,
        n_epochs=1000,
        n_tries=10,
        lr=1e-2,
        init_theta=None,
        device="cuda",
    ):
        """
        Does n_tries attempts at training, with different random initializations
        """

        self.n_epochs = n_epochs
        self.n_tries = n_tries
        self.lr = lr

        self.device = device

        self.init_theta = init_theta
        if self.init_theta is not None:
            self.n_tries = 1

        if self.verbose:
            print(
                "String fiting data with Prob. n_epochs: {}, n_tries: {}, lr: {}".format(
                    n_epochs, n_tries, lr
                )
            )
        # set up the best loss and best theta found so far
        self.best_loss = np.inf
        self.best_theta = self.init_theta

        best_acc = 0.5
        losses, accs = [], []
        self.validate_data(data)

        self.x0 = self.add_ones_dimension(data[0])
        self.x1 = self.add_ones_dimension(data[1])
        self.y = label.reshape(-1)
        self.d = self.x0.shape[-1]

        for _ in range(self.n_tries):
            # train
            theta_np, loss = self.train()

            # evaluate
            acc = self.get_acc(theta_np, data, label)

            # save
            losses.append(loss)
            accs.append(acc)

            # see if it's the best run so far
            if loss < self.best_loss:
                if self.verbose:
                    print(
                        "Found a new best theta. New loss: {:.4f}, new acc: {:.4f}".format(
                            loss, acc
                        )
                    )
                self.best_theta = theta_np
                self.best_loss = loss
                best_acc = acc

        if self.verbose:
            self.visualize(losses, accs)

        return self.best_theta, self.best_loss, best_acc

    def score(self, data: list, label, get_loss=True, get_probs=True) -> tuple[
        float,
        Optional[list[float]],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        self.validate_data(data)
        acc = self.get_acc(self.best_theta, data, label)

        if get_probs or get_loss:
            p0, p1 = self.transform(data, self.best_theta)
            probs = self.predict_probs(p0, p1)
        else:
            p0 = None
            p1 = None

        if get_loss:
            losses = [
                loss.cpu().detach().item()
                for loss in self.get_losses(torch.tensor(p0), torch.tensor(p1))
            ]
        else:
            losses = None

        return acc, losses, probs, p0, p1


CLASSIFICATION_METHODS = ["TPC", "LR", "BSS", "KMeans"]


class myClassifyModel(LogisticRegression):
    def __init__(self, method, print_more=False):
        if method not in CLASSIFICATION_METHODS:
            raise ValueError(
                f"currently only support method to be `TPC`, `LR`, 'KMeans` and `BSS`! Got {method}"
            )
        self.method = method
        super(myClassifyModel, self).__init__(max_iter=10000, n_jobs=1, C=0.1)
        self.print_more = print_more

    @classmethod
    def from_coef_and_bias(cls, method, coef, bias=None, **kwargs):
        if method != "LR":
            raise ValueError(
                "Only logistical regression classification model can be "
                f"initialized with a coefficient and intercept, got {method}"
            )

        instance = cls(method, **kwargs)
        instance.set_params(coef, bias)
        return instance

    def set_params(self, coef, bias):
        self.classes_ = np.array([0, 1])
        self.intercept_ = bias
        self.coef_ = coef

    def get_train_loss(self):
        assert self.method == "BSS", NotImplementedError(
            "`get_train_loss` supported only when method is `BSS`."
        )
        return self.loss

    def fit(
        self,
        data,
        label,
        times=20,
        use_scheduler=False,
        weights=None,
        lr=1e-1,
        n_epochs=20,
        device="cuda",
    ):
        if self.method == "LR":
            super().fit(data, label)
            if self.print_more:
                print(
                    "fitting to {} data, acc is {}".format(
                        len(label), self.score(data, label)
                    )
                )

        elif self.method == "TPC":
            assert (
                data.shape[1] == 1
            ), "When `avg` mode is used, #hidden_dim is expected to be 1, but it's {}".format(
                data.shape[1]
            )
            self.avg = 0.0
            self.sign = 1

            debias = (data > 0).reshape(label.shape).astype(int)
            if np.sum(debias == label) / label.shape[0] < 0.5:
                self.sign = -1

            # set to model parameters
            self.set_params(np.array(self.sign).reshape(1, 1), -self.sign * self.avg)

        elif self.method == "KMeans":
            self.model = KMeans(n_clusters=2)
            self.model.fit(data)
            if self.print_more:
                print(
                    "fitting to {} data, acc is {}".format(
                        len(label), self.score(data, label)
                    )
                )

        elif self.method == "BSS":  # in this case, `data` will be a list
            assert (
                type(data) == list
            ), "When using BSS mode, data should be a list instead of {}".format(
                type(data)
            )

            x = [torch.tensor(w, device=device) for w in data]
            dim = data[0].shape[1]  # hidden dimension

            if weights == None:
                weights = [1 / len(x) for _ in range(len(x))]
            else:
                assert type(weights) == list and len(weights) == len(
                    x
                ), "Length of `weights` mismatches length of `data`."
                weights = [w / sum(weights) for w in weights]  # normalize

            sample_weight = [
                u / w.shape[0] for u, w in zip(weights, data) for _ in range(w.shape[0])
            ]

            minloss = 1.0
            final_coef = np.random.randn(dim).reshape(1, -1)
            final_bias = 0.0
            for _ in range(times):
                init_theta = np.random.randn(dim).reshape(1, -1)
                init_theta /= np.linalg.norm(init_theta)

                theta = torch.tensor(
                    init_theta,
                    dtype=torch.float,
                    requires_grad=True,
                    device=device,
                )
                optimizer = torch.optim.AdamW([theta], lr=lr)
                if use_scheduler:
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        factor=0.5,
                        verbose=self.print_more,
                        min_lr=1e-6,
                    )

                for epoch in range(n_epochs):

                    z = [w @ theta.T for w in x]

                    loss = getLoss(z, weights)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        theta /= torch.norm(theta)

                    if use_scheduler:
                        scheduler.step(loss)

                    if ((epoch + 1) % 50 == 0 and self.print_more) or epoch in [
                        0,
                        n_epochs - 1,
                    ]:
                        theta_np = (
                            theta.cpu().detach().numpy().reshape(1, -1)
                        )  # same as coef

                        projected, gth = np.concatenate(
                            [w @ theta_np.T for w in data]
                        ).reshape(-1), np.concatenate(label).reshape(-1)

                        self.avg = 0.0
                        self.sign = 1
                        debias = (projected > 0).reshape(gth.shape).astype(int)
                        if np.sum(debias == gth) / gth.shape[0] < 0.5:
                            self.sign = -1

                        # set to model parameters
                        self.set_params(self.sign * theta_np, -self.sign * self.avg)
                        acc = self.score(
                            np.concatenate(data, axis=0),
                            np.concatenate(label),
                            sample_weight,
                        )
                        # acc = np.mean([self.score(u, v) for u,v in zip(data, label)])
                        # if self.print_more:
                        #     print("epoch {} acc: {:.2f}, loss: {:.4f}".format(epoch, 100 * acc, loss))

                # check whether this time gives a lower loss
                with torch.no_grad():
                    z = [w @ theta.T for w in x]
                    # if weights is None:
                    loss = sum([getSingleLoss(w, False) for w in z]) / len(z)
                    loss = loss.detach().cpu().item()
                    if loss < minloss:
                        if self.print_more:
                            print(
                                "update params, acc is {:.2f}, old loss is {:.4f}, new loss is {:.4f}".format(
                                    100
                                    * self.score(
                                        np.concatenate(data, axis=0),
                                        np.concatenate(label),
                                        sample_weight,
                                    ),
                                    minloss,
                                    loss,
                                )
                            )
                        minloss = loss
                        final_coef = self.coef_
                        final_bias = self.intercept_

            # update loss
            self.loss = minloss
            self.set_params(final_coef, final_bias)

    def predict_probs(self, data):
        """Predicts class probabilities for data."""
        if self.method == "KMeans":
            return self.model.predict(data)
        else:
            # Return the probability of class 1
            return super().predict_proba(data)[..., 1]

    def score(
        self,
        data,
        label,
        get_loss=False,
        get_probs=False,
        sample_weight=None,
    ) -> tuple[
        float,
        Optional[list[float]],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        if self.method == "KMeans":
            prediction = self.model.predict(data)
            acc = np.mean(prediction == label)
            return acc, prediction, None, None

        if get_probs:
            probs = super().predict_proba(data)
            p0 = probs[:, 0]
            p1 = probs[:, 1]
            # Return the probability of class 1
            probs = p1
        else:
            probs = None
            p0 = None
            p1 = None

        if sample_weight is not None:
            acc = super().score(data, label, sample_weight)
        else:
            acc = super().score(data, label)

        if get_loss:
            if self.method == "BSS":
                loss = getSingleLoss(data @ self.coef_.T + self.intercept_)
                # Total loss is `loss`, consistency and confidence
                # loss do not apply to BSS.
                losses = (loss, 0.0, 0.0)
            else:
                losses = None

        return acc, losses, probs, p0, p1


def mainResults(
    data_dict: PrefixDataDictType,
    train_data_dict: PromptIndicesDictType,
    permutation_dict: PermutationDictType,
    test_dict: PromptIndicesDictType,
    projection_dict: PromptIndicesDictType,
    mode: Mode,
    train_prefix: str,
    test_prefix: str,
    labeled_train_data_dict: Optional[PromptIndicesDictType] = None,
    projection_method="PCA",
    n_components: int = 2,
    projection_only=False,
    classification_method="BSS",
    print_more=False,
    train_kwargs={},
    save_probs=True,
    test_on_train=False,
    constraints=None,
    project_along_mean_diff=False,
    device="cuda",
    run_dir: Optional[str] = None,
    seed: Optional[int] = None,
    run_id: Optional[int] = None,
    save_orthogonal_directions=False,
    load_orthogonal_directions_run_dir: Optional[str] = None,
    projected_sgd: bool = False,
    logger=None,
):
    """
    Calculate the main results.

    Args:
        data_dict (dict): Dictionary of hidden states and labels for each
            dataset and prefix combination. First key is dataset name, second
            key is prefix. Each value is a list with one element per prompt.
            Each element is a tuple pair of (hidden_states, labels).
        train_data_dict: Dictionary mapping from train dataset names to prompt
            indices to use for the main train datasets.
        permutation_dict (dict): Dictionary of train/test split indices. Key is
            dataset name, value is a tuple pair containing the train split and
            test split indices.
        projection_dict (dict): Dictionary of prompt indices for projection.
            Key is dataset name, each value is a list of prompt indices used to
            do projection.
        test_dict (dict): Dictionary of test datasets and prompt indices on
            which the method is evaluated. Key is dataset name, each value is a
            list of prompt indices used to do evaluation.
        mode (str): Hidden states data mode.
        train_prefix (str): Prefix for train data.
        test_prefix (str): Prefix for test data.
        labeled_train_data_dict (dict, optional): Dictionary mapping from
            labeled train dataset names to prompt indices to use.
        projection_method (str, optional): Projection method. Defaults to "PCA".
        n_components (int, optional): The dimension you want to reduce to. -1 means no
            projection will be implemented. Defaults to 2.
        projection_only (bool, optional): When set to true, will immediately return after
            training the projection_model. res and classify_model will be None. Defaults
            to False.
        classification_method (str, optional): Classification method. Can be LR, TPC, and
            BSS. Defaults to "BSS".
        print_more (bool, optional): Whether to print more information. Defaults to False.
        learn_dict (dict, optional): Keyword arguments passed to the training
            function. Defaults to {}.
        save_probs (bool, optional): Whether to save probabilities. Defaults to True.
        test_on_train (bool, optional): If true, will use the train set to test the model.
            Defaults to False.
        constraints (None, optional): Constraints to do the projection (CCS only).
            Defaults to None.
        project_along_mean_diff (bool, optional): If true, will project the data along the
            mean difference of the two classes. Defaults to False.
        run_dir (str, optional): Root directory for the Sacred Run. Defaults to None.
        seed (int, optional): Seed for random number generation. Defaults to None.
        run_id (int, optional): Sacred Run ID. Defaults to None.
        load_orthogonal_directions_run_dir: Run directory from which to load the
            orthogonal directions. If provided, expects the file
            "{load_orthogonal_directions_run_dir}/train/orthogonal_directions.npy"
            to exist.
        projected_sgd (bool, optional): Whether to use projected SGD. If True,
            the orthogonal directions are projected out of the parameters after
            each update.
    """
    if print_more:
        print(
            "Projection method: {} (n_con = {}) in {}\nClassification method: {} in: {}".format(
                projection_method,
                n_components,
                projection_dict,
                classification_method,
                test_dict,
            )
        )

    no_train = False
    if classification_method == "Random":
        no_train = True
        classification_method = "CCS"
    if (
        classification_method not in ["CCS", "CCS+LR", "CCS-in-LR-span"]
        and constraints is not None
    ):
        raise ValueError("constraints only supported for CCS-based methods.")

    # Get the train and test prefix data.
    train_prefix_data_dict = data_dict[train_prefix]
    test_prefix_data_dict = data_dict[test_prefix]

    # TODO: should the projection be fit to the train prefix, test prefix, or
    # both?
    # Concatenate all the data (not split) to do PCA.
    # Shape: [num_total_samples, num_features]. If there is a constant number of
    # prompts per dataset, num_total_samples = num_datasets * num_prompts
    # * num_samples_per_prompt.
    proj_states = getConcat(
        [
            getConcat([train_prefix_data_dict[key][w][0] for w in lis])
            for key, lis in projection_dict.items()
        ]
    )
    projection_model = myReduction(
        method=projection_method,
        n_components=n_components,
        print_more=print_more,
    )
    projection_model.fit(proj_states)

    if projection_only:
        return None, projection_model, None

    # pairFunc = partial(getPair, data_dict = data_dict, permutation_dict = permutation_dict, projection_model = projection_model)

    # TODO: standardize fit_result.
    fit_result = None
    if classification_method == "CCS":
        # TODO: implement different weights for CCS loss terms.
        if train_kwargs.get("consistency_weight", 1.0) != 1.0:
            raise NotImplementedError(
                "consistency_weight != 1.0 not implemented for CCS method."
            )
        if train_kwargs.get("confidence_weight", 1.0) != 1.0:
            raise NotImplementedError(
                "confidence_weight != 1.0 not implemented for CCS method."
            )

        data, labels = make_contrast_pair_data(
            target_dict=train_data_dict,
            data_dict=train_prefix_data_dict,
            permutation_dict=permutation_dict,
            projection_model=projection_model,
            split="train",
            project_along_mean_diff=project_along_mean_diff,
            split_pair=mode == "concat",
        )

        include_bias = train_kwargs.pop("include_bias", True)
        classify_model = ConsistencyMethod(
            include_bias=include_bias,
            no_train=no_train,
            verbose=print_more,
            constraints=constraints,
        )
        ccs_train_kwargs_names = ["n_epochs", "n_tries", "lr"]
        ccs_train_kwargs = {
            k: train_kwargs[k] for k in ccs_train_kwargs_names if k in train_kwargs
        }
        classify_model.fit(data=data, label=labels, **ccs_train_kwargs)
    elif classification_method == "CCS+LR":
        if labeled_train_data_dict is None:
            raise ValueError("labeled_train_data_dict must be provided for CCS+LR.")

        # Use train_prefix for the labeled data and test_prefix for the
        # unlabeled data.
        classify_model, fit_result = train_ccs_lr(
            data_dict,
            permutation_dict,
            train_data_dict,
            labeled_train_data_dict,
            projection_model,
            train_prefix,
            test_prefix,
            train_kwargs=train_kwargs,
            project_along_mean_diff=project_along_mean_diff,
            device=device,
            logger=logger,
        )
    elif classification_method == "CCS-in-LR-span":
        if labeled_train_data_dict is None:
            raise ValueError(
                "labeled_train_data_dict must be provided for CCS-in-LR-span."
            )
        if "num_orthogonal_directions" not in train_kwargs:
            raise ValueError(
                "num_orthogonal_directions required for CCS-in-LR-span method."
            )
        num_orthogonal_directions = train_kwargs.pop("num_orthogonal_directions")

        # Use train_prefix for the labeled data and test_prefix for the
        # unlabeled data.
        classify_model, fit_result, orthogonal_dirs, intercepts = train_ccs_in_lr_span(
            data_dict,
            permutation_dict,
            train_data_dict,
            labeled_train_data_dict,
            projection_model,
            train_prefix,
            test_prefix,
            num_orthogonal_directions,
            mode,
            load_orthogonal_directions_run_dir=load_orthogonal_directions_run_dir,
            train_kwargs=train_kwargs,
            project_along_mean_diff=project_along_mean_diff,
            device=device,
            logger=logger,
        )

        if save_orthogonal_directions:
            load_utils.save_orthogonal_directions(
                orthogonal_dirs, intercepts, run_dir, seed, run_id
            )
    elif classification_method == "CCS+LR-in-span":
        if labeled_train_data_dict is None:
            raise ValueError(
                f"labeled_train_data_dict must be provided for {classification_method}."
            )
        if "num_orthogonal_directions" not in train_kwargs:
            raise ValueError(
                f"num_orthogonal_directions required for {classification_method} method."
            )
        num_orthogonal_directions = train_kwargs.pop("num_orthogonal_directions")

        # Use train_prefix for the labeled data and test_prefix for the
        # unlabeled data.
        classify_model, fit_result, orthogonal_dirs, intercepts = train_ccs_lr_in_span(
            data_dict,
            permutation_dict,
            train_data_dict,
            labeled_train_data_dict,
            projection_model,
            train_prefix,
            test_prefix,
            num_orthogonal_directions,
            load_orthogonal_directions_run_dir=load_orthogonal_directions_run_dir,
            projected_sgd=projected_sgd,
            train_kwargs=train_kwargs,
            project_along_mean_diff=project_along_mean_diff,
            device=device,
            logger=logger,
        )

        if save_orthogonal_directions:
            load_utils.save_orthogonal_directions(
                orthogonal_dirs, intercepts, run_dir, seed, run_id
            )
    elif classification_method == "CCS-select-LR":
        if labeled_train_data_dict is None:
            raise ValueError(
                "labeled_train_data_dict must be provided for CCS-select-LR."
            )
        if "num_orthogonal_directions" not in train_kwargs:
            raise ValueError(
                "num_orthogonal_directions required for CCS-select-LR method."
            )
        num_orthogonal_directions = train_kwargs.pop("num_orthogonal_directions")

        # Use train_prefix for the labeled data and test_prefix for the
        # unlabeled data.
        classify_model, fit_result, orthogonal_dirs, intercepts = train_ccs_select_lr(
            data_dict,
            permutation_dict,
            train_data_dict,
            labeled_train_data_dict,
            projection_model,
            train_prefix,
            test_prefix,
            num_orthogonal_directions,
            mode,
            load_orthogonal_directions_run_dir=load_orthogonal_directions_run_dir,
            train_kwargs=train_kwargs,
            project_along_mean_diff=project_along_mean_diff,
            device=device,
            logger=logger,
        )

        if save_orthogonal_directions:
            load_utils.save_orthogonal_directions(
                orthogonal_dirs, intercepts, run_dir, seed, run_id
            )
    elif classification_method == "pseudolabel":
        if labeled_train_data_dict is None:
            raise ValueError(
                "labeled_train_data_dict must be provided for pseudo_label."
            )

        pseudolabel_config = train_kwargs.pop("pseudolabel", None)
        if pseudolabel_config is None:
            raise ValueError("pseudolabel config must be provided.")

        # Use train_prefix for the labeled data and test_prefix for the
        # unlabeled data.
        probes, fit_result = train_pseudo_label(
            data_dict,
            labeled_train_data_dict,
            train_data_dict,
            permutation_dict,
            train_prefix,
            test_prefix,
            pseudolabel_config,
            project_along_mean_diff=project_along_mean_diff,
            projection_model=projection_model,
            train_kwargs=train_kwargs,
            device=device,
            logger=logger,
        )
        # Use the last probe as the classification model.
        classify_model = probes[-1]
    elif classification_method == "BSS":
        if project_along_mean_diff:
            raise ValueError("BSS does not support project_along_mean_diff")

        lis = [
            getPair(
                target_dict={key: [idx]},
                data_dict=train_prefix_data_dict,
                permutation_dict=permutation_dict,
                projection_model=projection_model,
            )
            for key, l in train_data_dict.items()
            for idx in l
        ]

        weights = [1 / len(l) for l in train_data_dict.values() for _ in l]

        classify_model = myClassifyModel(
            method=classification_method, print_more=print_more
        )
        classify_model.fit(
            [w[0] for w in lis],
            [w[1] for w in lis],
            weights=weights,
            **train_kwargs,
        )
    elif classification_method == "LR":
        data, labels = make_contrast_pair_data(
            target_dict=train_data_dict,
            data_dict=train_prefix_data_dict,
            permutation_dict=permutation_dict,
            projection_model=projection_model,
            split="train",
            project_along_mean_diff=project_along_mean_diff,
            split_pair=mode == "concat",
        )

        lr_train_kwargs = train_kwargs["log_reg"]
        classify_model = LogisticRegressionClassifier(n_jobs=1, **lr_train_kwargs)
        if logger is not None:
            logger.info(f"Fitting LR model, mode={mode}")
        classify_model.fit(data, labels, mode)
    else:
        data, labels = make_contrast_pair_data(
            target_dict=train_data_dict,
            data_dict=train_prefix_data_dict,
            permutation_dict=permutation_dict,
            projection_model=projection_model,
            split="train",
            project_along_mean_diff=project_along_mean_diff,
            split_pair=mode == "concat",
        )

        classify_model = myClassifyModel(classification_method, print_more=print_more)
        classify_model.fit(data, labels)

    eval_result = eval(
        test_prefix_data_dict,
        # Arbitrarily use the unsupervised permutation_dict.
        permutation_dict,
        test_dict,
        mode,
        projection_dict=projection_dict,
        projection_method=projection_method,
        n_components=n_components,
        classify_model=classify_model,
        projection_model=projection_model,
        train_prefix=train_prefix,
        classification_method=classification_method,
        print_more=print_more,
        save_probs=save_probs,
        test_on_train=test_on_train,
        project_along_mean_diff=project_along_mean_diff,
        run_dir=run_dir,
        seed=seed,
        run_id=run_id,
        logger=logger,
    )

    return *eval_result, fit_result


def eval(
    data_dict: DataDictType,
    permutation_dict: PermutationDictType,
    test_dict: PromptIndicesDictType,
    mode: Mode,
    projection_dict: Optional[PromptIndicesDictType] = None,
    projection_method="PCA",
    n_components: int = -1,
    classify_model=None,
    projection_model=None,
    train_prefix=None,
    classification_method: EvalClassificationMethodType = "CCS",
    print_more=False,
    save_probs=True,
    test_on_train=False,
    project_along_mean_diff=False,
    run_dir: Optional[str] = None,
    seed: Optional[str] = None,
    run_id: Optional[str] = None,
    logger=None,
):
    if classification_method not in typing.get_args(EvalClassificationMethodType):
        raise ValueError(
            f"Unsupported classification method for eval: {classification_method}"
        )
    if save_probs and (run_dir is None or seed is None or run_id is None):
        raise ValueError(
            "run_dir, seed, and run_id must be provided to save eval results"
        )

    # Train projection model if needed.
    if projection_model is None:
        if n_components != -1:
            if projection_dict is None:
                raise ValueError(
                    "projection_dict must be provided when n_components is not -1."
                )
            # Concatenate all the data (not split) to do PCA.
            # Shape: [num_total_samples, num_features]. If there is a constant number of
            # prompts per dataset, num_total_samples = num_datasets * num_prompts
            # * num_samples_per_prompt.
            proj_states = getConcat(
                [
                    getConcat([data_dict[key][w][0] for w in lis])
                    for key, lis in projection_dict.items()
                ]
            )
            projection_model = myReduction(
                method=projection_method,
                n_components=n_components,
                print_more=print_more,
            )
            projection_model.fit(proj_states)
        else:
            projection_model = IdentityReduction()

    # Load classification model params.
    if classify_model is None:
        # TODO
        if classification_method == "CCS+LR":
            raise NotImplementedError()
        coef, bias = load_utils.load_params(
            run_dir, classification_method, train_prefix
        )
        if classification_method in ["CCS", "Random"]:
            classify_model = ConsistencyMethod.from_coef_and_bias(
                coef, bias, verbose=print_more
            )
        elif classification_method == "LR":
            classify_model = LogisticRegressionClassifier.from_coef_and_bias(
                coef, bias=bias
            )
        else:
            classify_model = myClassifyModel.from_coef_and_bias(
                classification_method, coef, bias=bias, print_more=print_more
            )

    # Evaluate the model on the test sets.
    acc_dict, loss_dict, ece_dict = {}, {}, {}
    for dataset, prompt_indices in test_dict.items():
        # Create eval dir if needed.
        if save_probs:
            eval_dir = load_utils.get_eval_dir(run_dir, dataset)
            if not os.path.exists(eval_dir):
                os.makedirs(eval_dir)

        acc_dict[dataset], loss_dict[dataset], ece_dict[dataset] = [], [], []
        for prompt_idx in prompt_indices:
            # Get the data and labels for the current dataset and prompt.
            dataset_dict = {dataset: [prompt_idx]}
            # Split the hidden states into those for class "0" and class "1"
            # if the hidden states for x+ and x- are concatenated.
            split_pair = mode == "concat"
            data, label = make_contrast_pair_data(
                target_dict=dataset_dict,
                data_dict=data_dict,
                permutation_dict=permutation_dict,
                projection_model=projection_model,
                split=("train" if test_on_train else "test"),
                project_along_mean_diff=project_along_mean_diff,
                split_pair=split_pair,
            )

            if classification_method in [
                "CCS+LR",
                "CCS-in-LR-span",
                "CCS+LR-in-span",
                "CCS-select-LR",
                "pseudolabel",
            ]:
                device = classify_model.device
                x0, x1 = data
                x0 = torch.tensor(x0, device=device)
                x1 = torch.tensor(x1, device=device)
                y = torch.tensor(label, device=device).float().view(-1, 1)
                acc, p0, p1, probs = classify_model.evaluate_accuracy(x0, x1, y)
                p0 = p0.float().cpu().numpy().flatten()
                p1 = p1.float().cpu().numpy().flatten()
                probs = probs.float().cpu().numpy().flatten()
                with torch.no_grad():
                    classify_model.eval()
                    losses = classify_model.compute_loss(
                        x0,
                        x1,
                        y,
                        x0,
                        x1,
                    )
                    losses = {k: loss.item() for k, loss in losses.items()}
            elif classification_method == "LR":
                acc, probs, p0, p1 = classify_model.score(data, label, mode)
                losses = None
            else:
                acc, losses, probs, p0, p1 = classify_model.score(
                    data, label, get_loss=True, get_probs=True
                )
            if probs is None and not (p0 is None and p1 is None):
                raise ValueError("probs is None but p0 and p1 are not both None.")

            # Arbitrarily set losses to 0 if the method does not provide them.
            losses = losses or (0.0, 0.0, 0.0)

            if probs is not None:
                probs = probs.flatten()
                ece = metrics.expected_calibration_error(probs, label)[0]
                ece_flip = metrics.expected_calibration_error(1 - probs, label)[0]

                if save_probs:
                    save_probs_file = load_utils.get_probs_save_path(
                        eval_dir,
                        classification_method,
                        project_along_mean_diff,
                        prompt_idx,
                    )
                    probs_dict = {"prob": probs}
                    if p0 is not None:
                        probs_dict["p0"] = p0.flatten()
                    if p1 is not None:
                        probs_dict["p1"] = p1.flatten()
                    probs_dict["label"] = label
                    df = pd.DataFrame(probs_dict)
                    df.to_csv(save_probs_file, index=False)
            else:
                if logger is not None:
                    logger.warning(
                        "p0 and p1 are None for classification method "
                        f"{classification_method}, not saving probabilities."
                    )
                ece = None
                ece_flip = None

            acc_dict[dataset].append(acc)
            loss_dict[dataset].append(losses)
            ece_dict[dataset].append((ece, ece_flip))

    return acc_dict, loss_dict, ece_dict, projection_model, classify_model


# print("\
# ------ Func: printAcc ------\n\
# ## Input = (input_dict, verbose) ##\n\
#     input_dict: The dict generated by `mainResults`.\n\
#     verbose: Whether to print dataset level accuracy.\n\
# ## Output ##\n\
#     Directly print the accuracy and return the global level accuracy.\n\
# ")
def printAcc(input_dic, verbose=1):
    if type(input_dic) != dict:
        print(input_dic)
        return np.mean(input_dic)
    if verbose >= 2:
        for key in input_dic.keys():
            print(
                "Test on {}, avg acc is {:.2f}, best is {:.2f}, std is {:.2f}".format(
                    key,
                    100 * np.mean(input_dic[key]),
                    100 * np.max(input_dic[key]),
                    100 * np.std(input_dic[key]),
                )
            )
    global_acc = np.mean([100 * np.mean(w) for w in input_dic.values()])
    global_std = np.mean([100 * np.std(w) for w in input_dic.values()])
    if verbose >= 1:
        print("## Global accuracy: {:.2f}, std.: {:.2f}".format(global_acc, global_std))
    return global_acc
