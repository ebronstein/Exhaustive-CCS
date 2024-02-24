import os
import time
import typing
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from utils.types import DataDictType, PermutationDictType, PromptIndicesDictType
from utils_extraction import load_utils, metrics

UNSUPERVISED_METHODS = ("TPC", "KMeans", "BSS", "CCS", "Random")
SUPERVISED_METHODS = ("LR",)

EvalClassificationMethodType = Literal["LR", "BSS", "CCS"]


def is_method_unsupervised(method):
    return method in UNSUPERVISED_METHODS or method.startswith("RCCS")


class IdentityReduction:

    def fit(self, data):
        pass

    def transform(self, data):
        return data


class myReduction:
    def __init__(
        self, method, n_components, print_more=False, svd_solver="full"
    ) -> None:
        self.n_components = n_components
        self.method = method
        assert method in ["PCA", "UMAP"], NotImplementedError(
            "Only support PCA and UMAP to project data."
        )
        self.print_more = print_more
        self.num_feature = None
        if n_components != -1:
            if self.method == "PCA":
                self.model = PCA(
                    n_components=n_components, svd_solver=svd_solver
                )
            elif self.method == "UMAP":
                self.model = umap.UMAP(n_components=n_components)

    def fit(self, data):
        self.num_feature = data.shape[1]
        if self.n_components == -1:
            if self.print_more:
                print("n_components = -1, will return identity")

        else:
            if (
                self.method == "UMAP"
            ):  # for UMAP, explicitly centralize the data
                data = data - np.mean(data, axis=0)
            self.model.fit(data)
            if self.method == "PCA":  # for PCA, explicitly set mean to None
                self.model.mean_ = None
                if self.print_more:
                    print("Set the mean of PCA model to `None`.")
            if self.print_more:
                if self.method == "PCA":
                    print(
                        "PCA fit data. dim = {} and #data = {}, var is {}".format(
                            self.n_components,
                            data.shape,
                            sum(self.model.explained_variance_ratio_),
                        )
                    )
                else:
                    print(
                        "UMAP fit data. dim = {} and #data = {}.".format(
                            self.n_components, data.shape
                        )
                    )

    def getDirection(self):
        # return the component with shape (n_components, n_features)
        if self.n_components == -1:
            return np.eye(self.num_feature)
        else:
            return self.model.components_

    def transform(self, data):
        if self.n_components == -1:
            return data
        return self.model.transform(data)

    def __getattr__(self, __name):
        if __name == "n_components":
            return self.n_components
        return getattr(self.model, __name)


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


def normalize(directions):
    return directions / np.linalg.norm(directions, axis=-1, keepdims=True)


def assert_close_to_orthonormal(directions, atol=1e-3):
    assert np.allclose(
        directions @ directions.T, np.eye(directions.shape[0]), atol=atol
    ), "Not orthonormal"


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
    def transform(self, data: list, theta_np=None):
        if theta_np is None:
            theta_np = self.best_theta
        z0, z1 = torch.tensor(
            self.add_ones_dimension(data[0]).dot(theta_np.T)
        ), torch.tensor(self.add_ones_dimension(data[1]).dot(theta_np.T))
        p0, p1 = torch.sigmoid(z0).numpy(), torch.sigmoid(z1).numpy()

        return p0, p1

    def predict_probs(self, data):
        """Predicts class 1 probabilities for data."""
        p0, p1 = self.transform(data, self.best_theta)
        return 0.5 * (p1 + (1 - p0))

    # Return the accuracy of (data, label)
    def get_acc(
        self, theta_np, data: list, label, getloss=False, save_file=None
    ):
        """
        Computes the accuracy of a given direction theta_np represented as a numpy array
        """
        p0, p1 = self.transform(data, theta_np)
        avg_confidence = 0.5 * (p1 + (1 - p0))

        label = label.reshape(-1)
        predictions = (avg_confidence >= 0.5).astype(int)[:, 0]
        acc = (predictions == label).mean()

        # TODO: move this out of this function
        if save_file is not None:
            # save (p0, p1, label) to file using pandas
            df = pd.DataFrame({"p0": p0[:, 0], "p1": p1[:, 0], "label": label})
            df.to_csv(save_file, index=False)

        if getloss:
            losses = [
                l.cpu().detach().item()
                for l in self.get_losses(torch.tensor(p0), torch.tensor(p1))
            ]
            return acc, losses
        return acc

    def train(self):
        """
        Does a single training run of nepochs epochs
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
        for _ in range(self.nepochs):

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
        data: list,
        label,
        nepochs=1000,
        ntries=10,
        lr=1e-2,
        init_theta=None,
        device="cuda",
    ):
        """
        Does ntries attempts at training, with different random initializations
        """

        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr

        self.device = device

        self.init_theta = init_theta
        if self.init_theta is not None:
            self.ntries = 1

        if self.verbose:
            print(
                "String fiting data with Prob. nepochs: {}, ntries: {}, lr: {}".format(
                    nepochs, ntries, lr
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

        for _ in range(self.ntries):
            # train
            theta_np, loss = self.train()

            # evaluate
            acc = self.get_acc(theta_np, data, label, getloss=False)

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

    def score(self, data: list, label, getloss=False, save_file=None):
        self.validate_data(data)
        return self.get_acc(
            self.best_theta, data, label, getloss, save_file=save_file
        )


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
        epochs=20,
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
            self.set_params(
                np.array(self.sign).reshape(1, 1), -self.sign * self.avg
            )

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
                u / w.shape[0]
                for u, w in zip(weights, data)
                for _ in range(w.shape[0])
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

                for epoch in range(epochs):

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
                        epochs - 1,
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
                        self.set_params(
                            self.sign * theta_np, -self.sign * self.avg
                        )
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
        self, data, label, getloss=False, sample_weight=None, save_file=None
    ):
        # TODO: move file saving out of this function.
        if self.method == "KMeans":
            if save_file is not None:
                print("save_file not supported for KMeans")

            prediction = self.model.predict(data)
            acc = np.mean(prediction == label)
            if getloss:
                return acc, (0.0, 0.0, 0.0)
            return acc
        else:
            if save_file is not None:
                # compute for save_file
                predictions = super().predict_proba(data)
                df = pd.DataFrame(
                    {"label": label, "prediction": predictions[:, 1]}
                )
                df.to_csv(save_file, index=False)

            if sample_weight is not None:
                acc = super().score(data, label, sample_weight)
            else:
                acc = super().score(data, label)
            if getloss:
                if self.method == "BSS":
                    loss = getSingleLoss(data @ self.coef_.T + self.intercept_)
                else:
                    loss = (0.0, 0.0, 0.0)
                return acc, loss
            return acc


def getConcat(data_list, axis=0):
    sub_list = [w for w in data_list if w is not None]
    if not sub_list:
        return None
    return np.concatenate(sub_list, axis=axis)


def getPair(
    target_dict: PromptIndicesDictType,
    data_dict: DataDictType,
    permutation_dict: PermutationDictType,
    projection_model: myReduction,
    split: Literal["train", "test"] = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate the data and labels for the desired split.

    `projection_model` is used to transform the hidden states. This may be a
    noop (e.g., if `projection_model.n_components` is -1).

    Args:
        target_dict (dict): Dictionary of prompt indices to use for each
            dataset. Key is dataset name, each value is a list of prompt
            indices for which the corresponding data and labels are returned.
        data_dict (dict): Dictionary of hidden states. Key is dataset name, each
            value is a list with one element per prompt. Each element is a tuple
            pair of (hidden_states, labels).
        permutation_dict (dict): Dictionary of permutations. Key is dataset
            name, value is a tuple pair containing the train split and test
            split indices.
        projection_model (myReduction): Projection model used to transform the
            hidden states.
        split (str, optional): The desired split. Defaults to "train".
    """
    if split == "train":
        split_idx = 0
    elif split == "test":
        split_idx = 1
    else:
        raise ValueError(
            f"split should be either 'train' or 'test', got '{split}'"
        )

    lis = []
    for key, prompt_lis in target_dict.items():
        # Indices of the desired split for the current dataset.
        split_indices = permutation_dict[key][split_idx]
        for idx in prompt_lis:
            hidden_states = data_dict[key][idx][0][split_indices]
            labels = data_dict[key][idx][1][split_indices]
            lis.append(
                [
                    projection_model.transform(hidden_states),
                    labels,
                ]
            )

    data, label = getConcat([w[0] for w in lis]), getConcat([w[1] for w in lis])

    return data, label


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


def mainResults(
    data_dict: DataDictType,
    permutation_dict: PermutationDictType,
    test_dict: PromptIndicesDictType,
    projection_dict: PromptIndicesDictType,
    projection_method="PCA",
    n_components: int = 2,
    projection_only=False,
    prefix=None,
    classification_method="BSS",
    print_more=False,
    learn_dict={},
    save_probs=True,
    test_on_train=False,
    constraints=None,
    project_along_mean_diff=False,
    run_dir: Optional[str] = None,
    seed: Optional[str] = None,
    run_id: Optional[str] = None,
):
    """
    Calculate the main results.

    Args:
        data_dict (dict): Dictionary of hidden states and labels. Key is dataset
            name, each value is a list with one element per prompt. Each element
            is a tuple pair of (hidden_states, labels).
        permutation_dict (dict): Dictionary of train/test split indices. Key is
            dataset name, value is a tuple pair containing the train split and
            test split indices.
        projection_dict (dict): Dictionary of prompt indices for projection.
            Key is dataset name, each value is a list of prompt indices used to
            do projection.
        test_dict (dict): Dictionary of test datasets and prompt indices on
            which the method is evaluated. Key is dataset name, each value is a
            list of prompt indices used to do evaluation.
        projection_method (str, optional): Projection method. Defaults to "PCA".
        n_components (int, optional): The dimension you want to reduce to. -1 means no
            projection will be implemented. Defaults to 2.
        projection_only (bool, optional): When set to true, will immediately return after
            training the projection_model. res and classify_model will be None. Defaults
            to False.
        classification_method (str, optional): Classification method. Can be LR, TPC, and
            BSS. Defaults to "BSS".
        print_more (bool, optional): Whether to print more information. Defaults to False.
        learn_dict (dict, optional): Dictionary for learning. Defaults to {}.
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
    """
    start = time.time()
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
    if classification_method != "CCS" and constraints is not None:
        raise ValueError("constraints only supported for CCS")

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

    if projection_only:
        return None, projection_model, None

    # pairFunc = partial(getPair, data_dict = data_dict, permutation_dict = permutation_dict, projection_model = projection_model)

    if classification_method == "CCS":
        classify_model = ConsistencyMethod(
            no_train=no_train, verbose=print_more, constraints=constraints
        )
        datas, label = getPair(
            target_dict=projection_dict,
            data_dict=data_dict,
            permutation_dict=permutation_dict,
            projection_model=projection_model,
            split="train",
        )
        assert len(datas.shape) == 2
        if project_along_mean_diff:
            datas = project_data_along_axis(datas, label)
        data = [
            datas[:, : datas.shape[1] // 2],
            datas[:, datas.shape[1] // 2 :],
        ]

        classify_model.fit(data=data, label=label, **learn_dict)
    elif classification_method == "BSS":
        if project_along_mean_diff:
            raise ValueError("BSS does not support project_along_mean_diff")

        lis = [
            getPair(
                target_dict={key: [idx]},
                data_dict=data_dict,
                permutation_dict=permutation_dict,
                projection_model=projection_model,
            )
            for key, l in projection_dict.items()
            for idx in l
        ]

        weights = [1 / len(l) for l in projection_dict.values() for _ in l]

        classify_model = myClassifyModel(
            method=classification_method, print_more=print_more
        )
        classify_model.fit(
            [w[0] for w in lis],
            [w[1] for w in lis],
            weights=weights,
            **learn_dict,
        )

    else:
        classify_model = myClassifyModel(
            classification_method, print_more=print_more
        )
        data, labels = getPair(
            data_dict=data_dict,
            permutation_dict=permutation_dict,
            projection_model=projection_model,
            target_dict=projection_dict,
        )

        if project_along_mean_diff:
            data = project_data_along_axis(data, labels)

        classify_model.fit(data, labels)

    return eval(
        data_dict,
        permutation_dict,
        test_dict,
        projection_dict,
        projection_method=projection_method,
        n_components=n_components,
        classify_model=classify_model,
        projection_model=projection_model,
        prefix=prefix,
        classification_method=classification_method,
        print_more=print_more,
        save_probs=save_probs,
        test_on_train=test_on_train,
        project_along_mean_diff=project_along_mean_diff,
        run_dir=run_dir,
        seed=seed,
        run_id=run_id,
    )


def eval(
    data_dict: DataDictType,
    permutation_dict: PermutationDictType,
    test_dict: PromptIndicesDictType,
    projection_dict: Optional[PromptIndicesDictType] = None,
    projection_method="PCA",
    n_components: int = -1,
    classify_model=None,
    projection_model=None,
    prefix=None,
    classification_method: EvalClassificationMethodType = "CCS",
    print_more=False,
    save_probs=True,
    test_on_train=False,
    project_along_mean_diff=False,
    run_dir: Optional[str] = None,
    seed: Optional[str] = None,
    run_id: Optional[str] = None,
):
    if classification_method not in typing.get_args(
        EvalClassificationMethodType
    ):
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
        coef, bias = load_utils.load_params(
            run_dir, classification_method, prefix
        )
        if classification_method in ["CCS", "Random"]:
            classify_model = ConsistencyMethod.from_coef_and_bias(
                coef, bias, verbose=print_more
            )
        else:
            classify_model = myClassifyModel.from_coef_and_bias(
                classification_method, coef, bias=bias, print_more=print_more
            )

    # Evaluate the model on the test sets.
    res, lss, ece_dict = {}, {}, {}
    for dataset, prompt_indices in test_dict.items():
        # Create eval dir if needed.
        if save_probs:
            eval_dir = load_utils.get_eval_dir(run_dir, dataset, seed, run_id)
            if not os.path.exists(eval_dir):
                os.makedirs(eval_dir)

        res[dataset], lss[dataset], ece_dict[dataset] = [], [], []
        for prompt_idx in prompt_indices:
            # Get the data and labels for the current dataset and prompt.
            dataset_dict = {dataset: [prompt_idx]}
            data, label = getPair(
                target_dict=dataset_dict,
                data_dict=data_dict,
                permutation_dict=permutation_dict,
                projection_model=projection_model,
                split=("train" if test_on_train else "test"),
            )

            if project_along_mean_diff:
                data = project_data_along_axis(data, label)

            # Split the hidden states into those for class "0" and class "1".
            if classification_method == "CCS":
                if data.shape[1] % 2 != 0:
                    raise ValueError(
                        "CCS requires the same number of hidden states for "
                        "class '0' and class '1' to be concatenated, got "
                        f"{data.shape[1]} total features, which is odd."
                    )
                data = [
                    data[:, : data.shape[1] // 2],
                    data[:, data.shape[1] // 2 :],
                ]

            # TODO: save probs directly to the given path instead of to a CSV
            # in the directory.
            if save_probs:
                save_probs_file = load_utils.get_probs_save_path(
                    eval_dir,
                    classification_method,
                    project_along_mean_diff,
                    prompt_idx,
                )
            else:
                save_probs_file = None

            acc, losses = classify_model.score(
                data, label, getloss=True, save_file=save_probs_file
            )
            predicted_probs = classify_model.predict_probs(data).flatten()
            ece = metrics.expected_calibration_error(predicted_probs, label)[0]
            ece_flip = metrics.expected_calibration_error(
                1 - predicted_probs, label
            )[0]
            res[dataset].append(acc)
            lss[dataset].append(losses)
            ece_dict[dataset].append((ece, ece_flip))

    return res, lss, ece_dict, projection_model, classify_model


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
        print(
            "## Global accuracy: {:.2f}, std.: {:.2f}".format(
                global_acc, global_std
            )
        )
    return global_acc
