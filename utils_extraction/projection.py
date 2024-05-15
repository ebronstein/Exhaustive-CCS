import os
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

from utils.types import ProjectionMethod


class Reduction(ABC):

    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass


class myReduction(Reduction):
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
                self.model = PCA(n_components=n_components, svd_solver=svd_solver)
            elif self.method == "UMAP":
                self.model = umap.UMAP(n_components=n_components)

    def fit(self, data):
        self.num_feature = data.shape[1]
        if self.n_components == -1:
            if self.print_more:
                print("n_components = -1, will return identity")

        else:
            if self.method == "UMAP":  # for UMAP, explicitly centralize the data
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

    # TODO
    def save(self, path: str):
        raise NotImplementedError()

    # TODO
    def load(self, path: str):
        raise NotImplementedError()


class IdentityReduction(Reduction):

    def fit(self, X, y=None, **fit_params):
        pass

    def transform(self, X):
        return X

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass


class GaussianRandomMatrixProjection(GaussianRandomProjection):
    def save(self, path: str):
        with open(path, "wb") as f:
            np.save(f, self.components_)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.components_ = np.load(f)
            self.n_components_ = self.components_.shape[0]


def make_projection(method: Optional[ProjectionMethod], **kwargs) -> Reduction:
    if method is None:
        return IdentityReduction()
    if method == "PCA":
        return myReduction(method="PCA", **kwargs)
    if method == "UMAP":
        return myReduction(method="UMAP", **kwargs)
    if method == "gaussian_random":
        return GaussianRandomMatrixProjection(**kwargs)
    raise NotImplementedError(f"Projection method {method} is not supported.")


def get_projection_path(method: Optional[ProjectionMethod], dir_path: str) -> str:
    if method is None:
        return None
    return os.path.join(dir_path, f"projection_{method}.npy")


def maybe_save_projection(
    proj_model: Reduction, method: Optional[ProjectionMethod], dir_path: str
):
    if method is not None:
        projection_path = get_projection_path(method, dir_path)
        proj_model.save(projection_path)
