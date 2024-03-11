import numpy as np
import umap
from sklearn.decomposition import PCA


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


class IdentityReduction:

    def fit(self, data):
        pass

    def transform(self, data):
        return data
