from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from utils.types import Mode


class LogisticRegressionClassifier(LogisticRegression):

    @classmethod
    def from_coef_and_bias(cls, coef, bias=None, **kwargs):
        instance = cls(**kwargs)
        instance.set_params(coef, bias)
        return instance

    def set_params(self, coef, bias):
        self.classes_ = np.array([0, 1])
        self.intercept_ = bias
        self.coef_ = coef

    def fit(self, data, label, mode: Mode):
        if mode == "concat":
            assert len(data) == 2
            x0, x1 = data
            data = np.concatenate([x0, x1], axis=0)
            label = np.concatenate([1 - label, label], axis=0)

        super().fit(data, label)

    def score(
        self,
        data,
        label,
        mode: Mode,
    ) -> tuple[
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        if mode == "concat":
            assert len(data) == 2
            x0, x1 = data
            x0_probs = super().predict_proba(x0)
            p0 = x0_probs[:, 1]
            x1_probs = super().predict_proba(x1)
            p1 = x1_probs[:, 1]
            probs = ((1 - p0) + p1) / 2
        else:
            probs = super().predict_proba(data)
            p0 = probs[:, 0]
            p1 = probs[:, 1]
            # Return the probability of class 1
            probs = p1

        predictions = probs >= 0.5
        acc = (predictions == label).mean()

        return acc, probs, p0, p1
