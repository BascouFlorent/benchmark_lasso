import numpy as np
from numpy.linalg import norm

from benchopt import BaseObjective


class Objective(BaseObjective):
    name = "Lasso Regression"

    parameters = {
        "fit_intercept": [True, False],
        # "fit_intercept": [False],
        # "reg": [0.5, 0.1, 0.05],
        # "reg": [0.1, 0.05, 0.01, 0.005, 0.001],
        "reg": [0.5, 0.1],  # , 0.05, 0.01],
    }

    def __init__(self, reg=0.1, fit_intercept=False):
        self.reg = reg
        self.fit_intercept = fit_intercept

    def set_data(self, X, y):
        self.X, self.y = X, y
        self.lmbd = self.reg * self._get_lambda_max()
        self.n_features = self.X.shape[1]

    def compute(self, beta):
        # compute residuals
        if self.fit_intercept:
            beta, intercept = beta[: self.n_features], beta[self.n_features :]
        diff = self.y - self.X.dot(beta)
        if self.fit_intercept:
            diff -= intercept
        # compute primal objective and duality gap
        p_obj = 0.5 * diff.dot(diff) + self.lmbd * abs(beta).sum()
        theta = diff / self.lmbd
        theta /= norm(self.X.T @ theta, ord=np.inf)
        d_obj = (
            norm(self.y) ** 2 / 2.0
            - self.lmbd**2 * norm(self.y / self.lmbd - theta) ** 2 / 2
        )
        return dict(
            value=p_obj,
            support_size=(beta != 0).sum(),
            duality_gap=p_obj - d_obj,
        )

    def _get_lambda_max(self):
        return abs(self.X.T.dot(self.y)).max()

    def to_dict(self):
        return dict(
            X=self.X,
            y=self.y,
            lmbd=self.lmbd,
            fit_intercept=self.fit_intercept,
        )
