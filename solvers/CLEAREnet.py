import warnings

from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from clearenet.enet.dropin_sklearn import LASSO
    from clearenet.load_data.upload_data import upload_data
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = "CLEAREnet"
    stop_strategy = "iteration"
    install_cmd = "conda"
    requirements = ["clearenet", "numba"]
    references = ["F. Bascou, S. Lebre, and J. Salmon, "]

    def set_objective(self, X, y, lmbd, fit_intercept):
        print("X shape", X.shape)
        X, Y = upload_data()
        y = Y[:, 0]
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        n_samples = self.X.shape[0]
        self.clearlasso = LASSO(
            debiasing=False,
            alpha=self.lmbd / n_samples,
            max_epochs=10**1,
            max_iter=1,
            tol=1e-12,
            rescaling="None",
            fit_intercept=fit_intercept,
            fit_interaction=True,
        )

    def run(self, n_iter):
        # print("CLEARENET - n_iter : ", n_iter)
        self.clearlasso.max_iter = n_iter
        self.clearlasso.fit(
            self.X, self.y, fit_interaction=True, debiasing=False
        )

    # @staticmethod
    # def get_next(previous):
    #     "Linear growth for n_iter."
    #     return previous + 1

    def get_result(self):

        beta = self.clearlasso.coef_.flatten()
        if self.fit_intercept:
            beta = np.r_[beta, self.clearlasso.intercept_]
        return beta
