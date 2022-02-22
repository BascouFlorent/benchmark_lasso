from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from clearenet.load_data.upload_data import upload_data
    from clearenet.enet.utils_solvers import make_interaction, product_int
    import numpy as np


class Dataset(BaseDataset):

    name = "omics"
    parameters = {"interaction": ["True", "False"]}

    install_cmd = "conda"
    requirements = ["clearenet", "numba"]

    def get_data(self, interaction=False):
        X, Y = upload_data()
        y = Y[:, 0]
        if interaction:
            Z = make_interaction(X, fct_int=product_int)
            X = np.hstack([X, Z])
        else:
            X = X.copy()
        data = dict(X=X, y=y)

        return X.shape[1], data
