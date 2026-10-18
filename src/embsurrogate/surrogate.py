"""Surrogate embedding model, text -> vector mapping computed
exclusively on (text, vector) pairs from the teacher model.

All estimators are of dual form (via the kernel), i.e. the prediction is
a combination of TRAINING VECTORS, not a product with a weight matrix of size
D x 1024. There are two reasons: (1) D = 262144, so the primal matrix would have
~1 GB, while the dual coefficients are N x 1024; (2) in dual form, ridge
regression, Nadaraya-Watson and LOESS are the same family, they differ only
in the shape of the weights; so they can be fairly compared.

Methods:
    knn1       nearest neighbour (lower reference, "copy the neighbour")
    nw         Nadaraya-Watson: tricube kernel weighted average over k neighbours
    loess      local linear regression (LOESS degree 1) in Nyström
               coordinates, tricube weights, variant proposed by the client
    ridge      kernel ridge regression (global, closed-form solution)
    rbf        as above, but Gaussian kernel on angular distance
    hybrid     ridge as trend + LOESS on residuals (error smoothing)

All arithmetic in float64. The result is normalized to norm 1, because
SBERT/bge sentence models return normalized vectors and only direction matters
for the cosine."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from . import features as F
from .dual import (  # noqa: F401  (re-eksport: jeden zestaw wzorów dla całego pakietu)
    METHODS,
    csr_matvec_numpy,
    fit_dual,
    l2norm,
    predict_dual,
    tricube,
    _kernel_coords,
    _local,
    _rbf,
    _solve_ridge,
)


# ---------------------------------------------------------------- estimator


class Surrogate:
    """Frozen surrogate. `fit` computes only what cannot be reconstructed
    from the pairs alone; `predict` is deterministic with respect to the order of operations."""

    def __init__(
        self,
        method: str = "hybrid",
        *,
        cfg: dict | None = None,
        k: int = 200,
        lam: float = 1e-3,
        gamma: float = 1.0,
        rank: int = 32,
        ridge_local: float = 1e-6,
        degree: int = 1,
    ):
        if method not in METHODS:
            raise ValueError(f"unknown method: {method} (available: {METHODS})")
        self.method = method
        self.cfg = dict(cfg or F.DEFAULT_CONFIG)
        self.k = int(k)                 # number of neighbours in local methods
        self.lam = float(lam)           # kernel ridge regression regularization
        self.gamma = float(gamma)       # RBF kernel width
        self.rank = int(rank)           # dimension of local coordinates (Nyström)
        self.ridge_local = float(ridge_local)
        self.degree = int(degree)       # 0 = Nadaraya-Watson, 1 = linear LOESS

        self.idf: np.ndarray | None = None
        self.X: sparse.csr_matrix | None = None
        self.Y: np.ndarray | None = None
        self.alpha: np.ndarray | None = None
        self.U: np.ndarray | None = None      # kernel eigenvectors (N x rank)
        self.lam_eig: np.ndarray | None = None
        self.texts: list[str] = []
        self.ids: list[str] = []

    # ---------------------------------------------------------------- fit

    def fit(self, texts, Y: np.ndarray, ids=None) -> "Surrogate":
        texts = list(texts)
        self.texts = texts
        self.ids = list(ids) if ids is not None else [str(i) for i in range(len(texts))]
        self.Y = np.ascontiguousarray(Y, dtype=np.float64)
        self.X, self.idf = F.fit_transform(texts, self.cfg)

        K = np.asarray((self.X @ self.X.T).todense(), dtype=np.float64)
        np.fill_diagonal(K, 1.0)          # norms are 1 to within 1e-16

        self.state = fit_dual(
            self.method, K, self.Y,
            lam=self.lam, gamma=self.gamma, rank=self.rank, k=self.k,
            degree=self.degree, ridge_local=self.ridge_local,
        )
        self.alpha = self.state.get("alpha")
        self.U, self.lam_eig = self.state.get("U"), self.state.get("lam_eig")
        return self

    # ------------------------------------------------------------ predict

    def kstar(self, texts) -> np.ndarray:
        """Query x train similarity matrix (cosine)."""
        Xq = F.transform(texts, self.idf, self.cfg)
        return np.asarray((Xq @ self.X.T).todense(), dtype=np.float64)

    def predict(self, texts, ks: np.ndarray | None = None) -> np.ndarray:
        ks = self.kstar(texts) if ks is None else ks
        return predict_dual(self.state, ks)
