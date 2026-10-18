"""Product kernel: topic-dependent contribution of the text's n-gram (variant C5).

MECHANISM
---------
The linear kernel on n-grams assumes that the contribution of a given n-gram to
the teacher vector is CONSTANT, the same in every context. The T6 measurement
showed that this assumption is false: the teacher is non-additive (the vector
of a concatenation of two texts is not the sum of their vectors, residual
0.339), i.e. it is strongly contextualizing. We model this with a product
kernel:

    K2(t, t') = <phi(t), phi(t')> * kappa(z(t), z(t')),
    kappa(z, z') = ((1 + z^T z') / 2)^q,        q >= 1,

where z(t) is the normalized stage-I prediction (plain ridge on the linear
kernel), i.e. the approximate "topic" of the text in the OUTPUT space.
Interpretation: two texts borrow information from each other proportionally to
the product of (a) n-gram overlap and (b) topical closeness. For topically
distant pairs kappa suppresses the kernel (at q = 2 median kappa = 0.39),
so the regression does not average n-gram coefficients over mismatched
contexts.

Correctness: the Hadamard product of two PSD matrices is PSD (Schur's
theorem), and kappa > 0, so K2 is a valid kernel and the dual ridge works
without modification. The diagonal remains 1 (kappa(z,z) = 1), so the lambda
scale is comparable.

Leakage: the z coordinates for TRAINING ROWS must be an out-of-sample
prediction. We use the closed-form LOO (`dual.solve_ridge_loo_f32`, N <= 12000)
or the explicit inverse. The in-sample prediction at lambda = 1e-4 is nearly
equal to y_i, so kappa would be built from the target and the result would be
inflated.

MEASURED EFFECT (validation 1200, N = 9600, features signs 3-4, results/jadra_C3_C5.csv)
------------------------------------------------------------------------------------------
                                        rho glob  rho/zap  cos centr    R^2   recall@10
  baseline: ridge + residual correction (k=50)  0.7714    0.7613     0.7102   0.4293   0.4015
  K2, q=2, lam=1e-5, + correction (k=50)         0.7691    0.7624     0.7199   0.4477   0.4025
  control: without kappa at lam=1e-5             0.7578    0.7434     0.6894   0.3846   0.3952

Reading these numbers: at the MATCHED lambda kappa gives a clear gain
(+0.0113 rho, +0.063 R^2 versus the control). Against the BEST baseline
configuration the gain is already only marginal: R^2 +0.018 and centered cos
+0.010, whereas global rho is 0.0023 WORSE, i.e. indistinguishable (baseline
CI95 0.7584-0.7854). Practical meaning: the product kernel is worth enabling
when the goal is pointwise vector fidelity (R^2, cos), not pair ordering;
q = 4 is already worse on all metrics, q = 8 is markedly worse (rho 0.7515).

Structure of the problem: kappa and residual correction in the output space
implement the SAME idea (locality in the target space) and largely substitute
for each other, the gain from residual correction drops from +0.0198 (linear
kernel, lam=1e-4) to +0.0018 (K2 q=2, lam=1e-5), because stage I no longer
leaves a residual to correct."""

from __future__ import annotations

import numpy as np

from .dual import l2norm

DOMYSLNE = {"q": 2, "lam1": 1e-4, "lam2": 1e-5}


def _kappa(S: np.ndarray, q: int) -> np.ndarray:
    """((1 + s)/2)^q for the matrix of dot products of normalised z."""
    return np.power(0.5 * (1.0 + S), q)


def wspolrzedne_loo(K: np.ndarray, Ytr: np.ndarray, lam: float) -> tuple:
    """(alpha, z) of degree I: ridge + LOO prediction in closed form.

    z is normalized to norm 1, because only the direction enters kappa.
    """
    n = K.shape[0]
    Gi = np.linalg.inv(K + (lam * n) * np.eye(n))
    alpha = Gi @ Ytr
    z = l2norm(Ytr - alpha / np.diag(Gi)[:, None])
    return alpha, z


def fit_iloczyn(K: np.ndarray, Ytr: np.ndarray, q: int = 2, lam1: float = 1e-4,
                lam2: float = 1e-5, alpha1: np.ndarray | None = None,
                z: np.ndarray | None = None) -> dict:
    """Two-stage fit on a precomputed linear kernel matrix K (N x N).

    alpha1/z can be provided externally (e.g. computed once via
    `dual.solve_ridge_loo_f32` for large N), then lam1 is only a record.
    """
    n = K.shape[0]
    if alpha1 is None or z is None:
        alpha1, z = wspolrzedne_loo(K, Ytr, lam1)
    K2 = K * _kappa(z @ z.T, q)
    np.fill_diagonal(K2, 1.0)
    alpha2 = np.linalg.solve(K2 + (lam2 * n) * np.eye(n), Ytr)
    return {"metoda": "jadro_iloczynowe", "q": q, "lam1": lam1, "lam2": lam2,
            "alpha1": alpha1, "alpha2": alpha2, "z": z, "Y": Ytr}


def predict_iloczyn(st: dict, Ks: np.ndarray) -> np.ndarray:
    """Prediction for queries: Ks is the (M x N) query-training similarity.

    Two passes through the kernel: first, stage I determines the topic of the query,
    then stage II computes the vector on the kernel damped by kappa.
    """
    z_q = l2norm(Ks @ st["alpha1"])
    K2s = Ks * _kappa(z_q @ st["z"].T, st["q"])
    return l2norm(K2s @ st["alpha2"])
