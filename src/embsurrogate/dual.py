"""Pure surrogate mathematics, numpy only.

This module is deliberately cut off from scipy, sentence-transformers and anything
that might disappear or change behaviour. Inference from the frozen model
(`klon/predict.py`) relies only on this, so that reproducing vectors after years
depends on a single stable library."""

from __future__ import annotations

import numpy as np

METHODS = ("knn1", "nw", "loess", "ridge", "rbf", "hybrid", "hybrid_z")

# Note on `hybrid_z`: this is the variant that won after diagnostics from round 2.
# Residual correction looks for neighbours NOT in n-gram space (where the
# typical similarity between two unrelated texts is 0.2-0.3, so "neighbourhood"
# is illusory), but in the OUTPUT space: using the vector from the stage-I
# regression, where neighbours have cos 0.6-0.8. Training coordinates must be
# the LOO prediction (y_i - e_i), not the in-sample prediction: at small
# lambda the latter is nearly equal to y_i, which is not the case on the test
# text, and the correction would be learning leakage.

# ---------------------------------------------------------------- tools


def l2norm(Y: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(Y, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return Y / n


def tricube(u: np.ndarray) -> np.ndarray:
    """Cleveland's (1979) tricube kernel: (1-u^3)^3 on [0,1], 0 otherwise.

    Powers computed by multiplication, not the `**` operator: for exponent 3 numpy calls
    `pow` from libm, and `pow` is NOT covered by the exact-rounding requirement
    in IEEE-754, two math libraries can give results differing by 1 ulp.
    Multiplication is rounded exactly, so this version gives the same bit
    on every IEEE-754-compliant platform.
    """
    u = np.clip(u, 0.0, 1.0)
    t = 1.0 - u * u * u
    return t * t * t


def csr_matvec_numpy(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    q_idx: np.ndarray,
    q_val: np.ndarray,
    n_buckets: int,
    buf: np.ndarray | None = None,
) -> np.ndarray:
    """K_* = X_train @ q, purely on numpy (without scipy), archival path.

    The buffer is cleared only at the affected positions, so the cost does
    not depend on the number of buckets.
    """
    if buf is None:
        buf = np.zeros(n_buckets, dtype=np.float64)
    buf[q_idx] = q_val
    prod = buf[indices] * data
    out = np.add.reduceat(prod, indptr[:-1])
    empty = np.diff(indptr) == 0
    if empty.any():
        out[empty] = 0.0
    buf[q_idx] = 0.0
    return out


# ------------------------------------------------- core: dual form
#
# The entire computation reduces to two kernel matrices:
# K  (N x N)  train-train similarity,
# Ks (M x N)  query-train similarity,
# and the target matrix Ytr (N x d). Thanks to this, the same function handles
# the `Surrogate` class and the method comparison in `run_experiment`: there are no two
# implementations that could diverge.


def solve_ridge_loo_f32(G32: np.ndarray, Ytr: np.ndarray, lam: float):
    """alpha, diag(G^-1) and LOO prediction for large N, without a second copy of the matrix.

    Why float32 and why this gymnastics: for N = 43,000 the Gram matrix in float64
    is 14.8 GB, and `np.linalg.inv` needs a second one just like it. In float32
    it is 7.4 GB, and we do the Cholesky decomposition and triangular inverse
    IN PLACE, so peak usage is one matrix.

    The loss of precision is harmless because the matrix is well-conditioned:
    G = K + lambda*N*I, and with lambda*N of order unity the conditioning is
    ~10^2-10^3, i.e. the float32 relative error (1e-7) gives ~1e-4 on alpha.
    The archival contract is not affected by this: alpha is FROZEN AS DATA, and
    the inference path computes in float64 and quantizes to 6 places, training
    precision affects WHICH model we save, not its reproducibility.

    G32 is destroyed (overwritten by the decomposition, then by the triangular inverse).
    Returns (alpha, diag_inv, z_loo) in float64.
    """
    from scipy.linalg import cho_factor, cho_solve, lapack

    n = G32.shape[0]
    c, low = cho_factor(G32, lower=True, overwrite_a=True, check_finite=False)
    alpha = cho_solve((c, low), Ytr.astype(np.float32), check_finite=False)
    Linv, info = lapack.strtri(c, lower=1, overwrite_c=1)
    if info != 0:
        raise RuntimeError(f"strtri info={info}")
    # cho_factor/strtri leave garbage (input residuals) in the UNUSED triangle,
    # not zeros, summing over the full column would include them together with
    # the proper terms. We zero them in place, block by block, without copying
    # the matrix.
    B = 2048
    for c0 in range(0, n, B):
        c1 = min(c0 + B, n)
        if c0:
            Linv[:c0, c0:c1] = 0.0
        blk = Linv[c0:c1, c0:c1]
        iu = np.triu_indices(c1 - c0, 1)
        blk[iu] = 0.0
    # G^-1 = L^-T L^-1  =>  diag(G^-1)_i = sum_k (L^-1)_{ki}^2
    diag_inv = np.einsum("ki,ki->i", Linv, Linv, optimize=True).astype(np.float64)
    alpha = alpha.astype(np.float64)
    z_loo = l2norm(Ytr - alpha / diag_inv[:, None])
    return alpha, diag_inv, z_loo


def fit_dual(method: str, K: np.ndarray, Ytr: np.ndarray, **p) -> dict:
    """Returns the model state (dictionary) computed exclusively from K and Ytr."""
    st: dict = {"method": method, "Y": Ytr, **p}
    if method in ("ridge", "hybrid"):
        st["alpha"] = _solve_ridge(K, Ytr, p.get("lam", 1e-4))
    elif method == "rbf":
        st["alpha"] = _solve_ridge(_rbf(K, p.get("gamma", 1.0)), Ytr, p.get("lam", 1e-4))
    if method in ("loess", "hybrid") and p.get("degree", 1) > 0:
        st["U"], st["lam_eig"] = _kernel_coords(K, p.get("rank", 32))
    if method == "hybrid":
        st["R"] = Ytr - K @ st["alpha"]
    if method == "hybrid_z":
        lam = p.get("lam", 1e-4)
        n = K.shape[0]
        st["alpha"] = _solve_ridge(K, Ytr, lam)
        st["R"] = Ytr - K @ st["alpha"]
        Gi = np.linalg.inv(K + lam * n * np.eye(n))
        st["z"] = l2norm(Ytr - (Gi @ Ytr) / np.diag(Gi)[:, None])   # LOO prediction
    return st


def predict_dual(st: dict, Ks: np.ndarray) -> np.ndarray:
    """Prediction from a ready state; result normalized to a norm of 1."""
    m, Y = st["method"], st["Y"]
    k = st.get("k", 150)
    deg = st.get("degree", 1)
    if m == "knn1":
        out = Y[np.argmax(Ks, axis=1)]
    elif m == "nw":
        out = _local(Ks, Y, k, degree=0)
    elif m == "loess":
        out = _local(Ks, Y, k, degree=deg, U=st.get("U"), lam_eig=st.get("lam_eig"),
                     ridge=st.get("ridge_local", 1e-6))
    elif m == "ridge":
        out = Ks @ st["alpha"]
    elif m == "rbf":
        out = _rbf(Ks, st.get("gamma", 1.0)) @ st["alpha"]
    elif m == "hybrid_z":
        b = Ks @ st["alpha"]
        out = b + _local(l2norm(b) @ st["z"].T, st["R"], k, degree=0)
    elif m == "hybrid":
        out = Ks @ st["alpha"] + _local(
            Ks, st["R"], k, degree=deg, U=st.get("U"), lam_eig=st.get("lam_eig"),
            ridge=st.get("ridge_local", 1e-6))
    else:  # pragma: no cover
        raise AssertionError(m)
    return l2norm(out)


# ------------------------------------------------------------- internal


def _solve_ridge(K: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """alpha = (K + lam*N*I)^-1 Y, closed-form solution, via Cholesky."""
    n = K.shape[0]
    A = K + (lam * n) * np.eye(n)
    L = np.linalg.cholesky(A)
    z = np.linalg.solve(L, Y)
    return np.linalg.solve(L.T, z)


def _rbf(K: np.ndarray, gamma: float) -> np.ndarray:
    """Gaussian kernel expressed via cosine: ||a-b||^2 = 2-2cos."""
    return np.exp(-gamma * np.maximum(2.0 - 2.0 * K, 0.0))


def _kernel_coords(K: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Kernel-PCA coordinates (eigendecomposition of K), basis of local regression.

    For a query we use the Nyström extension: z_* = k_*^T U Lambda^-1/2.
    eigh on a symmetric matrix is deterministic with respect to the order
    of eigenvalues; the sign of the eigenvector is fixed explicitly.
    """
    w, V = np.linalg.eigh(K)
    order = np.argsort(w)[::-1][:rank]
    w = np.maximum(w[order], 1e-12)
    V = V[:, order]
    # unambiguous sign: the coordinate with the largest absolute value must be positive
    flip = np.sign(V[np.argmax(np.abs(V), axis=0), np.arange(V.shape[1])])
    flip[flip == 0] = 1.0
    return V * flip, w


def _local(
    ks: np.ndarray,
    Y: np.ndarray,
    k: int,
    *,
    degree: int = 0,
    U: np.ndarray | None = None,
    lam_eig: np.ndarray | None = None,
    ridge: float = 1e-6,
) -> np.ndarray:
    """Local regression: degree 0 = Nadaraya-Watson, degree 1 = LOESS.

    The neighbourhood is determined by the k largest cosines. Distance d = 1 - cos,
    scaled by the neighbourhood radius (adaptive window, like `span` in LOESS),
    tricube weights. For degree 1 the fit proceeds in Nyström coordinates
    centred on the query point, so the intercept IS
    the prediction at that point.
    """
    nq, n = ks.shape
    k = min(k, n)
    out = np.empty((nq, Y.shape[1]), dtype=np.float64)

    Z = None
    if degree > 0:
        if U is None or lam_eig is None:
            raise ValueError("LOESS requires kernel coordinates (U, lam_eig)")
        Z = U * np.sqrt(lam_eig)                      # training coordinates
        Wn = U / np.sqrt(lam_eig)                     # extension matrix

    # The choice of neighbours MUST be resolved explicitly. `argpartition` and `argsort`
    # return implementation-dependent ordering in case of a tie, and ties do
    # occur here: the measured smallest gap between the k-th and (k+1)-th
    # neighbour is 2e-16, and out of 1200 queries 44 had a different ORDER,
    # and 2 a different SET of neighbours than the reference implementation.
    # A different set is a different function, not a different rounding error, # i.e. a broken archival contract. Ordering key: (-similarity, training index).
    ar = np.arange(n)
    for i in range(nq):
        nb = np.lexsort((ar, -ks[i]))[:k]
        sims = ks[i, nb]
        d = 1.0 - sims
        # radius: distance to the farthest neighbour (+eps, so as not to zero it out
        # completely)
        h = d[-1] * 1.0000001
        if h <= 0:                                     # exact hit
            out[i] = Y[nb[0]]
            continue
        w = tricube(d / h)
        if w.sum() <= 0:
            out[i] = Y[nb[0]]
            continue
        if degree == 0:
            out[i] = (w @ Y[nb]) / w.sum()
            continue
        zq = ks[i] @ Wn                                # Nyström for query
        A = np.empty((k, 1 + Z.shape[1]), dtype=np.float64)
        A[:, 0] = 1.0
        A[:, 1:] = Z[nb] - zq                          # center of the system in the query
        AW = A * w[:, None]
        G = A.T @ AW
        G.flat[:: G.shape[0] + 1] += ridge             # stabilization
        beta = np.linalg.solve(G, AW.T @ Y[nb])
        out[i] = beta[0]                               # intercept = prediction
    return out
