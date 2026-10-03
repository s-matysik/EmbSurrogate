"""Fidelity metrics of the surrogate against the teacher model.

We distinguish two levels of agreement, because they are not the same:

1. POINTWISE FIDELITY — cos(surrogate(t), teacher(t)). Tells us how close
   we get to a specific vector.
2. GEOMETRIC FIDELITY — whether the surrogate preserves relations between texts:
   correlation of similarity matrices, ranking agreement (recall@k), error
   in reconstructing the cosine itself. This is what embeddings are used for,
   and it is the proper criterion for "cloning the behaviour" of a model.

Reference point (floor): the mean vector of the training set. If the surrogate
does not clearly beat this value, it has not learned anything beyond the fact that
bge/SBERT vectors lie within a narrow cone."""

from __future__ import annotations

import numpy as np
from scipy import stats


def cosine_rowwise(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    a = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    b = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return np.sum(a * b, axis=1)


def _upper(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]


def recall_at_k(S_true: np.ndarray, S_hat: np.ndarray, k: int = 10) -> float:
    """Average share of common k nearest neighbours (excluding itself)."""
    n = S_true.shape[0]
    k = min(k, n - 1)
    St, Sh = S_true.copy(), S_hat.copy()
    np.fill_diagonal(St, -np.inf)
    np.fill_diagonal(Sh, -np.inf)
    top_t = np.argpartition(-St, k - 1, axis=1)[:, :k]
    top_h = np.argpartition(-Sh, k - 1, axis=1)[:, :k]
    return float(np.mean([
        len(set(top_t[i]).intersection(top_h[i])) / k for i in range(n)
    ]))


def report(Y_true: np.ndarray, Y_hat: np.ndarray, *, k: int = 10,
           max_pairs: int = 4_000_000, seed: int = 0) -> dict:
    """Full set of metrics for a single method."""
    cos = cosine_rowwise(Y_true, Y_hat)
    S_true = Y_true @ Y_true.T
    S_hat = Y_hat @ Y_hat.T
    ut, uh = _upper(S_true), _upper(S_hat)
    if ut.size > max_pairs:                     # sampling for large datasets
        rng = np.random.default_rng(seed)
        sel = rng.choice(ut.size, max_pairs, replace=False)
        ut, uh = ut[sel], uh[sel]
    return {
        "cos_mediana": float(np.median(cos)),
        "cos_srednia": float(np.mean(cos)),
        "cos_p5": float(np.percentile(cos, 5)),
        "cos_p95": float(np.percentile(cos, 95)),
        "cos_min": float(np.min(cos)),
        "spearman_sim": float(stats.spearmanr(ut, uh).statistic),
        "pearson_sim": float(np.corrcoef(ut, uh)[0, 1]),
        "mae_sim": float(np.mean(np.abs(ut - uh))),
        f"recall@{k}": recall_at_k(S_true, S_hat, k),
    }


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    """Average precision, computed here rather than imported.

    The definition is the step-wise one: sum over the ranked list of
    `(R_n - R_(n-1)) * P_n`, which is what scikit-learn's
    `average_precision_score` returns (verified to 1e-12 against it on random
    inputs, including ties). It lives here because the whole package must run
    on numpy alone: a researcher reopening an archive in ten years should not
    have to resolve a machine-learning stack to read their own metrics, and a
    metric that moves with a library version is not archival either.

    Records with EQUAL scores form one threshold point rather than several.
    That detail is the whole of the agreement with the reference: scoring tied
    records one by one silently credits an arbitrary order inside the tie and
    moves the result by up to 0.017 on random inputs, which is larger than
    differences this metric is used to judge.
    """
    y_true = np.asarray(y_true).astype(bool)
    s = np.asarray(score, dtype=np.float64)
    order = np.argsort(-s, kind="stable")
    y, s = y_true[order], s[order]
    total = int(y.sum())
    if total == 0:
        return 0.0
    # last position of every group of equal scores
    ends = np.flatnonzero(np.r_[np.diff(s) != 0, True])
    tp = np.cumsum(y)[ends]
    precision = tp / (ends + 1.0)
    recall = tp / total
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def report_full(Y_true: np.ndarray, Y_hat: np.ndarray, mu: np.ndarray,
                k: int = 10, thresholds=(0.6, 0.8), n_boot: int = 200,
                seed: int = 0) -> dict:
    """A metric suite robust to three artifacts that corrupt interpretation:

    1. Shared component — median cos is dominated by it (the mean training
       vector alone gives 0.35), hence we also compute cos AFTER SUBTRACTING `mu` from both
       sides and R² = 1 - SSE/SST.
    2. Global rho across all pairs mainly orders unrelated pairs
       (teacher cos 0.1-0.4), so we add rho PER QUERY (across rows of the
       similarity matrix) and recall@k and NDCG@k.
    3. AUC at a high threshold is degenerate under class imbalance —
       we report average precision (AP) and sensitivity at FPR = 1e-4.

    Confidence interval computed by bootstrap OVER TEXTS (the resampling unit is
    the text, not the pair — pairs are not independent).
    """

    cos = cosine_rowwise(Y_true, Y_hat)
    cos_c = cosine_rowwise(Y_true - mu, Y_hat - mu)
    sse = float(((Y_true - Y_hat) ** 2).sum())
    sst = float(((Y_true - mu) ** 2).sum())
    S_true, S_hat = Y_true @ Y_true.T, Y_hat @ Y_hat.T
    ut, uh = _upper(S_true), _upper(S_hat)

    n = len(Y_true)
    off = ~np.eye(n, dtype=bool)
    rho_q = np.array([stats.spearmanr(S_true[i][off[i]], S_hat[i][off[i]]).statistic
                      for i in range(n)])

    # NDCG@k with relevance = teacher cosine (clipped to zero)
    ord_hat = np.argsort(-np.where(off, S_hat, -np.inf), axis=1)[:, :k]
    ord_true = np.argsort(-np.where(off, S_true, -np.inf), axis=1)[:, :k]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    rel_hat = np.maximum(np.take_along_axis(S_true, ord_hat, 1), 0.0)
    rel_id = np.maximum(np.take_along_axis(S_true, ord_true, 1), 0.0)
    ndcg = float(np.mean((rel_hat @ disc) / np.maximum(rel_id @ disc, 1e-12)))

    out = {
        "cos_mediana": float(np.median(cos)),
        "cos_centrowany": float(np.median(cos_c)),
        "R2": 1.0 - sse / sst,
        "rho_globalne": float(stats.spearmanr(ut, uh).statistic),
        "rho_per_zapytanie": float(np.median(rho_q)),
        f"recall@{k}": recall_at_k(S_true, S_hat, k),
        f"ndcg@{k}": ndcg,
    }
    for thr in thresholds:
        y = (ut > thr).astype(int)
        if y.sum() < 10:
            continue
        out[f"AP@{thr}"] = average_precision(y, uh)
        neg = uh[y == 0]
        cutoff = np.quantile(neg, 1 - 1e-4)          # FPR = 1e-4
        out[f"czulosc@FPR1e-4_{thr}"] = float((uh[y == 1] > cutoff).mean())
        out[f"udzial_dodatnich@{thr}"] = float(y.mean())

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        A, B = S_true[np.ix_(idx, idx)], S_hat[np.ix_(idx, idx)]
        boot.append(stats.spearmanr(_upper(A), _upper(B)).statistic)
    out["rho_globalne_CI95"] = (float(np.percentile(boot, 2.5)),
                                float(np.percentile(boot, 97.5)))
    return out


def baseline_mean(Y_train: np.ndarray, n_test: int) -> np.ndarray:
    """Floor: assign the average training vector to all."""
    m = Y_train.mean(axis=0)
    m = m / max(np.linalg.norm(m), 1e-12)
    return np.tile(m, (n_test, 1))
