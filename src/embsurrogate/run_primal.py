"""Surrogate for large N — iterative solver, without materializing the Gram matrix.

    python3 -m klon.run_primal data/ds_big3 model/surogat_v3.npz wyniki/v3

Why this way: `run_big.py` holds an N x N matrix. Cholesky decomposition for
N = 43 000 (7.4 GB in float32) gets killed in this environment — the ceiling lies
somewhere between 20 and 28 thousand pairs. Here the matrix never gets built:
conjugate gradients only need the product K v = Phi (Phi^T v), and Phi is sparse
(~815 nonzeros per text). Memory usage does not depend on N quadratically:
the dominant term is the intermediate array D x 384 (402 MB in float32).

Convergence measured at N = 9600 against the exact solution: 29 iterations for
tol = 1e-3 and 39 for 1e-4, with rho matching the exact value to 5 places.

Coordinates for residual correction (`z`) are computed K times out-of-sample (OOF), because
in this form there is no closed formula for LOO prediction. Measurement at N = 9600:
LOO 0.7688; pure teacher vector 0.7573; without correction 0.7516 — OOF is
worth the extra system solves."""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys
import time

import numpy as np

from . import dual as D
from . import evaluate as E
from . import features as F
from . import freeze as FZ
from .run_big import indices
from .run_experiment import load_dataset


def dual_cg(X, Y, lam: float, tol: float = 1e-4, max_iter: int = 200,
            verbose: bool = False):
    """Solves (K + lambda*N*I) a = Y, where K = X X^T (never formed)."""
    n = X.shape[0]
    shift = np.float32(lam * n)
    a = np.zeros_like(Y)
    r = Y.copy()
    p = r.copy()
    rs = (r * r).sum(0)
    rs0 = rs.copy()
    for it in range(max_iter):
        Ap = X @ (X.T @ p) + shift * p
        al = rs / np.maximum((p * Ap).sum(0), 1e-30)
        a += p * al
        r -= Ap * al
        rs_new = (r * r).sum(0)
        rel = float(np.sqrt(rs_new / rs0).max())
        if verbose and it % 10 == 0:
            print(f"    CG iter {it:3d}  rel = {rel:.2e}", flush=True)
        if rel < tol:
            break
        p = r + p * (rs_new / np.maximum(rs, 1e-30))
        rs = rs_new
    return a, it + 1, rel


def oof_coordinates(X, Y, lam: float, folds: int = 4, tol: float = 1e-4,
                    verbose: bool = True):
    """Out-of-sample predictions for each training row (K-fold split).

    The split is deterministic: the i-th row goes to fold i % folds.
    """
    n = X.shape[0]
    z = np.zeros_like(Y)
    fold_id = np.arange(n) % folds
    for f in range(folds):
        tr = np.where(fold_id != f)[0]
        ho = np.where(fold_id == f)[0]
        t = time.time()
        Xtr = X[tr]
        af, nit, rel = dual_cg(Xtr, Y[tr].copy(), lam, tol=tol)
        Ks = (X[ho] @ Xtr.T).astype(np.float32)
        z[ho] = np.asarray(Ks @ af)
        del Xtr, Ks, af
        gc.collect()
        if verbose:
            print(f"  fold {f+1}/{folds}: {nit}iterations, rel {rel:.1e},"
                  f" {time.time()-t:.0f}s", flush=True)
    return D.l2norm(z.astype(np.float64))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=pathlib.Path)
    ap.add_argument("out_model", type=pathlib.Path)
    ap.add_argument("outdir", type=pathlib.Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--ks", type=int, nargs="+", default=[20, 50, 100, 200])
    args = ap.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    texts, Y, _ = load_dataset(args.dataset)
    val, t0, pool = indices(len(texts))
    if args.limit:
        pool = pool[: args.limit]
    n = len(pool)
    cfg = dict(F.DEFAULT_CONFIG)
    print(f"pool {n} | validation {len(val)} | T0 {len(t0)}", flush=True)

    t = time.time()
    Xp, idf = F.fit_transform([texts[i] for i in pool], cfg)
    Xp = Xp.astype(np.float32)
    Xv = F.transform([texts[i] for i in val], idf, cfg).astype(np.float32)
    X0 = F.transform([texts[i] for i in t0], idf, cfg).astype(np.float32)
    print(f"features {Xp.shape} nnz/text {round(Xp.nnz/n)} ({time.time()-t:.0f}s)",
          flush=True)

    Ytr32 = Y[pool].astype(np.float32)
    t = time.time()
    alpha32, nit, rel = dual_cg(Xp, Ytr32.copy(), args.lam, tol=args.tol, verbose=True)
    print(f"CG: {nit}iterations, rel {rel:.1e} ({time.time()-t:.0f}s)", flush=True)
    alpha = alpha32.astype(np.float64)
    Ytr = Y[pool]
    R = Ytr - np.asarray(Xp @ (Xp.T @ alpha32), dtype=np.float64)

    t = time.time()
    z = oof_coordinates(Xp, Ytr32, args.lam, folds=args.folds, tol=args.tol)
    print(f"OOF ({time.time()-t:.0f}s)", flush=True)

    preds = {}
    for nm, Xq, idx in (("walidacja", Xv, val), ("T0", X0, t0)):
        Ks = np.asarray((Xq @ Xp.T).todense(), dtype=np.float64)
        b = Ks @ alpha
        preds[nm] = (idx, b, D.l2norm(b) @ z.T)
        del Ks
        gc.collect()

    # we select k on the validation set, we report on T0
    idxv, bv, Szv = preds["walidacja"]
    scores = {}
    for k in args.ks:
        scores[k] = E.report(Y[idxv], D.l2norm(bv + D._local(Szv, R, k, degree=0)))["spearman_sim"]
        print(f"  validation k={k}: rho {scores[k]:.4f}", flush=True)
    k_best = max(scores, key=scores.get)

    mu = Ytr.mean(0)
    metrics = {"N": n, "lam": args.lam, "k": k_best, "cg_iter": nit,
               "wybor_k_na_walidacji": {str(k): float(v) for k, v in scores.items()}}
    for nm, (idx, b, Sz) in preds.items():
        yr = D.l2norm(b)
        yh = D.l2norm(b + D._local(Sz, R, k_best, degree=0))
        metrics[nm] = {"ridge": E.report(Y[idx], yr),
                       "hybrid_z": E.report_full(Y[idx], yh, mu)}
        print(nm, "ridge rho", round(metrics[nm]["ridge"]["spearman_sim"], 4),
              "| hybrid_z rho", round(metrics[nm]["hybrid_z"]["rho_globalne"], 4),
              flush=True)
    (args.outdir / "metryki.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    man = FZ.build_explicit(
        [texts[i] for i in pool], [str(i) for i in pool], Ytr, cfg,
        "hybrid_z", {"lam": args.lam, "k": k_best, "degree": 0},
        {"idf": idf, "alpha": alpha, "R": R, "z": z}, args.out_model,
        meta_extra={"dataset": str(args.dataset),
                    "uklad": "pool / validation 1200 / T0 1200 (fresh)",
                    "rho_globalne_T0": metrics["T0"]["hybrid_z"]["rho_globalne"],
                    "solver": f"conjugate gradients (tol {args.tol}, {nit} iterations),"
                              f" wspolrzedne z: OOF {args.folds}-krotny"})
    print("frozen", man["plik"], man["sha256_pliku"][:16], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
