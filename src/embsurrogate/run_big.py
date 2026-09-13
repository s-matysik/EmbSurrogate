"""Fitting the surrogate on a large set of pairs — separate process, low memory.

    python3 -m klon.run_big data/ds_big3 model/surogat_v3.npz wyniki/v3

Why a separate script rather than a cell in the session: the Gram matrix for 43k pairs
is 7.4 GB in float32, and the Cholesky decomposition is done in place. A process that
previously computed something else already has memory occupied and the allocation fails
(in my case the kernel got SIGUSR1). A fresh process starts with ~100 MB.

The split is reproducible from seeds, and not saved to a file:
    validation = old 1200 (split(6000, 0.2, seed=0)) — variants were selected on it
    T0         = indices 18001..19200 — fresh test, single use
    pool       = the rest, in sequence: rng(1) on 0..19200, rng(2) on 19201..28021,
                 rng(3) on 28022..end — nested order, so
                 subsets from the learning curve are prefixes of the same list."""

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
from .run_experiment import load_dataset, split


def indices(n_total: int, n_stage2: int = 19201, n_stage3: int = 28022):
    _, val = split(6000, 0.2, 0)
    t0 = np.arange(n_stage2 - 1200, n_stage2)
    p1 = np.setdiff1d(np.arange(n_stage2), np.concatenate([val, t0]))
    p1 = np.random.default_rng(1).permutation(p1)[:16801]
    parts = [p1]
    if n_total > n_stage2:
        parts.append(np.random.default_rng(2).permutation(
            np.arange(n_stage2, min(n_stage3, n_total))))
    if n_total > n_stage3:
        parts.append(np.random.default_rng(3).permutation(
            np.arange(n_stage3, n_total)))
    return val, t0, np.concatenate(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=pathlib.Path)
    ap.add_argument("out_model", type=pathlib.Path)
    ap.add_argument("outdir", type=pathlib.Path)
    ap.add_argument("--limit", type=int, default=0, help="truncate the training pool")
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    texts, Y, meta = load_dataset(args.dataset)
    val, t0, pool = indices(len(texts))
    if args.limit:
        pool = pool[: args.limit]
    n = len(pool)
    cfg = dict(F.DEFAULT_CONFIG)
    print(f"pool {n} | validation {len(val)} | T0 {len(t0)} | G = {n*n*4/1e9:.1f} GB",
          flush=True)

    t = time.time()
    Xp, idf = F.fit_transform([texts[i] for i in pool], cfg)
    Xv = F.transform([texts[i] for i in val], idf, cfg)
    X0 = F.transform([texts[i] for i in t0], idf, cfg)
    print(f"features {Xp.shape} nnz/text {round(Xp.nnz/n)} ({time.time()-t:.0f}s)", flush=True)

    t = time.time()
    G = np.empty((n, n), dtype=np.float32, order="F")
    for i0 in range(0, n, 1000):
        i1 = min(i0 + 1000, n)
        G[i0:i1, :] = (Xp[i0:i1] @ Xp.T).toarray().astype(np.float32)
    G.flat[:: n + 1] = 1.0 + args.lam * n
    print(f"Gram ({time.time()-t:.0f}s)", flush=True)

    t = time.time()
    Ytr = Y[pool]
    alpha, diag_inv, z = D.solve_ridge_loo_f32(G, Ytr, args.lam)
    del G
    gc.collect()
    R = Ytr - (Xp @ (Xp.T @ alpha))
    print(f"Cholesky + LOO ({time.time()-t:.0f}s)", flush=True)

    mu = Ytr.mean(0)
    metrics = {}
    for nm, Xq, idx in (("walidacja", Xv, val), ("T0", X0, t0)):
        Ks = np.asarray((Xq @ Xp.T).todense())
        b = Ks @ alpha
        yr = D.l2norm(b)
        yh = D.l2norm(b + D._local(D.l2norm(b) @ z.T, R, args.k, degree=0))
        del Ks
        gc.collect()
        metrics[nm] = {"ridge": E.report(Y[idx], yr),
                       "hybrid_z": E.report_full(Y[idx], yh, mu)}
        print(nm, "ridge rho", round(metrics[nm]["ridge"]["spearman_sim"], 4),
              "| hybrid_z rho", round(metrics[nm]["hybrid_z"]["rho_globalne"], 4),
              flush=True)

    (args.outdir / "metryki.json").write_text(
        json.dumps({"N": n, "lam": args.lam, "k": args.k, "metryki": metrics},
                   indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    man = FZ.build_explicit(
        [texts[i] for i in pool], [str(i) for i in pool], Ytr, cfg,
        "hybrid_z", {"lam": args.lam, "k": args.k, "degree": 0},
        {"idf": idf, "alpha": alpha, "R": R, "z": z}, args.out_model,
        meta_extra={"dataset": str(args.dataset),
                    "uklad": "pool / validation 1200 / T0 1200 (fresh)",
                    "rho_globalne_T0": metrics["T0"]["hybrid_z"]["rho_globalne"],
                    "solver": "float32 Cholesky + strtri (training), float64 inference"})
    print("frozen", man["plik"], man["sha256_pliku"][:16], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
