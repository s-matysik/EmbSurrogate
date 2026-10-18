"""Comparison of surrogate variants on a set of (text, vector) pairs.

    python3 -m klon.run_experiment data/ds_multi wyniki/multi
    python3 -m klon.run_experiment data/ds_multi wyniki/multi --ceiling

Saves:
    metryki.csv          all variants from the grid, sorted
    najlepszy.json       winning configuration + its metrics
    krzywa_uczenia.csv   fidelity as a function of the number of training pairs
    sufit.csv            (with --ceiling) reference: features from another
                         neural model instead of n-grams, shows how much is
                         lost by the feature space itself, and how much by the
                         regression method

The train/test split is deterministic (permutation with a fixed
seed), so the result can be reproduced down to the row."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

from . import evaluate as E
from . import features as F
from . import surrogate as S


def load_dataset(d: pathlib.Path, limit: int = 0):
    Y = np.load(d / "vectors.npy").astype(np.float64)
    meta = [json.loads(l) for l in (d / "meta.jsonl").open(encoding="utf-8") if l.strip()]
    texts = [m["text"] for m in meta]
    if limit:
        Y, texts, meta = Y[:limit], texts[:limit], meta[:limit]
    assert len(texts) == Y.shape[0], "meta.jsonl and vectors.npy have different lengths"
    return texts, Y, meta


def split(n: int, test_frac: float, seed: int):
    perm = np.random.default_rng(seed).permutation(n)
    ntr = int(round((1.0 - test_frac) * n))
    return perm[:ntr], perm[ntr:]


def kernels(texts_tr, texts_te, cfg):
    Xtr, idf = F.fit_transform(texts_tr, cfg)
    Xte = F.transform(texts_te, idf, cfg)
    K = np.asarray((Xtr @ Xtr.T).todense(), dtype=np.float64)
    np.fill_diagonal(K, 1.0)
    Ks = np.asarray((Xte @ Xtr.T).todense(), dtype=np.float64)
    return K, Ks, idf, Xtr


GRID: list[tuple[str, dict]] = (
    [("knn1", {})]
    + [("nw", {"k": k}) for k in (10, 25, 50, 150, 400)]
    + [("loess", {"k": k, "rank": r, "degree": 1})
       for k in (50, 150, 400) for r in (8, 16, 32, 64)]
    + [("ridge", {"lam": l}) for l in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)]
    + [("rbf", {"lam": l, "gamma": g})
       for l in (1e-5, 1e-3) for g in (0.5, 1.0, 2.0, 4.0)]
    + [("hybrid", {"lam": l, "k": k, "rank": 32, "degree": d})
       for l in (1e-4, 1e-3, 1e-2) for k in (25, 50, 150) for d in (0, 1)]
)


def run_grid(K, Ks, Ytr, Yte, grid=GRID) -> pd.DataFrame:
    rows = []
    for method, params in grid:
        t0 = time.time()
        st = S.fit_dual(method, K, Ytr, **params)
        Yh = S.predict_dual(st, Ks)
        r = E.report(Yte, Yh)
        r.update(metoda=method, **{f"p_{k}": v for k, v in params.items()},
                 sekundy=round(time.time() - t0, 2))
        rows.append(r)
        print(f"  {method:8s} {params} -> cos={r['cos_mediana']:.4f} "
              f"rho={r['spearman_sim']:.4f}", flush=True)
    rows.append({**E.report(Yte, E.baseline_mean(Ytr, Yte.shape[0])),
                 "metoda": "floor_sredni_wektor"})
    keys = {k for r in rows for k in r}          # not just from the first row:
    metric_cols = [k for k in keys if not k.startswith("p_") and k != "metoda"]
    cols = ["metoda"] + sorted(k for k in keys if k.startswith("p_")) + sorted(metric_cols)
    return pd.DataFrame(rows).reindex(columns=cols)


def learning_curve(texts, Y, tr, te, cfg, method, params, sizes) -> pd.DataFrame:
    rows = []
    for n in sizes:
        sub = tr[:n]
        K, Ks, _, _ = kernels([texts[i] for i in sub], [texts[i] for i in te], cfg)
        st = S.fit_dual(method, K, Y[sub], **params)
        r = E.report(Y[te], S.predict_dual(st, Ks))
        r.update(n_treningowych=int(n), metoda=method)
        rows.append(r)
        print(f"  N={n}: cos={r['cos_mediana']:.4f} rho={r['spearman_sim']:.4f}", flush=True)
    return pd.DataFrame(rows)


def ceiling(texts, Y, tr, te, model_name: str) -> pd.DataFrame:
    """Upper reference: instead of n-grams we take vectors of ANOTHER
    neural model and perform the same kernel regression on them."""
    from sentence_transformers import SentenceTransformer

    ref = SentenceTransformer(model_name)
    P = ref.encode([texts[i] for i in np.concatenate([tr, te])], batch_size=64,
                   convert_to_numpy=True, normalize_embeddings=True,
                   show_progress_bar=False).astype(np.float64)
    Ptr, Pte = P[: len(tr)], P[len(tr):]
    K, Ks = Ptr @ Ptr.T, Pte @ Ptr.T
    rows = []
    for lam in (1e-6, 1e-5, 1e-4, 1e-3):
        st = S.fit_dual("ridge", K, Y[tr], lam=lam)
        r = E.report(Y[te], S.predict_dual(st, Ks))
        r.update(metoda="ridge_na_cechach_neuronowych", p_lam=lam, model=model_name)
        rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=pathlib.Path)
    ap.add_argument("outdir", type=pathlib.Path)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-chars", type=int, default=F.DEFAULT_CONFIG["max_chars"])
    ap.add_argument("--ceiling", action="store_true")
    ap.add_argument("--ceiling-model",
                    default="sentence-transformers/distiluse-base-multilingual-cased-v1")
    args = ap.parse_args(argv)

    cfg = dict(F.DEFAULT_CONFIG, max_chars=args.max_chars)
    texts, Y, _ = load_dataset(args.dataset, args.limit)
    tr, te = split(len(texts), args.test_frac, args.seed)
    print(f"trening {len(tr)}, test {len(te)}, wymiar {Y.shape[1]}")

    t0 = time.time()
    K, Ks, _, Xtr = kernels([texts[i] for i in tr], [texts[i] for i in te], cfg)
    print(f"features+kernel: {time.time()-t0:.1f}s, nnz/text {Xtr.nnz/Xtr.shape[0]:.0f}")

    df = run_grid(K, Ks, Y[tr], Y[te])
    args.outdir.mkdir(parents=True, exist_ok=True)
    df_sorted = df.sort_values("spearman_sim", ascending=False)
    df_sorted.to_csv(args.outdir / "metryki.csv", index=False)

    best = df_sorted.iloc[0]
    params = {k[2:]: v for k, v in best.items()
              if k.startswith("p_") and pd.notna(v)}
    for key in ("k", "rank", "degree"):
        if key in params:
            params[key] = int(params[key])
    (args.outdir / "najlepszy.json").write_text(json.dumps(
        {"metoda": best["metoda"], "parametry": params,
         "metryki": {k: float(v) for k, v in best.items()
                     if isinstance(v, (int, float, np.floating)) and not k.startswith("p_")},
         "cfg": cfg, "dataset": str(args.dataset), "seed": args.seed},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nnajlepszy: {best['metoda']} {params}")

    sizes = [n for n in (250, 500, 1000, 2000, 3000, 4000, len(tr)) if n <= len(tr)]
    lc = learning_curve(texts, Y, tr, te, cfg, best["metoda"], params, sizes)
    lc.to_csv(args.outdir / "krzywa_uczenia.csv", index=False)

    if args.ceiling:
        ceiling(texts, Y, tr, te, args.ceiling_model).to_csv(
            args.outdir / "sufit.csv", index=False)
    print(f"results -> {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
