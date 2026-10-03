"""Screen a corpus for inclusion, reproducibly, with honest baselines.

    embsurrogate select --vectors vectors.npy --texts corpus.jsonl \
        --seeds seeds.txt --budget 0.2 --out selection/

The ranking a researcher actually needs is not "most similar to a query" but
"read these first". This module runs the loop that screening practice uses --
read a batch, label it, refit, reorder the rest -- and writes the reading order
together with a report of what it is based on.

Three design decisions, each measured rather than assumed (see docs/METHOD.md):

* **The classifier is ours, not scikit-learn's.** Kernel logistic regression
  solved by damped Newton steps in pure numpy, with a wide-prior intercept. The
  reason is the determinism contract: a reading order that depends on which
  version of a solver happened to be installed cannot be reproduced years
  later. It costs nothing in quality -- on two CLEF topics used for tuning,
  recall at a 10% reading budget is 0.551 against 0.550 for scikit-learn's
  logistic regression, while plain ridge regression (simpler still, no Newton
  loop) loses 0.035. The two hyperparameters are the ones that matter and both
  are set to the values that recover the reference: the penalty equals
  scikit-learn's default `C = 1`, and the intercept is what closes the rest of
  the gap, because early in a screen the labelled set is small and lopsided.
* **Baselines run in the same call.** tf-idf similarity and random order are
  computed alongside, because on five Cochrane reviews the lexical baseline
  matched or beat the embedding model. A tool that hides that is selling the
  embedding step rather than measuring it.
* **Nothing is sampled.** Seed order, tie-breaking and batch selection are
  fully determined by the input; two runs on the same data produce identical
  files, which the report asserts by checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

from . import features as F

MAX_NEWTON = 25
NEWTON_TOL = 1e-8


def _kernel(A, B) -> np.ndarray:
    """Linear kernel for dense or sparse feature blocks."""
    if hasattr(A, "tocsr"):
        return np.asarray((A @ B.T).todense(), dtype=np.float64)
    return np.asarray(A, dtype=np.float64) @ np.asarray(B, dtype=np.float64).T


def kernel_logistic(K: np.ndarray, y: np.ndarray, w: np.ndarray,
                    lam: float = 1.0) -> np.ndarray:
    """Kernel logistic regression by Newton's method; returns dual coefficients.

    Minimises the class-weighted logistic loss over functions `f = K a` with the
    penalty `lam/2 * a' K a`. Each Newton step solves
    `(W K + lam I) a = W z` with the working response `z` of iteratively
    reweighted least squares -- an m-by-m solve, where m is the number of
    labelled records, so a few hundred at most in screening.

    Deterministic by construction: no random initialisation, no stochastic
    optimiser, a fixed stopping rule on the parameter change, and a cap on the
    number of steps. Two runs on the same labels return the same coefficients.
    """
    m = len(y)
    y01 = (y > 0).astype(np.float64)        # the working response needs 0/1 labels
    a = np.zeros(m)
    f = np.zeros(m)
    eye = np.eye(m)

    def loss(fv, av):
        # logistic loss, written through logaddexp so large |f| cannot overflow
        return float(np.sum(w * (np.logaddexp(0.0, fv) - y01 * fv))
                     + 0.5 * lam * float(av @ (K @ av)))

    cur = loss(f, a)
    for _ in range(MAX_NEWTON):
        p = 1.0 / (1.0 + np.exp(-np.clip(f, -30.0, 30.0)))
        g = np.maximum(p * (1.0 - p), 1e-10)
        v = np.maximum(w * g, 1e-12)
        z = f + (y01 - p) / g
        a_new = np.linalg.solve(v[:, None] * K + lam * eye, v * z)
        # damped step: a full Newton step can overshoot when the labelled set is
        # small and nearly separable, which is the normal state early in a screen
        step, ok = 1.0, False
        for _ in range(8):
            cand = a + step * (a_new - a)
            fc = K @ cand
            lc = loss(fc, cand)
            if lc <= cur + 1e-12:
                a, f, cur, ok = cand, fc, lc, True
                break
            step *= 0.5
        if not ok or np.max(np.abs(step * (a_new - a))) < NEWTON_TOL:
            break
    return a


def screen(features, labels_known: dict[int, bool], n: int,
           batch: int = 25, lam: float = 1.0, intercept_var: float = 10.0,
           label_fn=None, oracle: np.ndarray | None = None
           ) -> tuple[np.ndarray, np.ndarray]:
    """Reading order under the active loop.

    `labels_known` maps record index to its decision for the records already
    screened -- the seed set. Each batch then needs its decisions from
    somewhere, and there are exactly two sources:

    * `label_fn(indices) -> decisions` -- the interactive mode: the researcher
      reads the batch and the function returns what they decided;
    * `oracle` -- a full decision array, used when replaying a completed
      screening to evaluate how much reading the ranking would have saved.

    Passing neither is a programming error rather than a mode: without the
    decisions the loop has nothing to learn from and would silently degenerate
    into ranking by the seed set alone.
    """
    if label_fn is None and oracle is None:
        raise ValueError("the loop needs decisions: pass `label_fn` "
                         "(interactive) or `oracle` (evaluation)")
    seen = np.zeros(n, dtype=bool)
    order = list(labels_known)
    seen[order] = True
    scores_at_pick = np.zeros(n)
    while seen.sum() < n:
        li = np.asarray(order)
        y = np.array([1.0 if labels_known[i] else -1.0 for i in li])
        npos, nneg = max(1, int((y > 0).sum())), max(1, int((y < 0).sum()))
        w = np.where(y > 0, 0.5 * len(y) / npos, 0.5 * len(y) / nneg)
        Xl = features[li]
        rest = np.where(~seen)[0]
        # the constant added to every kernel entry is an intercept with a wide
        # prior: it costs one line and matters a great deal, because early in a
        # screen the labelled set is small and lopsided, and without it the
        # decision threshold cannot move away from zero
        K = _kernel(Xl, Xl) + intercept_var
        K.flat[:: len(li) + 1] += 1e-8          # stabilises the Newton solve
        a = kernel_logistic(K, y, w, lam=lam)
        s = _kernel(features[rest], Xl) @ a + intercept_var * float(a.sum())
        take = rest[np.lexsort((rest, -s))[:batch]]
        scores_at_pick[take] = s[np.lexsort((rest, -s))[:batch]]
        decisions = (label_fn(take) if label_fn is not None
                     else oracle[take])
        for i, dec in zip(take, decisions):
            labels_known[int(i)] = bool(dec)
        order.extend(int(i) for i in take)
        seen[take] = True
    return np.array(order), scores_at_pick


def baselines(texts: list[str], seeds: list[int], n: int) -> dict:
    """Orders that need no embedding model at all."""
    cfg = dict(F.DEFAULT_CONFIG)
    X, idf = F.fit_transform(texts, cfg)
    q = np.asarray(X[seeds].mean(0)).ravel()
    nq = np.linalg.norm(q)
    s = np.asarray((X @ (q / (nq if nq else 1.0))[:, None])).ravel()
    idx = np.arange(n)
    lexical = idx[np.lexsort((idx, -s))]
    rnd = np.random.default_rng(0).permutation(n)
    return {"lexical (tf-idf, no model)": lexical, "random order": rnd,
            "_features": X}


def recall_curve(order: np.ndarray, included: np.ndarray) -> dict:
    hits = np.cumsum(included[order])
    total = max(1, int(included.sum()))
    out = {f"recall@{int(f*100)}%": float(hits[max(0, int(f*len(order))-1)] / total)
           for f in (0.05, 0.10, 0.20, 0.50)}
    k95 = int(np.searchsorted(hits, 0.95 * total) + 1)
    out["read_to_95%"] = float(k95 / len(order))
    out["WSS@95"] = float(1 - k95 / len(order) - 0.05)
    return out


def run(vectors: pathlib.Path, texts_path: pathlib.Path, seeds_path: pathlib.Path,
        out: pathlib.Path, labels_path: pathlib.Path | None = None,
        batch: int = 25, budget: float = 0.2,
        provenance: pathlib.Path | None = None) -> dict:
    from . import archive as A

    ids, texts = A.read_texts(texts_path)
    Y = np.load(vectors).astype(np.float64)
    n = len(texts)
    if len(Y) != n:
        raise SystemExit(f"{n} texts but {len(Y)} vectors")
    seed_ids = [l.strip() for l in seeds_path.read_text(encoding="utf-8").splitlines()
                if l.strip()]
    pos = {i for i, x in enumerate(ids) if x in set(seed_ids)}
    if not pos:
        raise SystemExit("none of the seed identifiers occur in the corpus")

    labels = None
    if labels_path is not None:
        import csv
        dec = {}
        with labels_path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = row.get("id") or row.get("key")
                val = (row.get("decision") or row.get("include") or "").strip().lower()
                dec[key] = val in ("1", "true", "include", "included", "yes")
        labels = np.array([dec.get(x, False) for x in ids])

    # seeds: the known included records, plus the same number of records whose
    # decision is known to be negative when labels are available
    known = {i: True for i in sorted(pos)}
    if labels is not None:
        neg = [i for i in range(n) if not labels[i] and i not in pos][:len(pos)]
        known.update({i: False for i in neg})

    if labels is None:
        raise SystemExit(
            "no decisions given. `select` needs either --labels (replaying a "
            "finished screening, to measure how much reading the order saves) "
            "or interactive use through embsurrogate.select.screen(label_fn=...)")
    order, _ = screen(Y, dict(known), n, batch=batch, oracle=labels)
    base = baselines(texts, sorted(pos), n)

    out.mkdir(parents=True, exist_ok=True)
    import csv
    with (out / "reading_order.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "id", "seed"])
        for r, i in enumerate(order, 1):
            w.writerow([r, ids[i], int(i in known)])

    report = {
        "records": n, "seeds": len(pos), "batch": batch,
        "budget": budget,
        "order_sha256": hashlib.sha256(
            ",".join(ids[i] for i in order).encode("utf-8")).hexdigest(),
        "classifier": "kernel logistic regression, Newton steps (deterministic)",
    }
    if labels is not None:
        report["evaluation"] = {
            "surrogate or model vectors": recall_curve(order, labels),
            **{k: recall_curve(v, labels) for k, v in base.items()
               if not k.startswith("_")}}
        report["positives"] = int(labels.sum())
    if provenance is not None and provenance.exists():
        src = [r.split(",")[1].strip() for r in
               provenance.read_text(encoding="utf-8").splitlines()[1:]]
        report["vector_provenance"] = {"exact": src.count("exact"),
                                       "surrogate": src.count("surrogate")}
    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="embsurrogate select")
    ap.add_argument("--vectors", type=pathlib.Path, required=True)
    ap.add_argument("--texts", type=pathlib.Path, required=True)
    ap.add_argument("--seeds", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--labels", type=pathlib.Path, default=None,
                    help="known decisions; turns the run into an evaluation")
    ap.add_argument("--provenance", type=pathlib.Path, default=None,
                    help="provenance file from replay, folded into the report")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--budget", type=float, default=0.2)
    a = ap.parse_args(argv)
    rep = run(a.vectors, a.texts, a.seeds, a.out, a.labels, a.batch, a.budget,
              a.provenance)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
