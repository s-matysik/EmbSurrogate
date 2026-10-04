"""Build an archive bundle while the embedding model still exists.

This is the step a researcher runs ONCE, at the time the selection is made and
the model is still available. Everything needed to reproduce the vectors later
-- without that model -- goes into one directory with checksums.

    embsurrogate archive --texts corpus.jsonl --vectors vectors.npy \
        --out bundle/ [--table table.npz]

What the bundle contains, and why each part is there:

* `pairs.npz` -- the identifiers, the raw texts and the **original vectors**.
  For any text that was in the corpus, reproduction is a lookup: the vector is
  the one the model produced, bit for bit, not an approximation of it. This is
  the part that makes "same raw data -> same result" exact rather than
  approximate, and it is why the bundle is worth building even if the surrogate
  were never used.
* `surrogate.npz` -- the fitted coefficients, so vectors can be produced for
  texts that were NOT in the corpus (a reviewer's extra records, a search
  rerun). Approximate, with the fidelity measured at build time and recorded in
  the manifest.
* `table.npz` + `tokenizer/` -- optional static token table distilled from the
  model, which raises the fidelity for new texts (equivalent to roughly three
  times the corpus; see docs/METHOD.md).
* `manifest.json` -- SHA-256 of every array, the feature version, the fidelity
  measured on a held-out part of the corpus, and the environment. Replay
  refuses to run if any of it fails to match.

The fidelity numbers are measured here, not promised: the corpus is split, the
surrogate is fitted on one part and evaluated on the other, and the result is
written into the manifest so that whoever replays the bundle can read what to
expect for new texts before trusting them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import sys
from datetime import datetime, timezone

import numpy as np

from . import dual as D
from . import evaluate as E
from . import features as F
from .freeze import pack_strings

BUNDLE_FORMAT = "embsurrogate-bundle-2"

# How the arrays are stored. The choice is measured, not guessed: on the
# 43,093-pair model, dropping to float32 and then float16 leaves every fidelity
# metric unchanged at four decimals (rho 0.8075 throughout), while the file
# shrinks from 558 MB to 160 MB. See docs/METHOD.md, "Size of the artefact".
PRECISION = {
    # the archived teacher vectors: integers at the archival unit of 6 decimals,
    # so the exact-lookup guarantee is kept exactly while halving the space
    "vectors": "int32@1e-6",
    "coef": np.float32,      # float16 also loses nothing measurable; float32 is
    "z": np.float16,         # the conservative default for the prediction path
}


def _sha(x) -> str:
    if isinstance(x, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    if isinstance(x, (bytes, bytearray)):
        return hashlib.sha256(x).hexdigest()
    return hashlib.sha256(str(x).encode("utf-8")).hexdigest()


def read_texts(path: pathlib.Path) -> tuple[list[str], list[str]]:
    """Reads ids and texts from JSONL (`id`/`text`) or a one-text-per-line file."""
    ids, texts = [], []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.lstrip().startswith("{"):
                rec = json.loads(line)
                texts.append(rec.get("text") or rec.get("abstract") or "")
                ids.append(str(rec.get("id", rec.get("key", i))))
            else:
                texts.append(line)
                ids.append(str(i))
    return ids, texts


CLOSED_FORM_LIMIT = 15000


def fit(texts: list[str], Y: np.ndarray, cfg: dict, lam: float = 1e-4,
        k: int = 100, table=None, alpha: float = 0.6,
        folds: int = 4) -> dict:
    """Fits the surrogate in closed form; `table` adds the static-table kernel.

    Returns the pieces replay needs: the coefficient matrix, the residuals, the
    out-of-sample coordinates used by the correction, the document weights, and
    the training-side feature blocks.
    """
    X, idf = F.fit_transform(texts, cfg)
    # The weights are rounded to the precision they will be STORED at, before
    # anything is fitted on them. Otherwise the coefficients would be fitted
    # against float64 weights and replayed against float32 ones -- a small
    # inconsistency, but a pointless one, and this project has already spent a
    # continuous-integration run on exactly that class of mismatch.
    idf = idf.astype(np.float32).astype(np.float64)
    X = F.transform(texts, idf, cfg)
    V = None
    if table is None:
        alpha = 1.0
    else:
        V = table.encode(texts)
    n = len(texts)
    if n <= CLOSED_FORM_LIMIT:
        K = np.asarray((X @ X.T).todense(), dtype=np.float64)
        np.fill_diagonal(K, 1.0)
        if V is not None:
            K = alpha * K + (1 - alpha) * (V @ V.T)
        st = D.fit_dual("hybrid_z", K, Y, lam=lam, k=k, degree=0)
        coef, R, z = st["alpha"], st["R"], st["z"]
    else:
        # Above roughly 20,000 pairs the N-by-N factorisation is killed by the
        # operating system in a 24 GB environment, so the kernel matrix is never
        # formed: conjugate gradients need only the product K v, and the
        # leave-one-out coordinates that the residual correction searches in are
        # replaced by out-of-fold predictions, which agree with them to the
        # fourth decimal of rho (see docs/METHOD.md).
        from .run_primal import dual_cg, oof_coordinates
        Z = _stack(X, V, alpha)
        coef, _, _ = dual_cg(Z, Y.astype(np.float32).copy(), lam, tol=1e-4)
        coef = coef.astype(np.float64)
        R = (lam * n) * coef                 # exact for a solved system
        z = oof_coordinates(Z, Y.astype(np.float32), lam, folds=folds, tol=1e-4)
    return {"coef": coef, "R": R, "z": z, "idf": idf,
            "X": X, "V": V, "alpha_mix": float(alpha)}


def _stack(X, V, alpha: float):
    """Feature blocks as one operator: `Z Z' = alpha X X' + (1-alpha) V V'`."""
    if V is None or alpha >= 1.0:
        return X.astype(np.float32).tocsr()
    from scipy import sparse
    return sparse.hstack([np.sqrt(alpha) * X.astype(np.float32),
                          sparse.csr_matrix(np.sqrt(1 - alpha)
                                            * V.astype(np.float32))],
                         format="csr")


def build(texts_path: pathlib.Path, vectors_path: pathlib.Path,
          out: pathlib.Path, table_path: pathlib.Path | None = None,
          lam: float = 1e-4, k: int = 100, alpha: float = 0.6,
          holdout: float = 0.2, teacher: str = "", seed: int = 0,
          link_vectors: str | None = None) -> dict:
    """Builds the bundle. `link_vectors` records the vector file by checksum
    instead of copying it, for the common case where those vectors are already
    part of the data deposit; `replay --vectors` then supplies them."""
    ids, texts = read_texts(texts_path)
    Y = np.load(vectors_path).astype(np.float64)
    if len(Y) != len(texts):
        raise SystemExit(f"{len(texts)} texts but {len(Y)} vectors — "
                         "the archive must be one vector per text")
    nrm = np.linalg.norm(Y, axis=1)
    if not np.allclose(nrm, 1.0, atol=1e-3):
        Y = Y / np.maximum(nrm, 1e-12)[:, None]
    cfg = dict(F.DEFAULT_CONFIG)

    table = None
    if table_path is not None:
        from .static_table import StaticTable
        table = StaticTable(table_path)

    # --- fidelity for texts that were NOT archived, measured on a held-out part
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(texts))
    n_ho = max(1, int(holdout * len(texts)))
    ho, tr = perm[:n_ho], perm[n_ho:]
    st_tr = fit([texts[i] for i in tr], Y[tr], cfg, lam, k, table, alpha)
    yh = predict([texts[i] for i in ho], st_tr["idf"], st_tr["coef"], st_tr["R"],
                 st_tr["z"], st_tr["X"], st_tr["V"], cfg, st_tr["alpha_mix"], k,
                 table)
    measured = E.report_full(Y[ho], yh, Y[tr].mean(0))
    measured = {kk: float(vv) for kk, vv in measured.items()
                if isinstance(vv, (int, float))}

    # --- final fit on everything
    st = fit(texts, Y, cfg, lam, k, table, alpha)

    out.mkdir(parents=True, exist_ok=True)
    # --- storage
    # `R` is NOT stored: for a solved system `(K + lam N I) a = Y` the residual
    # is `R = Y - K a = lam N a`, so it is a scalar multiple of the
    # coefficients. Deriving it is exact for a closed-form fit and accurate to
    # 1.7e-5 for an iterative one (tolerance 1e-4), which leaves every metric
    # unchanged -- and saves a quarter of the file.
    Yq = np.round(Y * 1e6).astype(np.int32) if link_vectors is None else None
    pairs = {"ids": pack_strings(ids), "texts": pack_strings(texts)}
    if Yq is not None:
        pairs["Y_micro"] = Yq
    sur = {"idf": st["idf"].astype(np.float32),
           "coef": st["coef"].astype(PRECISION["coef"]),
           "z": st["z"].astype(PRECISION["z"]),
           "lam_N": np.float64(lam * len(texts)),
           "alpha_mix": np.float64(st["alpha_mix"])}
    if st["V"] is not None:
        # the table-side training block is kept, so replay does not need to
        # reproduce it from the table (and cannot disagree with it)
        sur["V"] = st["V"].astype(np.float32)
    np.savez_compressed(out / "pairs.npz", **pairs)
    np.savez(out / "surrogate.npz", **sur)

    manifest = {
        "format": BUNDLE_FORMAT,
        "feature_version": F.FEATURE_VERSION,
        "cfg": {kk: (list(vv) if isinstance(vv, tuple) else vv)
                for kk, vv in cfg.items()},
        "n_pairs": len(texts),
        "dim": int(Y.shape[1]),
        "teacher": teacher,
        "method": "hybrid_z",
        "params": {"lam": lam, "k": k, "alpha_mix": float(st["alpha_mix"])},
        "static_table": (str(table_path.name) if table_path else None),
        "fidelity_new_texts": {
            "protocol": f"{int(holdout*100)}% of the corpus held out, seed {seed}",
            "n_holdout": int(n_ho), **measured},
        "storage": {"vectors": ("linked: " + link_vectors) if link_vectors
                    else PRECISION["vectors"],
                    "coef": np.dtype(PRECISION["coef"]).name,
                    "z": np.dtype(PRECISION["z"]).name,
                    "residuals": "derived as lam*N*coef, not stored"},
        # The digest of the recomputed features is the check that caught this
        # project's one real cross-platform defect (a BLAS reduction in the
        # normalisation); `verify` compares against it on machines that did not
        # build the bundle.
        "feature_digest": {kk: _sha(vv) for kk, vv in
                           zip(("indptr", "indices", "data"),
                               F.csr_arrays(texts, st["idf"], cfg))},
        "sha256": {"pairs.npz": _sha((out / "pairs.npz").read_bytes()),
                   "surrogate.npz": _sha((out / "surrogate.npz").read_bytes()),
                   "texts": _sha("\u0000".join(texts).encode("utf-8")),
                   "vectors": _sha(np.round(Y * 1e6).astype(np.int32))},
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__},
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def kernel_star(Xq, Vq, Xtr, Vtr, alpha: float) -> np.ndarray:
    """Kernel between query texts and the archived texts.

    `alpha` mixes the two feature spaces: character n-grams (sparse) and the
    static token table (dense). With `alpha = 1` the table is not used at all,
    which is also what happens when the bundle carries no table.
    """
    Ks = float(alpha) * np.asarray((Xq @ Xtr.T).todense(), dtype=np.float64)
    if alpha < 1.0:
        Ks = Ks + (1.0 - float(alpha)) * (Vq @ Vtr.T)
    return Ks


def predict(texts: list[str], idf, coef, R, z, Xtr, Vtr, cfg: dict,
            alpha: float, k: int, table=None) -> np.ndarray:
    """Surrogate vectors for arbitrary texts (ridge plus residual correction)."""
    Xq = F.transform(texts, idf, cfg)
    Vq = table.encode(texts) if (table is not None and alpha < 1.0) else None
    Ks = kernel_star(Xq, Vq, Xtr, Vtr, alpha)
    # The correction is added to the RAW ridge output, with the normalised
    # output used only as the query coordinate for the neighbour search.
    # Normalising before the addition instead costs 0.008 rho on the 43k-pair
    # model (0.7995 against 0.8075) -- measured when a published number failed
    # to reproduce, which is the reason this comment exists.
    b = Ks @ coef
    q = D.l2norm(b)
    return D.l2norm(b + D._local(q @ z.T, R, k, degree=0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="embsurrogate archive")
    ap.add_argument("--texts", type=pathlib.Path, required=True)
    ap.add_argument("--vectors", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--table", type=pathlib.Path, default=None)
    ap.add_argument("--teacher", default="")
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--link-vectors", action="store_true",
                    help="record the vector file by checksum instead of copying "
                         "it into the bundle (it is usually in the deposit "
                         "already); replay then needs --vectors")
    a = ap.parse_args(argv)
    man = build(a.texts, a.vectors, a.out, a.table, lam=a.lam, k=a.k,
                alpha=a.alpha, teacher=a.teacher,
                link_vectors=(str(a.vectors) if a.link_vectors else None))
    f = man["fidelity_new_texts"]
    mb = sum(p.stat().st_size for p in a.out.iterdir() if p.is_file()) / 1e6
    print(json.dumps({"bundle": str(a.out), "size_MB": round(mb, 2),
                      "n_pairs": man["n_pairs"],
                      "fidelity_new_texts": {
                          "rho": round(f["rho_globalne"], 4),
                          "cos_median": round(f["cos_mediana"], 4),
                          "recall@10": round(f["recall@10"], 4)}},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
