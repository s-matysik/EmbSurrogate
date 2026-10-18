"""Freezing the surrogate to a single file + manifest with checksums.

The frozen model contains EVERYTHING needed to reconstruct the vectors:
training texts, teacher vectors, the idf table, dual coefficients
and feature configuration. There are no references to the network, to HuggingFace or to the
teacher model, after freezing the original may disappear.

    python3 -m klon.freeze data/ds_multi wyniki/multi/najlepszy.json model/surogat.npz

The manifest (`surogat.manifest.json`) records the sha256 of each array separately
as well as the sha256 of the entire file, so corruption or substitution is visible immediately."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import sys
from datetime import datetime, timezone

import numpy as np

from . import features as F
from . import surrogate as S


def _sha(obj) -> str:
    if isinstance(obj, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(obj).tobytes()).hexdigest()
    if isinstance(obj, (bytes, bytearray)):
        return hashlib.sha256(obj).hexdigest()
    return hashlib.sha256(str(obj).encode("utf-8")).hexdigest()


def pack_strings(items) -> np.ndarray:
    """List of strings -> single uint8 array (separated by \\x00)."""
    return np.frombuffer("\x00".join(items).encode("utf-8"), dtype=np.uint8)


def unpack_strings(arr: np.ndarray) -> list[str]:
    return arr.tobytes().decode("utf-8").split("\x00")


def build(dataset: pathlib.Path, best: dict, out: pathlib.Path,
          *, limit: int = 0, seed: int = 0, test_frac: float = 0.2,
          with_features: bool = False) -> dict:
    from .run_experiment import load_dataset, split

    texts, Y, meta = load_dataset(dataset, limit)
    tr, _ = split(len(texts), test_frac, seed)
    cfg = best.get("cfg") or dict(F.DEFAULT_CONFIG)
    method, params = best["metoda"], best["parametry"]

    texts_tr = [texts[i] for i in tr]
    ids_tr = [meta[i]["id"] for i in tr]
    Ytr = Y[tr]

    indptr, indices, data = F.csr_arrays(texts_tr, np.ones(cfg["n_buckets"]), cfg)
    # idf is computed on the same texts as training
    from scipy import sparse
    tf = F.tf_matrix(texts_tr, cfg)
    idf = F.fit_idf(tf, cfg)
    X = F.apply_idf(tf, idf)
    K = np.asarray((X @ X.T).todense(), dtype=np.float64)
    np.fill_diagonal(K, 1.0)
    st = S.fit_dual(method, K, Ytr, **params)

    indptr, indices, data = F.csr_arrays(texts_tr, idf, cfg)

    payload = {
        "idf": idf.astype(np.float64),
        "Y": Ytr.astype(np.float64),
        "texts": pack_strings(texts_tr),
        "ids": pack_strings(ids_tr),
    }
    if with_features:
        # The feature matrix is ~110 MB for 5 thousand texts, and it can be
        # reconstructed from the texts in tens of seconds. By default we only save
        # its checksums, this is sufficient to detect a mismatch in the feature code
        # version.
        payload.update(indptr=indptr, indices=indices, data=data)
    for key in ("alpha", "R", "U", "lam_eig", "z"):
        if key in st:
            payload[key] = np.ascontiguousarray(st[key], dtype=np.float64)

    spec = {
        "format": "klon-surogat-2",
        "features_embedded": bool(with_features),
        "sha256_cech": {"indptr": _sha(indptr), "indices": _sha(indices),
                        "data": _sha(data)},
        "feature_version": F.FEATURE_VERSION,
        "cfg": cfg,
        "cfg_fingerprint": F.config_fingerprint(cfg),
        "method": method,
        "params": params,
        "n_train": int(len(texts_tr)),
        "dim": int(Ytr.shape[1]),
        "dataset": str(dataset),
        "seed": seed, "test_frac": test_frac,
    }
    payload["spec"] = np.frombuffer(
        json.dumps(spec, ensure_ascii=False, sort_keys=True).encode("utf-8"),
        dtype=np.uint8)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)          # without compression: deterministic bytes

    manifest = {
        "plik": out.name,
        "sha256_pliku": _sha(out.read_bytes()),
        "sha256_tablic": {k: _sha(v) for k, v in sorted(payload.items())},
        "spec": spec,
        "zapisano_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "srodowisko_zapisu": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": sparse.__name__ and __import__("scipy").__version__,
            "platforma": platform.platform(),
        },
        "uwaga": ("Bit-for-bit agreement is guaranteed for vectors "
                  "quantized to 6 decimal places (see replay_check.py). "
                  "Raw float64 may differ by ~1e-12 between BLAS implementations."),
    }
    mpath = out.with_suffix(".manifest.json")
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def build_explicit(texts_tr, ids_tr, Ytr, cfg, method, params, state, out,
                   meta_extra=None, with_features=False) -> dict:
    """Freezing from an explicitly given state, without repeating the split from the file.

    `build` reconstructs the train/test split from the seed, which is convenient, but does not
    support the three-way layout (pool / validation / fresh test T0) used
    in round 2. Here the model state has already been computed and we are only serializing it.
    """
    indptr, indices, data = F.csr_arrays(texts_tr, state["idf"], cfg)
    payload = {
        "idf": state["idf"].astype(np.float64),
        "Y": np.ascontiguousarray(Ytr, dtype=np.float64),
        "texts": pack_strings(list(texts_tr)),
        "ids": pack_strings(list(ids_tr)),
    }
    if with_features:
        payload.update(indptr=indptr, indices=indices, data=data)
    for key in ("alpha", "R", "U", "lam_eig", "z"):
        if key in state and state[key] is not None:
            payload[key] = np.ascontiguousarray(state[key], dtype=np.float64)

    spec = {
        "format": "klon-surogat-2",
        "features_embedded": bool(with_features),
        "sha256_cech": {"indptr": _sha(indptr), "indices": _sha(indices),
                        "data": _sha(data)},
        "feature_version": F.FEATURE_VERSION,
        "cfg": cfg,
        "cfg_fingerprint": F.config_fingerprint(cfg),
        "method": method,
        "params": params,
        "n_train": int(len(texts_tr)),
        "dim": int(Ytr.shape[1]),
        **(meta_extra or {}),
    }
    payload["spec"] = np.frombuffer(
        json.dumps(spec, ensure_ascii=False, sort_keys=True).encode("utf-8"),
        dtype=np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    manifest = {
        "plik": out.name,
        "sha256_pliku": _sha(out.read_bytes()),
        "sha256_tablic": {k: _sha(v) for k, v in payload.items()},
        "spec": spec,
        "zbudowano": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "srodowisko": {"python": platform.python_version(),
                       "numpy": np.__version__},
        "uwaga": ("Bit-for-bit agreement for vectors quantized to 6 decimal "
                  "places; raw float64 may differ by ~1e-12 "
                  "between BLAS implementations."),
    }
    out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=pathlib.Path)
    ap.add_argument("best_json", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--with-features", action="store_true",
                    help="embed the feature matrix in the file (+110 MB, faster loading)")
    args = ap.parse_args(argv)

    best = json.loads(args.best_json.read_text(encoding="utf-8"))
    m = build(args.dataset, best, args.out, limit=args.limit,
              seed=best.get("seed", 0), with_features=args.with_features)
    print(json.dumps({"plik": m["plik"], "sha256": m["sha256_pliku"],
                      "metoda": m["spec"]["method"], "n_train": m["spec"]["n_train"]},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
