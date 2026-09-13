"""Build the small frozen surrogate that the test suite and CI run against.

    python3 examples/build_tiny_fixture.py data/ds_multi --n 900 --probes 20

The fixture is the load-bearing part of the reproducibility claim: it is a
complete frozen model, small enough to live in the repository, so continuous
integration can check on a DIFFERENT operating system and a different numpy
build that the archived vectors come out identical to the ones stored here.

What is written (all under tests/data/):

    surrogate_tiny.npz            frozen model (method hybrid_z, numpy-only inference)
    surrogate_tiny.manifest.json  sha256 of the file and of every array
    probes_tiny.jsonl             control texts, held out of the training pool
    expected_tiny.npy             vectors for those texts, quantized to 6 decimals

The training pool is the first --n pairs of the dataset in file order and the
control texts are the pairs immediately after it, so the selection carries no
random seed at all: rerunning the script on the same dataset reproduces the
same fixture. The dataset itself is not part of the repository (it is teacher
output, not code); --n defaults to the value that keeps the .npz below 15 MB
for a 384-dimensional teacher, because four N x d float64 arrays (Y, alpha, R,
z) plus the idf table dominate the file size.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

from embsurrogate import dual as D
from embsurrogate import features as F
from embsurrogate import freeze as FZ
from embsurrogate.predict import FrozenSurrogate
from embsurrogate.run_experiment import load_dataset

LAM = 1e-4      # same regularisation as the published full-size model
K_NEIGHBOURS = 20   # neighbours of the residual correction at small N


def build(dataset: pathlib.Path, out_dir: pathlib.Path, n_train: int,
          n_probes: int, decimals: int = 6) -> dict:
    texts, Y, meta = load_dataset(dataset, limit=n_train + n_probes)
    if len(texts) < n_train + n_probes:
        raise SystemExit(f"dataset has {len(texts)} pairs, need {n_train + n_probes}")
    pool = list(range(n_train))
    probe_idx = list(range(n_train, n_train + n_probes))

    cfg = dict(F.DEFAULT_CONFIG)
    texts_tr = [texts[i] for i in pool]
    Ytr = Y[pool]

    X, idf = F.fit_transform(texts_tr, cfg)
    K = np.asarray((X @ X.T).todense(), dtype=np.float64)
    np.fill_diagonal(K, 1.0)
    st = D.fit_dual("hybrid_z", K, Ytr, lam=LAM, k=K_NEIGHBOURS, degree=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "surrogate_tiny.npz"
    manifest = FZ.build_explicit(
        texts_tr, [str(meta[i]["id"]) for i in pool], Ytr, cfg,
        "hybrid_z", {"lam": LAM, "k": K_NEIGHBOURS, "degree": 0},
        {"idf": idf, "alpha": st["alpha"], "R": st["R"], "z": st["z"]},
        model_path,
        meta_extra={"dataset": str(dataset),
                    "uklad": f"first {n_train} pairs in file order = training pool; "
                             f"next {n_probes} = control texts",
                    "solver": "float64 Cholesky (training), float64 inference",
                    "purpose": "test fixture: cross-platform bit-for-bit check"})

    probes_path = out_dir / "probes_tiny.jsonl"
    with probes_path.open("w", encoding="utf-8") as f:
        for i in probe_idx:
            f.write(json.dumps({"id": str(meta[i]["id"]), "text": texts[i]},
                               ensure_ascii=False) + "\n")

    model = FrozenSurrogate(model_path)
    V = model.encode([texts[i] for i in probe_idx], decimals=decimals)
    expected_path = out_dir / "expected_tiny.npy"
    np.save(expected_path, V)

    info = {"model": str(model_path), "size_bytes": model_path.stat().st_size,
            "n_train": n_train, "n_probes": n_probes, "dim": int(Ytr.shape[1]),
            "decimals": decimals, "sha256_model": manifest["sha256_pliku"],
            "spec_method": manifest["spec"]["method"]}
    print(json.dumps(info, indent=2))
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=pathlib.Path,
                    help="directory with meta.jsonl + vectors.npy (teacher output)")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("tests/data"))
    ap.add_argument("--n", type=int, default=900, help="training pairs in the fixture")
    ap.add_argument("--probes", type=int, default=20, help="held-out control texts")
    ap.add_argument("--decimals", type=int, default=6)
    args = ap.parse_args(argv)
    build(args.dataset, args.out, args.n, args.probes, args.decimals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
