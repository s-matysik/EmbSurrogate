"""Read a frozen surrogate, embed new texts, query the archive, verify replay.

    python3 examples/quickstart.py

Uses the small model that ships with the tests, so it needs nothing but numpy
and the repository itself. Everything here is the archival path: no scipy, no
network, no teacher model.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from embsurrogate.predict import FrozenSurrogate

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "tests" / "data"
MODEL = DATA / "surrogate_tiny.npz"

NEW_TEXTS = [
    "umowa najmu lokalu użytkowego zawarta na czas oznaczony",
    "kernel ridge regression in dual form with a tricube correction",
    "Prva liga Jugoslavije — tabela końcowa sezonu",
]


def main() -> int:
    model = FrozenSurrogate(MODEL)
    print(f"model      : {MODEL.name}")
    print(f"method     : {model.spec['method']}  params={model.spec['params']}")
    print(f"training   : {model.spec['n_train']} pairs, dim {model.spec['dim']}")
    print(f"features   : {model.spec['feature_version']} "
          f"(fingerprint {model.spec['cfg_fingerprint']})")

    # 1. embed new, unseen texts -------------------------------------------
    V = model.encode(NEW_TEXTS)                     # quantised to 6 decimals
    print(f"\nencoded    : {V.shape}, norms {np.round(np.linalg.norm(V, axis=1), 6)}")
    print(f"first 6 dims of #1: {V[0, :6]}")

    # 2. the same call twice must give identical bits -----------------------
    assert np.array_equal(V, model.encode(NEW_TEXTS))
    print("replay     : identical within the process")

    # 3. query the archive with a surrogate vector --------------------------
    # Teacher vectors of the archived texts live in the model file; a query
    # embedded by the surrogate can be scored against them directly.
    Ytr = np.asarray(model.state["Y"])
    q = model.encode(["lease of commercial premises for a fixed term"])[0]
    sims = Ytr @ q
    top = np.argsort(-sims)[:3]
    print("\nnearest archived texts:")
    for rank, i in enumerate(top, 1):
        print(f"  {rank}. id={model.ids[i]:>12s} cos={sims[i]:+.4f} "
              f"| {model.texts[i][:70]}...")

    # 4. for an ARCHIVED text, prefer the stored teacher vector -------------
    # encode() always recomputes a surrogate vector, which throws away exact
    # information; the archive already holds the teacher's own answer.
    i = 0
    surrogate_i = model.encode([model.texts[i]], decimals=None)[0]
    cos = float(surrogate_i @ Ytr[i])
    print(f"\narchived text #{i}: cos(surrogate, stored teacher vector) = {cos:.4f}")
    print("   -> look the vector up by id instead of re-encoding it")

    # 5. golden file shipped with the tests ---------------------------------
    with (DATA / "probes_tiny.jsonl").open(encoding="utf-8") as f:
        probes = [json.loads(line)["text"] for line in f if line.strip()]
    expected = np.load(DATA / "expected_tiny.npy")
    same = np.array_equal(model.encode(probes), expected)
    print(f"\ngolden check: {'identical' if same else 'MISMATCH'} "
          f"on {len(probes)} control texts")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
