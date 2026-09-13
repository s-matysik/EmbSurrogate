"""End-to-end: fit a surrogate from pairs, evaluate it, freeze it, replay it.

    python3 examples/fit_from_pairs.py            # writes to ./tmp_example/

Needs scipy (fitting) but no teacher model and no network: the "teacher" here is
a fixed random projection of a bag-of-characters representation, which is enough
to exercise the whole pipeline. The fidelity numbers printed by this script are
therefore about a synthetic teacher and say nothing about the real one — see the
README for measured fidelity against a sentence encoder.

The point of the example is the *shape* of a real run:

    pairs -> Surrogate.fit -> evaluate.report -> freeze.build_explicit
          -> FrozenSurrogate (numpy only) -> replay_check.check
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

import numpy as np

from embsurrogate import evaluate as E
from embsurrogate import features as F
from embsurrogate import freeze as FZ
from embsurrogate.predict import FrozenSurrogate
from embsurrogate.replay_check import check
from embsurrogate.surrogate import Surrogate

WORDS = ("umowa najem lokal mieszkanie sprzedaż faktura podatek rejestr sąd "
         "wyrok kredyt bank spółka udział akcja zebranie protokół uchwała "
         "kernel regression vector embedding archive checksum neighbour "
         "residual determinism quantisation corpus token").split()


def synthetic_pairs(n: int, dim: int = 64, seed: int = 0):
    """Texts from a small vocabulary, 'teacher' = mean of random word vectors.

    The teacher must be a function that the surrogate can plausibly learn AND
    one that actually varies between texts — otherwise every fidelity number
    degenerates. A bag-of-words embedding satisfies both: texts sharing words
    are similar, texts sharing none are not, and words are visible to a
    character-n-gram feature map.
    """
    rng = np.random.default_rng(seed)
    # Each text draws a few words from one of a handful of topics, so the
    # similarity structure is neither constant nor pure noise.
    topics = [WORDS[i::4] for i in range(4)]
    texts, Y = [], np.zeros((n, dim))
    word_vec = {w: v for w, v in zip(WORDS, rng.standard_normal((len(WORDS), dim)))}
    for i in range(n):
        topic = topics[i % len(topics)]
        k_topic = int(rng.integers(3, 7))
        k_other = int(rng.integers(1, 4))
        words = list(rng.choice(topic, size=k_topic, replace=False)) + \
            list(rng.choice(WORDS, size=k_other, replace=False))
        rng.shuffle(words)
        texts.append(" ".join(words))
        Y[i] = np.mean([word_vec[w] for w in words], axis=0)
    Y += 0.02 * rng.standard_normal(Y.shape)       # a little irreducible noise
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    ids = [f"syn-{i:05d}" for i in range(n)]
    return texts, Y, ids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="training pairs")
    ap.add_argument("--n-test", type=int, default=120)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("tmp_example"))
    args = ap.parse_args(argv)

    texts, Y, ids = synthetic_pairs(args.n + args.n_test)
    tr = slice(0, args.n)
    te = slice(args.n, args.n + args.n_test)

    # --- fit ---------------------------------------------------------------
    sur = Surrogate("hybrid_z", k=20, lam=1e-4, degree=0)
    sur.fit(texts[tr], Y[tr], ids=ids[tr])
    Yhat = sur.predict(texts[te])

    rep = E.report(Y[te], Yhat)
    with warnings.catch_warnings():
        # The floor returns the same vector for every text, so its
        # similarity matrix is constant and rho is undefined by construction.
        warnings.simplefilter("ignore")
        floor = E.report(Y[te], E.baseline_mean(Y[tr], args.n_test))
    print(f"pairs      : {args.n} train / {args.n_test} test, dim {Y.shape[1]}")
    print(f"surrogate  : cos median {rep['cos_mediana']:.4f}  "
          f"rho {rep['spearman_sim']:.4f}  recall@10 {rep['recall@10']:.4f}")
    print(f"floor      : cos median {floor['cos_mediana']:.4f}  "
          f"rho n/a (constant predictor)   (mean training vector)")

    # --- freeze ------------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / "surrogate_example.npz"
    manifest = FZ.build_explicit(
        texts[tr], ids[tr], Y[tr], dict(F.DEFAULT_CONFIG),
        "hybrid_z", {"lam": 1e-4, "k": 20, "degree": 0},
        {"idf": sur.idf, "alpha": sur.state["alpha"], "R": sur.state["R"],
         "z": sur.state["z"]},
        model_path, meta_extra={"dataset": "synthetic (examples/fit_from_pairs.py)"})
    print(f"\nfrozen     : {model_path.name} "
          f"({model_path.stat().st_size / 1e6:.1f} MB) "
          f"sha256 {manifest['sha256_pliku'][:16]}...")

    # --- reload without scipy and compare ----------------------------------
    frozen = FrozenSurrogate(model_path)
    Yfrozen = frozen.encode(texts[te], decimals=None)
    print(f"reload     : max |scipy path - numpy path| = "
          f"{np.abs(Yhat - Yfrozen).max():.2e}")

    # --- golden file + full replay check -----------------------------------
    expected = args.out / "expected_example.npy"
    np.save(expected, frozen.encode(texts[te]))
    ok, log = check(model_path, list(texts[te]), expected, decimals=6)
    print("\n".join(log))
    print("\nRESULT:", "REPRODUCIBLE" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
