"""Archival and consistent mode must differ, and differ in the right direction.

The default returns the stored vector for an archived text; that is what makes
a finished selection reproducible exactly. It is also the wrong choice when
archived and new records are ranked against each other, because ridge shrinks
surrogate vectors towards the corpus mean and a centroid query then favours
them. These tests pin the behaviour of both modes and the warning that tells a
user which one they are in.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from embsurrogate import archive as A
from embsurrogate import replay as RP
from embsurrogate.freeze import unpack_strings

DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    z = np.load(DATA / "surrogate_tiny.npz", allow_pickle=False)
    texts = unpack_strings(z["texts"])
    Y = z["Y"] if "Y" in z.files else z["Y_micro"] / 1e6
    out = tmp_path_factory.mktemp("b")
    A.build_from_arrays(texts, [str(i) for i in range(len(texts))],
                        np.asarray(Y, dtype=np.float64), out / "bundle",
                        k=20) if hasattr(A, "build_from_arrays") else None
    if not (out / "bundle").exists():
        corpus = out / "corpus.jsonl"
        with corpus.open("w", encoding="utf-8") as fh:
            for i, t in enumerate(texts):
                fh.write(json.dumps({"id": str(i), "text": t}) + "\n")
        np.save(out / "vectors.npy", np.asarray(Y, dtype=np.float64))
        A.build(corpus, out / "vectors.npy", out / "bundle", k=20)
    return out / "bundle"


def test_archival_returns_stored_vectors_and_consistent_does_not(bundle):
    b = RP.Bundle(bundle)
    texts = list(b.texts[:40])
    exact, prov_a = b.vectors_for(texts, mode="archival")
    approx, prov_c = b.vectors_for(texts, mode="consistent")
    assert set(prov_a) == {"exact"}
    assert set(prov_c) == {"surrogate (out-of-sample)"}
    assert np.allclose(exact, b.Y[:40])
    # out-of-sample coordinates are predictions, so they must NOT be the stored
    # vectors; anything closer than this would mean the fit saw its own answer
    cos = (exact * approx).sum(1)
    assert cos.max() < 0.999, cos.max()


def test_consistent_mode_shrinks_towards_the_corpus_centroid(bundle):
    b = RP.Bundle(bundle)
    texts = list(b.texts[:120])
    exact, _ = b.vectors_for(texts, mode="archival")
    approx, _ = b.vectors_for(texts, mode="consistent")
    cen = b.Y.mean(0)
    cen = cen / np.linalg.norm(cen)
    assert float((approx @ cen).mean()) > float((exact @ cen).mean())


def test_mixed_run_is_flagged(bundle, tmp_path):
    corpus = tmp_path / "mixed.jsonl"
    with corpus.open("w", encoding="utf-8") as fh:
        for i, t in enumerate(list(RP.Bundle(bundle).texts[:10])):
            fh.write(json.dumps({"id": str(i), "text": t}) + "\n")
        fh.write(json.dumps({"id": "new", "text": "a text never archived"}) + "\n")
    rep = RP.run(bundle, corpus, tmp_path / "v.npy")
    assert rep["mode"] == "archival" and rep["exact"] == 10
    assert "consistent" in rep["warning"]
    rep2 = RP.run(bundle, corpus, tmp_path / "v2.npy", mode="consistent")
    assert rep2["warning"] == "" and rep2["exact"] == 0
