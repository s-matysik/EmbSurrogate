"""The encoder route: better when available, absent when it is not.

The package exists because a model can disappear, so a bundle must stay
readable without one. It is also true that if ANY encoder can still be run, its
vectors describe a new text far better than character n-grams do. The bundle
therefore carries both routes and replay picks what the machine can actually
load, recording the choice per record rather than leaving the user to guess.
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


def fake_encoder(Y_all, texts_all, noise=0.35):
    """An 'encoder' that knows the teacher space imperfectly.

    Standing in for a real model keeps the test offline and fast. It returns a
    noisy, rotated view of the teacher vector for a known text, which is what a
    different real encoder is: informative about meaning, not identical.
    """
    rng = np.random.default_rng(0)
    d = Y_all.shape[1]
    Q = np.linalg.qr(rng.normal(size=(d, d)))[0]
    lookup = {t: Y_all[i] @ Q + noise * rng.normal(size=d)
              for i, t in enumerate(texts_all)}
    fallback = (Y_all.mean(0) @ Q)

    def run(texts):
        out = np.array([lookup.get(t, fallback) for t in texts])
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return run


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    z = np.load(DATA / "surrogate_tiny.npz", allow_pickle=False)
    texts = unpack_strings(z["texts"])
    Y = np.asarray(z["Y"] if "Y" in z.files else z["Y_micro"] / 1e6, dtype=np.float64)
    out = tmp_path_factory.mktemp("bridge")
    corpus = out / "corpus.jsonl"
    with corpus.open("w", encoding="utf-8") as fh:
        for i, t in enumerate(texts):
            fh.write(json.dumps({"id": str(i), "text": t}) + "\n")
    np.save(out / "vectors.npy", Y)
    man_plain = A.build(corpus, out / "vectors.npy", out / "plain", k=20)
    man_good = A.build(corpus, out / "vectors.npy", out / "withenc", k=20,
                       encoder=fake_encoder(Y, texts, noise=0.05))
    man_weak = A.build(corpus, out / "vectors.npy", out / "weakenc", k=20,
                       encoder=fake_encoder(Y, texts, noise=1.5))
    return out, man_plain, man_good, man_weak


def test_bridge_is_recorded_and_measured(built):
    _, man_plain, man_bridge, _ = built
    assert man_plain["bridge"] is None
    b = man_bridge["bridge"]
    assert b is not None and b["deterministic"] is False
    assert set(("rho_globalne", "cos_mediana")) <= set(b["fidelity_new_texts"])


def test_bridge_is_kept_only_when_it_wins(built):
    """An informative encoder is stored; a mismatched one is refused.

    This is the behaviour that makes the option safe to offer: naming an
    encoder cannot make a bundle worse, because the choice is a measurement on
    the user's own corpus rather than a promise in the documentation.
    """
    _, man_plain, man_good, man_weak = built
    base = man_plain["fidelity_new_texts"]["rho_globalne"]
    good = man_good["bridge"]["fidelity_new_texts"]["rho_globalne"]
    assert good > base, (base, good)
    assert isinstance(man_weak["bridge"], str)
    assert "measured WORSE" in man_weak["bridge"]


def test_replay_falls_back_when_the_encoder_cannot_be_loaded(built):
    out, _, _, _ = built
    # the manifest names a callable, which no later process can resolve --
    # exactly the situation this package is for
    rep = RP.run(out / "withenc", out / "corpus.jsonl", out / "v.npy")
    # every text here IS archived, so no route runs at all
    assert rep["exact"] == rep["records"]
    assert rep["route_for_new_texts"].startswith("none")


def test_a_new_text_is_labelled_with_the_route_that_produced_it(built):
    out, _, _, _ = built
    extra = out / "extra.jsonl"
    extra.write_text(json.dumps({"id": "x", "text": "a text never archived"}) + "\n",
                     encoding="utf-8")
    prov = out / "prov.csv"
    RP.run(out / "withenc", extra, out / "v2.npy", provenance=prov)
    rows = prov.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0] == "id,source,route"
    # the bundle names a callable encoder, which no later process can resolve,
    # so the dependency-free route must take over and say so
    assert rows[-1].split(",")[1:] == ["surrogate", "n-gram"], rows
