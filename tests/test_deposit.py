"""A deposit must carry no source text, and must still work.

Abstracts from the major bibliographic databases cannot be redistributed, so a
bundle that stores them is useless as a deposit however good its fidelity is.
These tests pin the two properties that make the text-free form usable: not one
character of the archived texts appears in the files, and a record the author
still holds locally is matched by digest and returns its stored vector.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from embsurrogate import archive as A
from embsurrogate import deposit as DP
from embsurrogate.freeze import unpack_strings

DATA = pathlib.Path(__file__).parent / "data"


def fake_encoder(Y_all, texts_all, noise=0.05):
    rng = np.random.default_rng(1)
    d = Y_all.shape[1]
    Q = np.linalg.qr(rng.normal(size=(d, d)))[0]
    lookup = {t: Y_all[i] @ Q + noise * rng.normal(size=d)
              for i, t in enumerate(texts_all)}
    fallback = Y_all.mean(0) @ Q

    def run(texts):
        out = np.array([lookup.get(t, fallback) for t in texts])
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return run


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    z = np.load(DATA / "surrogate_tiny.npz", allow_pickle=False)
    texts = unpack_strings(z["texts"])
    Y = np.asarray(z["Y"] if "Y" in z.files else z["Y_micro"] / 1e6,
                   dtype=np.float64)
    out = tmp_path_factory.mktemp("dep")
    corpus = out / "corpus.jsonl"
    with corpus.open("w", encoding="utf-8") as fh:
        for i, t in enumerate(texts):
            fh.write(json.dumps({"id": f"rec{i}", "text": t}) + "\n")
    np.save(out / "vectors.npy", Y)
    man_b = A.build(corpus, out / "vectors.npy", out / "bundle", k=20)
    enc = fake_encoder(Y, texts)
    man_d = DP.build(out / "bundle", out / "deposit", enc, k=20)
    return out, texts, Y, man_b, man_d


def test_no_source_text_is_present_in_the_deposit(made):
    out, texts, _, _, man_d = made
    assert man_d["contains_source_texts"] is False
    blob = b"".join(p.read_bytes() for p in (out / "deposit").iterdir())
    for t in texts[:50]:
        frag = t.strip()[:40].encode("utf-8")
        if len(frag) >= 20:
            assert frag not in blob, frag[:30]


def test_archived_records_still_return_their_stored_vector(made):
    out, texts, Y, _, _ = made
    d = DP.Deposit(out / "deposit", encoder=fake_encoder(Y, texts))
    got, prov = d.vectors_for(list(texts[:30]))
    assert set(prov) == {"exact"}
    assert np.abs(got - np.round(Y[:30], 6)).max() < 5e-7


def test_a_new_text_is_embedded_through_the_matrix(made):
    out, texts, Y, _, man_d = made
    d = DP.Deposit(out / "deposit", encoder=fake_encoder(Y, texts))
    got, prov = d.vectors_for(["a text that was never archived anywhere"])
    assert prov == ["surrogate"]
    assert abs(float(np.linalg.norm(got[0])) - 1.0) < 1e-9
    # W does not grow with the corpus: it is encoder-dim by teacher-dim
    assert man_d["W_shape"][1] == Y.shape[1]


def test_the_deposit_is_smaller_than_the_bundle(made):
    out, _, _, _, _ = made
    size = lambda p: sum(f.stat().st_size for f in p.iterdir() if f.is_file())
    assert size(out / "deposit") < size(out / "bundle")
