"""The claim the whole tool rests on: same raw data, same vectors.

These tests build a bundle from the committed tiny fixture, replay it, and
check the two cases separately -- archived texts must come back exactly, new
texts must be marked as approximated rather than quietly mixed in.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from embsurrogate import archive as A
from embsurrogate import replay as RP
from embsurrogate import verify as VF
from embsurrogate.freeze import unpack_strings


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Texts and vectors read out of the committed frozen fixture."""
    d = tmp_path_factory.mktemp("corpus")
    z = np.load("tests/data/surrogate_tiny.npz", allow_pickle=False)
    texts = unpack_strings(z["texts"])
    Y = z["Y"]
    n = min(400, len(texts))
    with (d / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({"id": f"r{i}", "text": texts[i]},
                                ensure_ascii=False) + "\n")
    np.save(d / "vectors.npy", Y[:n])
    return {"dir": d, "texts": texts, "Y": Y, "n": n}


@pytest.fixture(scope="module")
def bundle(corpus):
    out = corpus["dir"] / "bundle"
    A.build(corpus["dir"] / "corpus.jsonl", corpus["dir"] / "vectors.npy", out,
            k=20, teacher="fixture")
    return out


def test_manifest_records_measured_fidelity(bundle):
    man = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    f = man["fidelity_new_texts"]
    # the number must be measured on held-out records, not asserted
    assert f["n_holdout"] > 0
    assert 0.0 < f["rho_globalne"] <= 1.0
    assert man["params"]["alpha_mix"] == 1.0          # no table in this bundle


def test_same_raw_data_reproduces_exactly(bundle, corpus):
    rep = RP.run(bundle, corpus["dir"] / "corpus.jsonl",
                 corpus["dir"] / "replayed.npy")
    assert rep["exact"] == corpus["n"]
    assert rep["surrogate"] == 0
    assert rep["fully_reproducible"] is True
    got = np.load(corpus["dir"] / "replayed.npy")
    assert np.array_equal(got, np.round(corpus["Y"][: corpus["n"]], 6))


def test_new_records_are_flagged_not_hidden(bundle, corpus):
    extra = corpus["dir"] / "corpus_plus.jsonl"
    with extra.open("w", encoding="utf-8") as fh:
        for i in range(corpus["n"]):
            fh.write(json.dumps({"id": f"r{i}", "text": corpus["texts"][i]},
                                ensure_ascii=False) + "\n")
        for j in range(5):
            fh.write(json.dumps({"id": f"new{j}",
                                 "text": corpus["texts"][corpus["n"] + j]},
                                ensure_ascii=False) + "\n")
    prov = corpus["dir"] / "prov.csv"
    rep = RP.run(bundle, extra, corpus["dir"] / "replayed2.npy", prov)
    assert rep["exact"] == corpus["n"] and rep["surrogate"] == 5
    assert rep["fully_reproducible"] is False
    rows = prov.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0] == "id,source"
    assert [r.split(",")[1] for r in rows[1:]][-5:] == ["surrogate"] * 5


def test_a_modified_text_is_treated_as_new(bundle, corpus):
    """A record that differs from the archived one must not inherit its vector."""
    p = corpus["dir"] / "corpus_mod.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": "r0", "text": corpus["texts"][0] + " addendum"},
                            ensure_ascii=False) + "\n")
    rep = RP.run(bundle, p, corpus["dir"] / "replayed3.npy")
    assert rep["surrogate"] == 1 and rep["exact"] == 0


def test_damaged_bundle_refuses_to_load(bundle, corpus, tmp_path):
    import shutil

    broken = tmp_path / "broken"
    shutil.copytree(bundle, broken)
    man = json.loads((broken / "manifest.json").read_text(encoding="utf-8"))
    man["sha256"]["pairs.npz"] = "0" * 64
    (broken / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(SystemExit):
        RP.Bundle(broken)


def test_verify_passes_on_a_fresh_bundle(bundle):
    rep = VF.check(bundle, strict=True)
    assert rep["passed"], rep["checks"]
    assert rep["checks"]["archived_texts_exact"]
    assert rep["checks"]["new_text_routed_to_surrogate"]
