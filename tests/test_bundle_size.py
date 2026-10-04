"""The artefact must be small, and the two solver paths must agree.

Size is not cosmetic here. An embedding model that a review depended on is
87 MB for the smallest practical sentence encoder and 420 MB for one trained on
scholarly text, which is why depositing the model alongside a paper is not
normally an option. A bundle that approached those sizes would inherit the
problem it exists to solve, so the storage decisions are pinned by tests.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import embsurrogate.archive as AA
from embsurrogate import archive as A
from embsurrogate import evaluate as E
from embsurrogate import features as F
from embsurrogate import replay as RP
from embsurrogate.freeze import unpack_strings


@pytest.fixture(scope="module")
def fixture_pairs():
    z = np.load("tests/data/surrogate_tiny.npz", allow_pickle=False)
    return unpack_strings(z["texts"]), z["Y"]


def _write(tmp, texts, Y):
    with (tmp / "c.jsonl").open("w", encoding="utf-8") as fh:
        for i, t in enumerate(texts):
            fh.write(json.dumps({"id": f"r{i}", "text": t}, ensure_ascii=False) + "\n")
    np.save(tmp / "v.npy", Y)
    return tmp / "c.jsonl", tmp / "v.npy"


def test_arrays_are_stored_at_the_measured_precisions(tmp_path, fixture_pairs):
    """float32 coefficients, float16 coordinates, integer vectors.

    Measured on the 43,093-pair model: every fidelity metric is unchanged at
    four decimals under these precisions while the file drops from 558 MB to
    160 MB, so anything wider is paid for and buys nothing.
    """
    texts, Y = fixture_pairs
    tp, vp = _write(tmp_path, texts[:300], Y[:300])
    A.build(tp, vp, tmp_path / "b", k=20)
    sur = np.load(tmp_path / "b" / "surrogate.npz", allow_pickle=False)
    pairs = np.load(tmp_path / "b" / "pairs.npz", allow_pickle=False)
    assert sur["coef"].dtype == np.float32
    assert sur["z"].dtype == np.float16
    assert pairs["Y_micro"].dtype == np.int32
    # the integer vectors must still carry the full archival unit of 6 decimals
    assert np.array_equal(pairs["Y_micro"], np.round(Y[:300] * 1e6).astype(np.int32))


def test_bundle_is_smaller_than_the_float64_arrays(tmp_path, fixture_pairs):
    texts, Y = fixture_pairs
    n = 400
    tp, vp = _write(tmp_path, texts[:n], Y[:n])
    A.build(tp, vp, tmp_path / "b", k=20)
    size = sum(p.stat().st_size for p in (tmp_path / "b").iterdir())
    # what a naive format needs: vectors, coefficients, residuals and the
    # out-of-sample coordinates, all in float64. The bundle also carries a
    # fixed ~1 MB of bucket weights, which dominates at this corpus size and is
    # why the margin here is looser than the 3.5x measured at 43,093 pairs.
    naive = 4 * Y[:n].nbytes
    assert size < 0.6 * naive, (size, naive)


def test_residuals_are_derived_not_stored(tmp_path, fixture_pairs):
    texts, Y = fixture_pairs
    tp, vp = _write(tmp_path, texts[:300], Y[:300])
    A.build(tp, vp, tmp_path / "b", k=20)
    arrays = np.load(tmp_path / "b" / "surrogate.npz", allow_pickle=False).files
    assert "R" not in arrays and "lam_N" in arrays
    b = RP.Bundle(tmp_path / "b")
    assert np.allclose(b.R, float(np.load(tmp_path / "b" / "surrogate.npz")["lam_N"])
                       * b.coef)


def test_linked_bundle_needs_the_vector_file(tmp_path, fixture_pairs):
    texts, Y = fixture_pairs
    tp, vp = _write(tmp_path, texts[:300], Y[:300])
    A.build(tp, vp, tmp_path / "b", k=20, link_vectors=str(vp))
    with pytest.raises(SystemExit):
        RP.Bundle(tmp_path / "b")                    # no vectors supplied
    b = RP.Bundle(tmp_path / "b", vectors=vp)
    assert b.Y.shape == (300, Y.shape[1])
    other = tmp_path / "other.npy"
    np.save(other, np.roll(Y[:300], 1, axis=0))
    with pytest.raises(SystemExit):
        RP.Bundle(tmp_path / "b", vectors=other)     # wrong file, caught


def test_iterative_and_closed_form_paths_agree(fixture_pairs, monkeypatch):
    texts, Y = fixture_pairs
    cfg = dict(F.DEFAULT_CONFIG)
    tr, te = slice(0, 700), slice(700, 900)
    out = {}
    for limit, label in ((10_000, "closed"), (10, "iterative")):
        monkeypatch.setattr(AA, "CLOSED_FORM_LIMIT", limit)
        st = A.fit(texts[tr], Y[tr], cfg, k=20)
        yh = A.predict(texts[te], st["idf"], st["coef"], st["R"], st["z"],
                       st["X"], None, cfg, 1.0, 20)
        out[label] = E.report_full(Y[te], yh, Y[tr].mean(0))["rho_globalne"]
    assert abs(out["closed"] - out["iterative"]) < 0.005, out
