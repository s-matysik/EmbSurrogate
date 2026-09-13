"""The cross-platform contract, checked against a frozen model in the repository.

This is the test that makes the determinism claim falsifiable: the vectors in
tests/data/expected_tiny.npy were produced on the machine where the fixture was
built, and CI recomputes them on other operating systems and other numpy
builds. Comparison is exact (`array_equal`) on vectors quantized to 6 decimals.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from embsurrogate import features as F
from embsurrogate.predict import FrozenSurrogate
from embsurrogate.replay_check import check

DECIMALS = 6


def test_model_file_matches_its_manifest(model_path):
    man = json.loads(model_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    got = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert got == man["sha256_pliku"]
    assert man["spec"]["feature_version"] == F.FEATURE_VERSION
    assert man["spec"]["method"] == "hybrid_z"


def test_training_features_reproduce_bit_for_bit(frozen):
    """Features recomputed from the archived texts must match the frozen checksums."""
    indptr, indices, data = F.csr_arrays(frozen.texts, frozen.idf, frozen.cfg)
    sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    assert {"indptr": sha(indptr), "indices": sha(indices), "data": sha(data)} == \
        frozen.spec["sha256_cech"]


def test_golden_vectors_exact(frozen, probes, expected):
    V = frozen.encode(probes, decimals=DECIMALS)
    assert V.shape == expected.shape
    assert np.array_equal(V, expected), \
        f"max |difference| = {np.abs(V - expected).max():.3e}"


def test_encode_is_idempotent_within_a_process(frozen, probes):
    assert np.array_equal(frozen.encode(probes, decimals=DECIMALS),
                          frozen.encode(probes, decimals=DECIMALS))


def test_replay_check_entry_point(model_path, tmp_path, probes):
    """The shipped verification tool must pass on the shipped model."""
    ok, log = check(model_path, probes,
                    expected=model_path.parent / "expected_tiny.npy",
                    decimals=DECIMALS)
    assert ok, "\n".join(log)


def test_feature_version_mismatch_is_rejected(model_path, tmp_path, monkeypatch):
    """A model built on a different feature space must refuse to load."""
    z = dict(np.load(model_path, allow_pickle=False))
    spec = json.loads(bytes(z["spec"]).decode("utf-8"))
    spec["feature_version"] = "klon-features-1"
    z["spec"] = np.frombuffer(json.dumps(spec, ensure_ascii=False,
                                         sort_keys=True).encode("utf-8"),
                              dtype=np.uint8)
    bad = tmp_path / "bad_version.npz"
    np.savez(bad, **z)
    with pytest.raises(RuntimeError, match="klon-features-1"):
        FrozenSurrogate(bad)


def test_quantization_is_the_archival_unit(frozen, probes):
    """Raw float64 differs from the archived vector only by the rounding itself."""
    raw = frozen.encode(probes, decimals=None)
    q = frozen.encode(probes, decimals=DECIMALS)
    assert np.abs(raw - q).max() <= 0.5 * 10 ** -DECIMALS
    assert np.allclose(np.linalg.norm(raw, axis=1), 1.0, atol=1e-12)
