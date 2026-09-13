"""The feature map must be a pure function of the text — in this process, in
another process, and under any hash seed.

Everything the surrogate archives rests on this: if Phi(text) is allowed to
drift, the stored dual coefficients no longer describe the same function.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from embsurrogate import features as F

TEXTS = [
    "umowa najmu lokalu użytkowego zawarta w Krakowie",
    "The quick brown fox jumps over 13 lazy dogs.",
    "Ångström ß ΣΙΓΜΑ — NFKC and casefold corner cases",
    "digits 2024 and 1999 fold to the same feature",
    "",
    "x",
]


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _digest(texts, idf, cfg) -> dict:
    indptr, indices, data = F.csr_arrays(texts, idf, cfg)
    return {"indptr": _sha(indptr), "indices": _sha(indices), "data": _sha(data),
            "nnz": int(len(data))}


@pytest.fixture(scope="module")
def idf():
    # A fixed idf table: the feature map is tested, not idf estimation.
    rng = np.random.default_rng(0)
    return 1.0 + rng.random(F.DEFAULT_CONFIG["n_buckets"])


def test_same_process_two_calls_identical(idf):
    cfg = F.DEFAULT_CONFIG
    a = F.csr_arrays(TEXTS, idf, cfg)
    b = F.csr_arrays(TEXTS, idf, cfg)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_row_order_independence(idf):
    """A text's row must not depend on the other texts in the batch."""
    cfg = F.DEFAULT_CONFIG
    indptr, indices, data = F.csr_arrays(TEXTS, idf, cfg)
    for i, t in enumerate(TEXTS):
        i0, i1 = indptr[i], indptr[i + 1]
        alone = F.csr_arrays([t], idf, cfg)
        assert np.array_equal(indices[i0:i1], alone[1])
        assert np.array_equal(data[i0:i1], alone[2])


CHILD = r"""
import hashlib, json, sys
import numpy as np
from embsurrogate import features as F
rng = np.random.default_rng(0)
idf = 1.0 + rng.random(F.DEFAULT_CONFIG["n_buckets"])
texts = json.loads(sys.argv[1])
indptr, indices, data = F.csr_arrays(texts, idf, F.DEFAULT_CONFIG)
sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
print(json.dumps({"indptr": sha(indptr), "indices": sha(indices),
                  "data": sha(data), "nnz": int(len(data))}))
"""


def _child(seed: str) -> dict:
    env = dict(os.environ, PYTHONHASHSEED=seed)
    out = subprocess.run([sys.executable, "-c", CHILD, json.dumps(TEXTS)],
                         capture_output=True, text=True, env=env, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_second_process_identical(idf):
    assert _child("0") == _digest(TEXTS, idf, F.DEFAULT_CONFIG)


@pytest.mark.parametrize("seed", ["0", "12345"])
def test_pythonhashseed_independent(idf, seed):
    """Hashing is SHA-256, not Python's randomized hash(), so the seed is inert."""
    assert _child(seed) == _digest(TEXTS, idf, F.DEFAULT_CONFIG)


def test_scipy_and_numpy_paths_agree(idf):
    """features.transform (scipy) must equal features.csr_arrays (numpy only).

    The two paths exist because inference has to run without scipy; if they
    disagreed even in the last bit, the archived vector would depend on which
    one the reader happened to use.
    """
    pytest.importorskip("scipy")
    cfg = F.DEFAULT_CONFIG
    X = F.transform(TEXTS, idf, cfg)
    indptr, indices, data = F.csr_arrays(TEXTS, idf, cfg)
    assert np.array_equal(X.indptr, indptr)
    assert np.array_equal(X.indices, indices)
    assert np.array_equal(X.data, data)


def test_config_fingerprint_stable():
    assert F.FEATURE_VERSION == "klon-features-3"
    assert F.config_fingerprint(dict(F.DEFAULT_CONFIG)) == \
        F.config_fingerprint(dict(F.DEFAULT_CONFIG))
