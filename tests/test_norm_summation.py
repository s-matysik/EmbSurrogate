"""The feature values must not depend on the order of summation.

The first CI run of this repository failed here: the L2 normalisation used a
BLAS dot product, whose reduction order depends on the architecture and the
thread count, and the feature-value array came out with three different
digests across four operating-system/interpreter combinations. These tests
pin the fix.
"""

from __future__ import annotations

import math

import numpy as np

from embsurrogate import features as F


def test_default_config_uses_correctly_rounded_summation():
    assert F.DEFAULT_CONFIG["norm_sum"] == "fsum"


def test_row_norm_is_invariant_to_permutation():
    rng = np.random.default_rng(0)
    v = rng.normal(size=2000) * 10.0 ** rng.integers(-6, 6, size=2000)
    n = F._row_norm(v, F.DEFAULT_CONFIG)
    for seed in range(5):
        perm = np.random.default_rng(seed).permutation(len(v))
        assert F._row_norm(v[perm], F.DEFAULT_CONFIG) == n


def test_row_norm_equals_correctly_rounded_reference():
    rng = np.random.default_rng(1)
    v = rng.normal(size=777)
    assert F._row_norm(v, F.DEFAULT_CONFIG) == math.sqrt(math.fsum(v * v))


def test_dot_path_still_available_for_version_2_models():
    cfg = dict(F.DEFAULT_CONFIG, norm_sum="dot")
    rng = np.random.default_rng(2)
    v = rng.normal(size=500)
    assert F._row_norm(v, cfg) == float(np.sqrt(float(v @ v)))


def test_rows_are_unit_norm_under_fsum():
    texts = ["a text about kernels", "zupełnie inny tekst", "third one, quite short"]
    indptr, indices, data = F.csr_arrays(texts, *F.fit_transform(texts)[1:], )
    for i in range(len(texts)):
        lo, hi = indptr[i], indptr[i + 1]
        assert abs(math.fsum(data[lo:hi] * data[lo:hi]) - 1.0) < 1e-15
