"""The fast numpy path must agree with the integer reference implementation.

`embsurrogate.reference` recomputes the same vector in arbitrary-precision
integer arithmetic: no floating-point summation (integer addition is
associative, so BLAS blocking cannot matter) and no functions outside IEEE-754
(`log` via `decimal`, the tricube powers as integer multiplications). It is the
normative definition of the archived vector; the numpy path is a fast
approximation of it, and this test is what keeps the approximation honest.

The comparison is made on the quantized INTEGERS, so the test has no tolerance
of its own. It also checks that the difference between the two paths stays an
order of magnitude below the distance to the nearest rounding boundary — if it
does not, agreement would be luck rather than a property.
"""

from __future__ import annotations

import numpy as np
import pytest

from embsurrogate.replay_check import check_reference

DECIMALS = 6
N_REF = 20          # all control texts of the shipped fixture


def test_numpy_path_equals_integer_reference(model_path, frozen, probes):
    ok, log = check_reference(model_path, frozen, probes[:N_REF], DECIMALS)
    assert ok, "\n".join(log)


def test_integer_reference_vectors_match_the_golden_file(model_path, probes, expected):
    from embsurrogate import reference as REF

    ref = REF.ReferenceModel(model_path)
    scale = 10 ** DECIMALS
    V6, _ = ref.encode_int_fast(probes[:5], chunk=5, scales=(scale, 10 ** 18))
    got = np.asarray(V6, dtype=np.int64)
    want = np.rint(expected[:5] * scale).astype(np.int64)
    assert np.array_equal(got, want)
