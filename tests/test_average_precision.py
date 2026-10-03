"""Average precision is computed in-package; this pins it to the definition.

The reference implementation is scikit-learn's `average_precision_score`, which
is NOT a dependency of this package -- the test skips when it is absent and
falls back to properties that hold regardless.
"""

from __future__ import annotations

import numpy as np
import pytest

from embsurrogate.evaluate import average_precision


def test_perfect_and_worst_ranking():
    y = np.array([True, True, False, False])
    assert average_precision(y, np.array([4.0, 3.0, 2.0, 1.0])) == pytest.approx(1.0)
    assert average_precision(y, np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(
        (1 / 3 + 2 / 4) / 2)


def test_all_tied_scores_give_the_prevalence():
    y = np.array([True, False, False, False])
    assert average_precision(y, np.zeros(4)) == pytest.approx(0.25)


def test_no_positives_is_zero_not_a_crash():
    assert average_precision(np.zeros(5, dtype=bool), np.arange(5.0)) == 0.0


def test_matches_scikit_learn_when_available():
    skl = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(60):
        n = int(rng.integers(20, 300))
        y = rng.random(n) < rng.uniform(0.05, 0.5)
        if not y.any():
            continue
        s = rng.normal(size=n)
        if trial % 3 == 0:
            s = np.round(s, 1)          # force ties, where the definitions differ
        worst = max(worst, abs(average_precision(y, s)
                               - skl.average_precision_score(y, s)))
    assert worst < 1e-12
