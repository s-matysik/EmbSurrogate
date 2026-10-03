"""The selection loop must be reproducible and must not need scikit-learn.

The reading order is the deliverable of a screening run, so it has to be a
function of the inputs alone -- no seeds, no solver version, no thread count.
"""

from __future__ import annotations

import numpy as np
import pytest

from embsurrogate import select as SEL


def _toy(n=120, d=16, seed=0):
    """Two Gaussian blobs: a separable problem with a known answer."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)
    X = rng.normal(size=(n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    inc = (X @ w) > np.quantile(X @ w, 0.85)       # ~15% positives
    return X, inc


def test_reading_order_is_identical_across_runs():
    X, inc = _toy()
    known = {int(i): bool(inc[i]) for i in list(np.where(inc)[0][:5])
             + list(np.where(~inc)[0][:5])}
    o1, _ = SEL.screen(X, dict(known), len(inc), batch=10, oracle=inc)
    o2, _ = SEL.screen(X, dict(known), len(inc), batch=10, oracle=inc)
    assert np.array_equal(o1, o2)
    assert sorted(o1.tolist()) == list(range(len(inc)))     # a permutation


def test_order_does_not_depend_on_record_order_beyond_the_data():
    """Shuffling the corpus must permute the order, not change which records win."""
    X, inc = _toy()
    perm = np.random.default_rng(1).permutation(len(inc))
    inv = np.argsort(perm)
    known_idx = list(np.where(inc)[0][:5]) + list(np.where(~inc)[0][:5])
    known = {int(i): bool(inc[i]) for i in known_idx}
    known_p = {int(inv[i]): bool(inc[i]) for i in known_idx}
    o, _ = SEL.screen(X, dict(known), len(inc), batch=10, oracle=inc)
    op, _ = SEL.screen(X[perm], dict(known_p), len(inc), batch=10, oracle=inc[perm])
    first = set(o[:40].tolist())
    first_p = {int(perm[i]) for i in op[:40]}
    # ties and a different labelling path make exact equality too strong; the
    # overlap of the first third is the claim that matters for a screener
    assert len(first & first_p) >= 30


def test_learned_order_beats_the_corpus_order():
    X, inc = _toy()
    known = {int(i): bool(inc[i]) for i in list(np.where(inc)[0][:5])
             + list(np.where(~inc)[0][:5])}
    o, _ = SEL.screen(X, dict(known), len(inc), batch=10, oracle=inc)
    learned = SEL.recall_curve(o, inc)["recall@20%"]
    baseline = SEL.recall_curve(np.arange(len(inc)), inc)["recall@20%"]
    assert learned > baseline


def test_no_decisions_is_an_error_not_a_silent_mode():
    X, inc = _toy()
    with pytest.raises(ValueError):
        SEL.screen(X, {0: True, 1: False}, len(inc), batch=10)


def test_kernel_logistic_separates_a_trivial_case():
    K = np.array([[1.0, 0.9, -0.9, -1.0],
                  [0.9, 1.0, -0.8, -0.9],
                  [-0.9, -0.8, 1.0, 0.9],
                  [-1.0, -0.9, 0.9, 1.0]]) + 10.0
    y = np.array([1.0, 1.0, -1.0, -1.0])
    a = SEL.kernel_logistic(K, y, np.ones(4), lam=1.0)
    f = K @ a
    assert f[0] > 0 and f[1] > 0 and f[2] < 0 and f[3] < 0


def test_no_scikit_learn_import():
    import embsurrogate.select as m
    src = open(m.__file__, encoding="utf-8").read()
    assert "sklearn" not in src and "scikit" not in src.replace(
        "scikit-learn's", "")
