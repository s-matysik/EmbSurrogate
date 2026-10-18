"""Ties in the neighbour similarities must be resolved by an explicit key.

`argpartition`/`argsort` are free to order equal keys however the underlying
implementation likes, and on the published model the smallest measured gap
between the k-th and the (k+1)-th neighbour was 1.99e-16, i.e. ties do occur
in practice. A different neighbour SET is a different function, not a rounding
difference, so the selection key is fixed to (-similarity, training index).
"""

from __future__ import annotations

import numpy as np

from embsurrogate import dual as D


def _onehot(n: int, d: int | None = None) -> np.ndarray:
    return np.eye(n, d or n, dtype=np.float64)


def test_all_ties_select_lowest_indices():
    """With every similarity equal, the first k training points must win."""
    n, k = 8, 3
    Y = _onehot(n)
    ks = np.full((1, n), 0.5)
    out = D._local(ks, Y, k, degree=0)
    # equal similarities -> equal tricube weights -> plain mean of the chosen rows
    got = np.flatnonzero(out[0] > 1e-12)
    assert got.tolist() == [0, 1, 2]


def test_ties_at_the_top_select_the_lowest_tied_indices():
    """Tied maxima: the k lowest training indices among them must be taken."""
    n, k = 9, 3
    Y = _onehot(n)
    ks = np.full((1, n), 0.1)
    ks[0, [7, 2, 5, 8]] = 0.9        # four tied candidates, only three seats
    out = D._local(ks, Y, k, degree=0)
    assert np.flatnonzero(out[0] > 1e-12).tolist() == [2, 5, 7]


def _manual(ks_row, Y, k, nb):
    """Reference recomputation of `_local(degree=0)` for a GIVEN neighbour set."""
    sims = ks_row[nb]
    d = 1.0 - sims
    h = d[-1] * 1.0000001
    w = D.tricube(d / h)
    return (w @ Y[nb]) / w.sum()


def test_selection_is_invariant_to_repetition_and_to_batching():
    rng = np.random.default_rng(0)
    n, k = 50, 7
    Y = rng.standard_normal((n, 4))
    ks = np.round(rng.random((12, n)), 2)      # rounding manufactures many ties
    a = D._local(ks, Y, k, degree=0)
    b = D._local(ks, Y, k, degree=0)
    assert np.array_equal(a, b)
    rows = np.vstack([D._local(ks[i:i + 1], Y, k, degree=0) for i in range(len(ks))])
    assert np.array_equal(a, rows)


def test_key_matches_the_documented_lexsort():
    """The neighbourhood used must be exactly the (-similarity, index) key.

    Checked bit for bit against a recomputation with that key, and shown to be
    discriminating: swapping one tied neighbour for another changes the result,
    so passing this is evidence about the SET, not only about the weights.
    """
    rng = np.random.default_rng(1)
    n, k = 40, 5
    ks = np.round(0.5 * rng.random((1, n)), 2)   # rounding manufactures ties
    ks[0, 3] = 1.0                               # one strictly nearest neighbour
    ks[0, [4, 11, 19, 27, 31]] = 0.9             # five tied candidates, four seats
    Y = _onehot(n)
    nb = np.lexsort((np.arange(n), -ks[0]))[:k]
    out = D._local(ks, Y, k, degree=0)
    assert np.array_equal(out[0], _manual(ks[0], Y, k, nb))

    alt = nb.copy()
    tied = [j for j in range(n) if ks[0, j] == ks[0, nb[-1]] and j not in nb]
    assert tied, "the test case must contain an unused tied neighbour"
    alt[-1] = tied[0]
    assert not np.array_equal(out[0], _manual(ks[0], Y, k, alt))


def test_tricube_is_computed_by_multiplication_not_pow():
    """tricube must use exactly-rounded operations only (no libm pow)."""
    u = np.linspace(0.0, 1.0, 101)
    t = 1.0 - u * u * u
    assert np.array_equal(D.tricube(u), t * t * t)
    assert np.array_equal(D.tricube(np.array([-1.0, 2.0])), np.array([1.0, 0.0]))
