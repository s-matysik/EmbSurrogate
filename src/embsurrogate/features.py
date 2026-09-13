"""Deterministic featurization of text for the embedding model surrogate.

Principle: the feature function Phi: text -> R^D must give an identical result on every
machine, in every year, without any learned tokenizer and without a dictionary
that would need to be distributed. Therefore:

* feature hashing (hashing trick, Weinberger et al. 2009) computed on SHA-256,
  and not on Python's built-in hash() — the latter is randomized per process
  (PYTHONHASHSEED) and is not stable across interpreter versions;
* the feature sign taken from the last bit of the same hash (signed hashing),
  which cancels the bias resulting from collisions;
* tf-idf weights in the sublinear version; the idf table is the ONLY element
  learned from data and is frozen in the model file;
* L2 normalization at the end, because the linear kernel on such vectors is cosine.

The entire arithmetic is float64, and the only step dependent on summation order
(tf accumulation) proceeds over sorted keys — see `_ordered_counts`."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from typing import Iterable, Sequence

import numpy as np

# scipy is imported lazily, only in the functions that build sparse matrices — this way
# the inference path (hashed_row + csr_arrays) runs on numpy alone.

# Changing anything below changes the feature space, and therefore the model.
# The version is saved in the model file and checked on load.
FEATURE_VERSION = "klon-features-3"

DEFAULT_CONFIG: dict = {
    "n_buckets": 1 << 18,          # 262144 buckets
    # Feature set selected by measurement (round 2, results/kompromis_cechy.csv):
    # a richer set worsens the result, because each additional feature type increases
    # the part of the new text's vector that lies OUTSIDE the span of the
    # training set (median p²: 0.72 for znaki3-5+słowa1-2 versus 0.58 for
    # znaki3-4), and the nonlinearity floor is negligible anyway (12.7% error).
    "char_ngrams": (3, 4),         # character n-grams on text with word boundaries
    "word_ngrams": (),             # features measurably harm
    "sublinear_tf": True,          # 1 + log(tf) instead of raw tf
    "digit_folding": True,         # each digit -> '0'
    "max_chars": 20000,            # hard limit on input length
    # Summation used by the L2 normalisation. "fsum" is correctly rounded
    # (math.fsum): independent of the order of addition, hence identical on
    # every IEEE-754 platform. "dot" is numpy's dot product, i.e. a BLAS call
    # whose reduction order depends on the architecture and the thread count.
    # Measured in CI across four operating-system/interpreter combinations:
    # the "dot" path produced THREE different digests of the feature-value
    # array (macOS 6c932cfb, Ubuntu 3.11 9f8ad920, Ubuntu 3.12 4fee5b61),
    # while "fsum" produced one and the same row norm, 0x1.5e8d69687db1ep+6,
    # on all of them. Models frozen before this change carry no such key and
    # are read through the "dot" path, so they stay bit-reproducible on the
    # architecture that produced them.
    "norm_sum": "fsum",
}

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d")

# SHA-256 is expensive (~1 us), and n-grams repeat massively — the cache gives
# a several-fold speedup and does not change the result.
_HASH_CACHE: dict[str, tuple[int, float]] = {}


def _hash(feature: str) -> tuple[int, float]:
    """(position before modulo, sign) — deterministic for a given string."""
    hit = _HASH_CACHE.get(feature)
    if hit is not None:
        return hit
    digest = hashlib.sha256(feature.encode("utf-8")).digest()
    h = int.from_bytes(digest[:8], "big")
    out = (h >> 1, 1.0 if (h & 1) else -1.0)
    if len(_HASH_CACHE) < 4_000_000:
        _HASH_CACHE[feature] = out
    return out


def normalize(text: str, cfg: dict | None = None) -> str:
    """NFKC -> casefold -> (optionally) digits to '0' -> whitespace collapsing.

    Casefold instead of lower(), because it correctly handles German ß and Greek;
    for Polish there is no difference, but the corpus can be multilingual.
    """
    cfg = cfg or DEFAULT_CONFIG
    t = unicodedata.normalize("NFKC", text)[: cfg["max_chars"]]
    t = t.casefold()
    if cfg["digit_folding"]:
        t = _DIGIT_RE.sub("0", t)
    return _SPACE_RE.sub(" ", t).strip()


def raw_features(text: str, cfg: dict | None = None) -> Counter:
    """Raw feature counts (string -> counter), before hashing.

    Word boundaries are explicitly marked with the '\\x02' character, so that an n-gram spanning
    the start of a word differs from the same sequence in the middle of a word.
    """
    cfg = cfg or DEFAULT_CONFIG
    t = normalize(text, cfg)
    counts: Counter = Counter()

    words = _WORD_RE.findall(t)
    for n in cfg["word_ngrams"]:
        if n == 1:
            counts.update("w1\x01" + w for w in words)
        else:
            counts.update(
                "w%d\x01" % n + "\x01".join(words[i : i + n])
                for i in range(len(words) - n + 1)
            )

    padded = "\x02" + t.replace(" ", "\x02") + "\x02"
    for n in cfg["char_ngrams"]:
        if len(padded) >= n:
            counts.update(
                "c%d\x01" % n + padded[i : i + n] for i in range(len(padded) - n + 1)
            )
    return counts


def _ordered_counts(counts: Counter) -> list[tuple[str, int]]:
    """Sorted by key — removes the result's dependence on iteration order."""
    return sorted(counts.items())


def hashed_row(text: str, cfg: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Text -> (bucket indices, tf values) without idf and without normalization."""
    cfg = cfg or DEFAULT_CONFIG
    nb = cfg["n_buckets"]
    acc: dict[int, float] = {}
    for feat, cnt in _ordered_counts(raw_features(text, cfg)):
        pos, sign = _hash(feat)
        val = (1.0 + np.log(cnt)) if cfg["sublinear_tf"] else float(cnt)
        idx = pos % nb
        acc[idx] = acc.get(idx, 0.0) + sign * val
    if not acc:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    idx = np.fromiter(sorted(acc), dtype=np.int64, count=len(acc))
    vals = np.array([acc[i] for i in idx], dtype=np.float64)
    return idx, vals


def tf_matrix(texts: Sequence[str], cfg: dict | None = None) -> sparse.csr_matrix:
    """Matrix tf (without idf, without normalisation), row = text."""
    from scipy import sparse
    cfg = cfg or DEFAULT_CONFIG
    indptr = [0]
    all_idx: list[np.ndarray] = []
    all_val: list[np.ndarray] = []
    for t in texts:
        idx, val = hashed_row(t, cfg)
        all_idx.append(idx)
        all_val.append(val)
        indptr.append(indptr[-1] + len(idx))
    data = np.concatenate(all_val) if all_val else np.empty(0)
    indices = np.concatenate(all_idx) if all_idx else np.empty(0, dtype=np.int64)
    return sparse.csr_matrix(
        (data, indices, np.array(indptr)),
        shape=(len(texts), cfg["n_buckets"]),
        dtype=np.float64,
    )


def fit_idf(tf: sparse.csr_matrix, cfg: dict | None = None) -> np.ndarray:
    """Smoothed idf computed on buckets (not on features) — as in sklearn."""
    n_docs = tf.shape[0]
    df = np.bincount(tf.indices, minlength=tf.shape[1]).astype(np.float64)
    return np.log((1.0 + n_docs) / (1.0 + df)) + 1.0


def apply_idf(tf: "sparse.csr_matrix", idf: np.ndarray,
              cfg: dict | None = None) -> "sparse.csr_matrix":
    """tf * idf plus row-wise L2 normalisation. Empty rows stay zero.

    Normalisation goes through `_row_norm`, the same code as the scipy-free
    path in `csr_arrays`. It used to sum the squares with `multiply().sum()`,
    a scipy reduction different from the one there, so the two paths could
    disagree in their last bits.
    """
    from scipy import sparse

    out = tf.copy()
    out.data = out.data * idf[out.indices]
    for i in range(out.shape[0]):
        lo, hi = out.indptr[i], out.indptr[i + 1]
        if hi > lo:
            n = _row_norm(out.data[lo:hi], cfg)
            if n > 0.0:
                out.data[lo:hi] = out.data[lo:hi] / n
    return out


def transform(
    texts: Iterable[str], idf: np.ndarray, cfg: dict | None = None
) -> "sparse.csr_matrix":
    """Full path of Phi: texts -> sparse matrix, rows with norm 1.

    Computed via `csr_arrays`, i.e. exactly the same code as the
    inference path without scipy — otherwise the two versions would differ in the last
    bits (multiplication by the inverse of the norm vs division).
    """
    from scipy import sparse

    cfg = cfg or DEFAULT_CONFIG
    texts = list(texts)
    indptr, indices, data = csr_arrays(texts, idf, cfg)
    return sparse.csr_matrix(
        (data, indices, indptr), shape=(len(texts), cfg["n_buckets"]), dtype=np.float64
    )


def fit_transform(
    texts: Sequence[str], cfg: dict | None = None
) -> tuple["sparse.csr_matrix", np.ndarray]:
    """Returns (features, idf). idf computed on the same texts."""
    cfg = cfg or DEFAULT_CONFIG
    idf = fit_idf(tf_matrix(texts, cfg), cfg)
    return transform(texts, idf, cfg), idf


def _row_norm(val: np.ndarray, cfg: dict | None = None) -> float:
    """L2 norm of one feature row, summed according to ``cfg["norm_sum"]``.

    This was the only place on the feature path whose result depended on the
    platform. ``val @ val`` goes through BLAS, where the reduction order
    depends on the architecture (SIMD width) and on the number of threads.
    ``math.fsum`` returns a correctly rounded sum, so it does not depend on the
    order at all; the element-wise squaring before it is exactly rounded too,
    which makes the whole quantity reproducible bit for bit across operating
    systems and interpreters.
    """
    mode = (cfg or DEFAULT_CONFIG).get("norm_sum", "dot")
    if mode == "fsum":
        return math.sqrt(math.fsum(val * val))
    if mode == "dot":                     # compatibility with feature version 2
        return float(np.sqrt(float(val @ val)))
    raise ValueError(f"unknown summation mode: {mode!r}")


def csr_arrays(
    texts: Sequence[str], idf: np.ndarray, cfg: dict | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The same as `transform`, but without scipy — returns raw CSR arrays.

    Legacy path: inference from a frozen model must depend solely
    on numpy, because the fewer dependencies, the greater the chance that in ten years
    the code will still run. The results of both paths must be identical bit for bit —
    this is enforced by `replay_check.py`.
    """
    cfg = cfg or DEFAULT_CONFIG
    indptr = np.zeros(len(texts) + 1, dtype=np.int64)
    idx_parts, val_parts = [], []
    for i, t in enumerate(texts):
        idx, val = hashed_row(t, cfg)
        val = val * idf[idx]
        nrm = _row_norm(val, cfg)
        if nrm > 0.0:
            val = val / nrm
        idx_parts.append(idx)
        val_parts.append(val)
        indptr[i + 1] = indptr[i] + len(idx)
    indices = np.concatenate(idx_parts) if idx_parts else np.empty(0, dtype=np.int64)
    data = np.concatenate(val_parts) if val_parts else np.empty(0, dtype=np.float64)
    return indptr, indices, data


def config_fingerprint(cfg: dict) -> str:
    """Configuration checksum — included in the model manifest."""
    payload = FEATURE_VERSION + "|" + "|".join(
        f"{k}={cfg[k]}" for k in sorted(cfg)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
