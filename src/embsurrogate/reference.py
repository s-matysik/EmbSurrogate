"""Reference, INTEGER-ONLY definition of the surrogate vector (archival contract).

PROBLEM THAT THIS MODULE SOLVES
-----------------------------------
`klon/predict.py` computes in float64 and quantizes the result to 6 decimal places.
The measured discrepancies (scipy vs numpy 2.8e-15; raw float64 vs quantized
5e-7) are an EMPIRICAL argument from one machine, not a guarantee. Floating-point
addition is not associative, so the sum of 20 million products depends on how
BLAS split the loop into blocks and threads, a different machine, a different
version of OpenBLAS, or a different number of cores may give a different last
bit. On top of that, two functions on the path are not covered by the IEEE-754
standard, so they are allowed to differ BETWEEN PLATFORMS by more than rounding:

* `np.log(cnt)` in `features.hashed_row` (value tf = 1 + ln cnt),
* `u ** 3` in `dual.tricube`: numpy has no fast path for exponent 3,
  so it calls `pow()` from libm.

IEEE-754 requires correct rounding only for +, -, *, /, sqrt and fma.
`log` and `pow` are "recommended", not required, glibc, Apple's libm and MSVC may
legally differ here by 1 ulp.

MECHANISM
---------
The entire path is recomputed using Python's INTEGER ARITHMETIC
(arbitrary precision, no overflows). Integer addition IS associative,
so the summation order stops mattering, there is nothing left to ask BLAS about.
All divisions are explicit rounding to the nearest integer
using the "half to even" rule (`_rhe`), and roots are computed by `math.isqrt`
(the exact integer part of the root, without floating-point approximation).

Scales (numbers of bits, constants below): the real value x is represented
by the integer round(2^s * x).

  1. text -> NFKC/casefold normalization -> n-grams  (pure Python, no floats)
  2. tf: TFQ(cnt) = round(2^S_TF * (1 + ln cnt)), ln computed with the `decimal`
     module at 60 digits, i.e. CORRECTLY ROUNDED and independent of libm. For cnt=1
     the result is exactly 2^S_TF.
  3. feature sign and bucket: the same operations on SHA-256 as in `features._hash`
     (integer arithmetic only, this was deterministic and remains so)
  4. idf: IDFQ[d] = round(2^S_IDF * idf[d]), `idf` is a frozen float64
     array, and multiplying float64 by a power of two together with `rint` are
     exact operations, so this step is bit-identical everywhere
  5. u_d = ACC_d * IDFQ_d, norm = isqrt(sum u_d^2),
     PHI_d = round(2^S1 * u_d / norm)                       [query feature]
  6. training features: XQ_pd = round(2^S1 * data_pd) from the frozen CSR array
     (its sha256 is in the manifest, it, not the text, is the oracle;
      agreement of reconstruction from text is measured separately, see report)
  7. KS_j = round(sum_d XQ_jd * PHI_d / 2^S1)                 [kernel, scale 2^S1]
  8. B_c = sum_j KS_j * AQ_jc, AQ = round(2^S2 * alpha)       [scale 2^(S1+S2)]
  9. BN_c = round(2^S3 * B_c / isqrt(sum B^2))                [l2norm(b)]
 10. SIM_j = sum_c BN_c * ZQ_jc                               [scale 2^(S3+S2)]
 11. neighbourhood: k smallest indices in order (-SIM_j, j), see below
 12. D_j = 1 - SIM_j, h = D_max * f (f = the exact value of the literal 1.0000001),
     U_j = round(2^S_U * D_j / h) clamped to [0, 2^S_U],
     W_j = round(2^S_W * (1 - (U_j/2^S_U)^3)^3), powers as integer multiplications
     (S_U is the most sensitive scale on the entire path, see comment by the constant)
 13. KOREKTA_c = round(sum_j W_j * RQ_(nb_j)c / sum_j W_j)
 14. OUT_c = B_c + KOREKTA_c * 2^S1,
     WYNIK_c = round(10^6 * OUT_c / isqrt(sum OUT^2)), this is an integer; the archival vector is WYNIK_c / 10^6.

TIES IN NEIGHBOUR SORTING
----------------------------
The `hybrid_z` method selects k = 20 neighbours by similarity in the OUTPUT
space. `np.argpartition` + `np.argsort` have no defined tie-breaking
behaviour (argpartition uses introselect, order depends on the
implementation, and `argsort` is not stable by default). Two processes could
therefore select a DIFFERENT set of neighbours and get a result differing by much
more than 1e-6.
Here the order is defined explicitly and completely:

    neighbours = the first k indices after sorting by key (-SIM_j, j)

i.e. descending by similarity, and for identical similarity, ascending by
training index. Similarities are integers, so "identical"
means exactly identical, there is no grey zone.

WHERE TO FIND THE NUMBERS
------------------
This module by itself proves nothing, the choice of scales (S1, S2, S3) and agreement
with the float64 path are the RESULT OF MEASUREMENT, not an assumption. The measurement is done by the script
`zmierz_kontrakt.py`; the results (maximum difference, number of texts non-matching after
quantization, seconds per text for each path, S1 sweep) go into
wyniki/kontrakt_precyzja.csv, and the conclusions, including the delineation of what is
guaranteed versus merely measured, into wyniki/kontrakt_wnioski.md."""

from __future__ import annotations

import json
import math
import pathlib
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction

import numpy as np

from . import features as F
from .predict import FrozenSurrogate

# --------------------------------------------------------------- scales
# Selection: see results/precision_contract.csv (S1 sweep = 24/32/40/48).
S_TF = 48        # value tf = 1 + ln(cnt)
S_IDF = 48       # idf array
S1 = 48          # normalized features (query and training) and kernel
S2 = 48          # frozen alpha, R, z coefficients
S3 = 48          # normalized 1st-degree vector (input to k-NN)
# Tricube kernel scales, NOT cosmetic. At s_u = 30 the relative weight error
# is ~3e-9, and the weight enters the weighted average of residuals, so the
# vector came out ~5e-10 off the exact value. That was enough for 10 of 1200
# validation texts to get a different sixth digit (wyniki/kontrakt_zrodla_bledu.json:
# raising only the coefficients to s2 = 54 changed nothing, whereas raising
# tricube alone to s_u = 54 removed all 10 discrepancies).
S_U = 54         # tricube kernel argument
S_W = 108        # tricube weight
DECIMALS = 6     # this many decimal places has the archival vector
OUT_SCALE = 10 ** DECIMALS

# exact value of the float64 literal 1.0000001 from dual._local (window radius)
H_SLACK = Fraction(1.0000001)


# ------------------------------------------------------- rounding
def _rhe(num: int, den: int) -> int:
    """round(num/den) using the "round half to even" rule, exclusively on integers.

    `divmod` in Python performs floor division, so for den > 0 the remainder is non-negative and
    the rule behaves identically on both sides of zero.
    """
    q, r = divmod(num, den)
    r2 = 2 * r
    if r2 > den or (r2 == den and (q & 1)):
        q += 1
    return q


def _rhe_pow2(num: np.ndarray, bits: int) -> np.ndarray:
    """Vectorized `_rhe(num, 2**bits)` on int64, without creating Python objects."""
    q = num >> bits
    r = num - (q << bits)               # 0 <= r < 2**bits
    half = np.int64(1) << np.int64(bits - 1)
    up = (r > half) | ((r == half) & ((q & 1) == 1))
    return q + up.astype(np.int64)


def quantize_array(a: np.ndarray, bits: int) -> np.ndarray:
    """round(2^bits * a) for a float64 array, an EXACT operation.

    Multiplication by a power of two introduces no rounding, and `np.rint` is
    rounding to the nearest integer using the "round half to even" rule
    required by IEEE-754. The result is therefore identical on every platform,
    as long as |2^bits * a| < 2^53 (checked by an assertion).
    """
    scaled = a * float(2 ** bits)
    if not np.all(np.abs(scaled) < 2.0 ** 53):
        raise ValueError(f"quantization at {bits} bitach wychodzi poza 2^53")
    return np.rint(scaled).astype(np.int64)


def quantize_float(x: float, bits: int) -> int:
    """Same for a single number (Python's `round` is "half to even")."""
    return int(round(x * float(2 ** bits)))


# --------------------------------------------- correctly rounded logarithm
_TFQ_CACHE: dict[int, int] = {}
_TFF_CACHE: dict[int, float] = {}


def tf_quant(cnt: int, bits: int = S_TF) -> int:
    """round(2^bits * (1 + ln cnt)), independent of libm.

    `Decimal.ln()` is correctly rounded to the specified number of digits
    (norm of the `decimal` module), so at 60 digits the value is ~1e-45 more
    precise than needed and rounding to an integer is unambiguous.
    """
    key = (cnt, bits)
    hit = _TFQ_CACHE.get(key)
    if hit is not None:
        return hit
    if cnt == 1:
        out = 1 << bits            # ln 1 = 0 exactly
    else:
        with localcontext() as ctx:
            ctx.prec = 60
            v = (Decimal(1) + Decimal(cnt).ln()) * (Decimal(2) ** bits)
            out = int(v.quantize(Decimal(1), rounding=ROUND_HALF_EVEN))
    _TFQ_CACHE[key] = out
    return out


def tf_float(cnt: int) -> float:
    """1 + ln(cnt) as a correctly rounded float64 (without libm), for the fsum path."""
    hit = _TFF_CACHE.get(cnt)
    if hit is not None:
        return hit
    if cnt == 1:
        out = 1.0
    else:
        with localcontext() as ctx:
            ctx.prec = 40
            out = float(Decimal(1) + Decimal(cnt).ln())
    _TFF_CACHE[cnt] = out
    return out


def _hash_int(feature: str) -> tuple[int, int]:
    """(position, sign as +-1), exactly the same bits as `features._hash`."""
    import hashlib
    h = int.from_bytes(hashlib.sha256(feature.encode("utf-8")).digest()[:8], "big")
    return h >> 1, (1 if (h & 1) else -1)


# --------------------------------------------------------- query features
def features_int(text: str, idfq: np.ndarray, cfg: dict,
                 s1: int = S1, s_tf: int = S_TF) -> tuple[list[int], list[int]]:
    """Text -> (sorted buckets, features as integers on a scale of 2^s1)."""
    nb = cfg["n_buckets"]
    sub = cfg["sublinear_tf"]
    acc: dict[int, int] = {}
    for feat, cnt in sorted(F.raw_features(text, cfg).items()):
        pos, sign = _hash_int(feat)
        val = tf_quant(cnt, s_tf) if sub else (cnt << s_tf)
        b = pos % nb
        acc[b] = acc.get(b, 0) + sign * val
    if not acc:
        return [], []
    keys = sorted(acc)
    u = [acc[b] * int(idfq[b]) for b in keys]
    nrm = math.isqrt(sum(x * x for x in u))
    if nrm == 0:
        return keys, [0] * len(keys)
    shift = 1 << s1
    return keys, [_rhe(x * shift, nrm) for x in u]


def features_fsum(text: str, idf: np.ndarray, cfg: dict
                  ) -> tuple[list[int], list[float]]:
    """This same path in float64, but each sum via `math.fsum`.

    `math.fsum` returns a CORRECTLY ROUNDED sum, which is unique
    regardless of the order of summands (Shewchuk 1997). Products, division and
    square root are covered by IEEE-754. The only function outside the standard would be the logarithm, here replaced by `tf_float`, so the entire path is fully deterministic.
    """
    nb = cfg["n_buckets"]
    sub = cfg["sublinear_tf"]
    terms: dict[int, list[float]] = {}
    for feat, cnt in sorted(F.raw_features(text, cfg).items()):
        pos, sign = _hash_int(feat)
        val = tf_float(cnt) if sub else float(cnt)
        terms.setdefault(pos % nb, []).append(sign * val)
    if not terms:
        return [], []
    keys = sorted(terms)
    u = [math.fsum(terms[b]) * float(idf[b]) for b in keys]
    nrm = math.sqrt(math.fsum(x * x for x in u))
    if nrm == 0.0:
        return keys, [0.0] * len(keys)
    return keys, [x / nrm for x in u]


# ========================================================== reference model
class ReferenceModel:
    """Frozen surrogate with three computation paths for the same vector.

    * `encode_int`: normative, exclusively Python integers, slow;
    * `encode_int_fast`: the same numbers, but sums computed in numpy such that
      EVERY intermediate result is an exact integer (decomposition into "limbs",
      see `_kstar_int`); the result must be identical to `encode_int`;
    * `encode_fsum`: float64 with `math.fsum` (correctly rounded sum).

    The `hybrid_z` method is supported (and `ridge` as its stage I).
    """

    # The widths of "limbs" are NOT fixed, they result from scales. The decomposition
    # in the kernel must satisfy s1 = 2*LIMB, because only then can the assembly in
    # `_kstar_int` (division by 2^s1) be done without overflowing int64.
    LIMB16 = 16        # basis of the decomposition in the Ks @ alpha product
    CAND = 200         # how many neighbour candidates we take before the exact computation

    def __init__(self, path, *, s1: int = S1, s2: int = S2, s3: int = S3,
                 s_tf: int = S_TF, s_idf: int = S_IDF, s_u: int = S_U, s_w: int = S_W,
                 csr_cache: pathlib.Path | None = None):
        if s1 % 2:
            raise ValueError("s1 must be even (split into two limbs of s1/2 bits)")
        self.s1, self.s2, self.s3 = s1, s2, s3
        self.s_tf, self.s_idf, self.s_u, self.s_w = s_tf, s_idf, s_u, s_w
        self.LIMB = s1 // 2
        self.fs = _load_frozen(path, csr_cache)
        self.cfg = self.fs.cfg
        self.method = self.fs.state["method"]
        if self.method not in ("hybrid_z", "ridge"):
            raise NotImplementedError(f"reference path supports hybrid_z/ridge, not {self.method}")
        self.k = int(self.fs.state.get("k", 20))
        self.idf = self.fs.idf
        self.idfq = quantize_array(self.idf, s_idf)
        self.indptr = np.asarray(self.fs.indptr, dtype=np.int64)
        self.indices = np.asarray(self.fs.indices, dtype=np.int32)
        self.n_train = len(self.indptr) - 1
        self.dim = self.fs.state["alpha"].shape[1]

        # --- training features as integers (scale 2^s1)
        Xq = quantize_array(self.fs.data, s1)
        self.Xq = Xq
        self.Xh = Xq >> self.LIMB
        self.Xl = Xq - (self.Xh << self.LIMB)
        self.row_nnz_max = int(np.diff(self.indptr).max())
        bound = int(np.abs(self.Xh).max()) * (1 << self.LIMB) * self.row_nnz_max
        if bound >= 2 ** 62:
            raise ValueError("limb decomposition would overflow int64, reduce LIMB")
        self._limb_bound = bound

        # --- frozen coefficients as integers (scale 2^s2)
        self.aq = quantize_array(self.fs.state["alpha"], s2)
        if self.method == "hybrid_z":
            self.Rq = quantize_array(self.fs.state["R"], s2)
            self.zq = quantize_array(self.fs.state["z"], s2)
            self.zf = self.fs.state["z"]
            zmax = int(np.abs(self.zq).max())
            bmax = 1 << self.s3
            self.LIMB_SIM, self._sim_int64_ok = 24, False
            for L in range(26, 15, -1):
                hi = (zmax >> L) * (bmax >> L) * self.dim
                lo = (1 << (2 * L)) * self.dim
                if max(hi, lo) < 2 ** 62:
                    self.LIMB_SIM, self._sim_int64_ok = L, True
                    break
            self.zqh = self.zq >> self.LIMB_SIM
            self.zql = self.zq - (self.zqh << self.LIMB_SIM)
        self._buf_h = np.zeros(self.cfg["n_buckets"], dtype=np.int64)
        self._buf_l = np.zeros(self.cfg["n_buckets"], dtype=np.int64)
        self.diag: dict = {}

    # ----------------------------------------------------------- kernel K_*
    def _kstar_int(self, keys: list[int], phi: list[int]) -> np.ndarray:
        """KS_j = round(sum_d XQ_jd * PHI_d / 2^s1), EXACTLY, in int64.

        The product of two 48-bit numbers does not fit in int64, so each
        is decomposed into two 24-bit parts (phi = qh*2^24 + ql) and we sum
        the four cross terms separately. Each of these sums is exact (the bound
        is checked in the constructor), and integer addition is associative,
        so `np.add.reduceat` can sum in any order and the result is
        unambiguous, this is the crux of the difference relative to the float64 path.
        The combination formula assumes s1 = 2*LIMB (see constructor).
        """
        n, L = self.n_train, self.LIMB
        if not keys:
            return np.zeros(n, dtype=np.int64)
        ki = np.asarray(keys, dtype=np.int64)
        pv = np.asarray(phi, dtype=np.int64)
        qh = pv >> L
        ql = pv - (qh << L)
        bh, bl = self._buf_h, self._buf_l
        bh[ki] = qh
        bl[ki] = ql
        start = self.indptr[:-1]
        gh = bh[self.indices]
        gl = bl[self.indices]
        A = np.add.reduceat(gh * self.Xh, start)
        M1 = np.add.reduceat(gh * self.Xl, start)
        M2 = np.add.reduceat(gl * self.Xh, start)
        Lo = np.add.reduceat(gl * self.Xl, start)
        del gh, gl
        bh[ki] = 0
        bl[ki] = 0
        empty = np.diff(self.indptr) == 0
        if empty.any():
            A[empty] = M1[empty] = M2[empty] = Lo[empty] = 0
        m1 = M1 >> L
        m2 = M2 >> L
        rest = (((M1 - (m1 << L)) + (M2 - (m2 << L))) << L) + Lo
        return (A + m1 + m2) + _rhe_pow2(rest, self.s1)

    # ------------------------------------------------- Level II: correction
    def _tricube_weights(self, sims: list[int], sim_scale: int) -> list[int] | None:
        """Tricube weights as integers in scale 2^S_W; None = exact match."""
        D = [sim_scale - s for s in sims]
        dmax = D[-1]
        if dmax <= 0:
            return None
        p, q = H_SLACK.numerator, H_SLACK.denominator
        den = dmax * p
        US = 1 << self.s_u
        US3 = US ** 3
        scale_w = 1 << self.s_w
        den9 = US3 ** 3
        out = []
        for dj in D:
            U = _rhe(dj * q * US, den)
            U = 0 if U < 0 else (US if U > US else U)
            t = US3 - U ** 3
            out.append(_rhe(t ** 3 * scale_w, den9))
        return out if sum(out) > 0 else None

    def _finish(self, b_raw: list[int], corr: list[int] | None,
                scales: tuple[int, ...] = (OUT_SCALE,)) -> list[list[int]]:
        """OUT = b + korekta*2^s1, normalization via isqrt, quantization to 1/scale.

        `scales` allows the same vector to be produced at several resolutions from
        A SINGLE computation: 10^6 is the archival vector, while e.g. 10^18 is used for
        comparing paths before quantization (a difference of 1e-12 would be invisible
        after rounding to 6 places).
        """
        s1 = self.s1
        out = [b_raw[c] + (corr[c] << s1) for c in range(self.dim)] if corr is not None \
            else list(b_raw)
        nrm = math.isqrt(sum(x * x for x in out))
        if nrm == 0:
            return [[0] * self.dim for _ in scales]
        return [[_rhe(x * sc, nrm) for x in out] for sc in scales]

    def _stage2(self, b_raw: list[int], sims_exact: list[tuple[int, int]],
                scales: tuple[int, ...] = (OUT_SCALE,)) -> list[list[int]]:
        """From the (sim, index) list sorted in descending order, produces a correction and the final vector."""
        nb = [j for _, j in sims_exact[:self.k]]
        sv = [s for s, _ in sims_exact[:self.k]]
        sim_scale = 1 << (self.s3 + self.s2)
        w = self._tricube_weights(sv, sim_scale)
        rows = [self.Rq[j].tolist() for j in nb]
        if w is None:
            corr = rows[0]
        else:
            sw = sum(w)
            corr = [_rhe(sum(wi * r[c] for wi, r in zip(w, rows)), sw)
                    for c in range(self.dim)]
        return self._finish(b_raw, corr, scales)

    def _bn(self, b_raw: list[int]) -> list[int]:
        """BN = round(2^s3 * b / ||b||), entry point to the k-NN output space."""
        nrm = math.isqrt(sum(x * x for x in b_raw))
        if nrm == 0:
            return [0] * self.dim
        sh = 1 << self.s3
        return [_rhe(x * sh, nrm) for x in b_raw]

    # ---------------------------------------------- path 1: pure numbers
    def encode_int(self, texts, scales: tuple[int, ...] = (OUT_SCALE,)) -> list:
        """Normative path: everything in Python integers.

        No shortcuts, the kernel is computed row by row, neighbours selected
        by full sorting of 25622 similarities with key (-sim, index).
        """
        outs = [np.zeros((len(texts), self.dim), dtype=object) for _ in scales]
        Xl_all = self.Xq
        indptr, indices = self.indptr.tolist(), self.indices
        shift = 1 << self.s1
        for t_i, text in enumerate(texts):
            keys, phi = features_int(text, self.idfq, self.cfg, self.s1, self.s_tf)
            qd = dict(zip(keys, phi))
            ks: list[int] = []
            for j in range(self.n_train):
                a, b_ = indptr[j], indptr[j + 1]
                s = 0
                for ix, xv in zip(indices[a:b_].tolist(), Xl_all[a:b_].tolist()):
                    pv = qd.get(ix)
                    if pv is not None:
                        s += xv * pv
                ks.append(_rhe(s, shift))
            acc = [0] * self.dim
            for j in range(self.n_train):
                kj = ks[j]
                if kj == 0:
                    continue
                row = self.aq[j].tolist()
                for c in range(self.dim):
                    acc[c] += kj * row[c]
            if self.method == "ridge":
                res = self._finish(acc, None, scales)
            else:
                bn = self._bn(acc)
                sims = []
                for j in range(self.n_train):
                    row = self.zq[j].tolist()
                    sims.append((sum(bn[c] * row[c] for c in range(self.dim)), j))
                sims.sort(key=lambda p: (-p[0], p[1]))
                res = self._stage2(acc, sims, scales)
            for o, r in zip(outs, res):
                o[t_i] = r
        return [o.astype(object) for o in outs]

    # ------------------------------------- path 2: the same numbers, vectorized
    def encode_int_fast(self, texts, *, chunk: int = 64, verbose: bool = False,
                        scales: tuple[int, ...] = (OUT_SCALE,)) -> list:
        """Identical arithmetic, sums via numpy with exact intermediate results.

        The Ks @ alpha product here is computed by dgemm calls on float64 arrays, which
        store EXCLUSIVELY integers < 2^53. Every multiplication and every
        addition is then exact, so the result does not depend on the summation
        order in BLAS, this is the same argument as with integers,
        only faster. Assertions guard that the 2^53 limit is not exceeded.
        """
        texts = list(texts)
        m = len(texts)
        outs = [np.zeros((m, self.dim), dtype=object) for _ in scales]
        gap_cand: list[float] = []
        gap_knn: list[float] = []
        nb_sets: list[tuple[int, ...]] = []
        fallbacks = 0
        SIMU = float(1 << (self.s3 + self.s2))
        for c0 in range(0, m, chunk):
            c1 = min(c0 + chunk, m)
            KS = np.empty((c1 - c0, self.n_train), dtype=np.int64)
            for i, text in enumerate(texts[c0:c1]):
                keys, phi = features_int(text, self.idfq, self.cfg, self.s1, self.s_tf)
                KS[i] = self._kstar_int(keys, phi)
            B = self._b_exact(KS)                       # list of lists of integers
            if self.method == "ridge":
                for i in range(c1 - c0):
                    for o, r in zip(outs, self._finish(B[i], None, scales)):
                        o[c0 + i] = r
                continue
            BN = [self._bn(b) for b in B]
            BNf = np.array(BN, dtype=np.float64) / float(2 ** self.s3)
            simf = BNf @ self.zf.T                      # candidate selection only
            for i in range(c1 - c0):
                cand, gap = self._candidates(simf[i])
                gap_cand.append(gap)
                sims = self._exact_sims(BN[i], cand)
                if gap <= 1e-9:                         # uncertain margin
                    fallbacks += 1
                    sims = self._exact_sims(BN[i], np.arange(self.n_train))
                sims.sort(key=lambda p: (-p[0], p[1]))
                gap_knn.append((sims[self.k - 1][0] - sims[self.k][0]) / SIMU)
                nb_sets.append(tuple(j for _, j in sims[:self.k]))
                for o, r in zip(outs, self._stage2(B[i], sims, scales)):
                    o[c0 + i] = r
            if verbose:
                print(f"  {c1}/{m}", flush=True)
        self.diag["margines_kandydatow_min"] = min(gap_cand) if gap_cand else None
        self.diag["odstep_k_kp1_min"] = min(gap_knn) if gap_knn else None
        self.diag["odstep_k_kp1"] = gap_knn
        self.diag["sasiedzi"] = nb_sets
        self.diag["pelne_sortowanie_awaryjne"] = fallbacks
        return [o for o in outs]

    def _b_exact(self, KS: np.ndarray) -> list[list[int]]:
        """B = Ks @ alpha exactly: decomposition into three limbs of 16 bits + dgemm."""
        L = self.LIMB16
        mask = (1 << L) - 1
        k2 = KS >> (2 * L)
        k1 = (KS >> L) & mask
        k0 = KS & mask
        a2 = self.aq >> (2 * L)
        a1 = (self.aq >> L) & mask
        a0 = self.aq & mask
        Ks_l = [k0.astype(np.float64), k1.astype(np.float64), k2.astype(np.float64)]
        A_l = [a0.astype(np.float64), a1.astype(np.float64), a2.astype(np.float64)]
        groups: dict[int, np.ndarray] = {}
        for i in range(3):
            for j in range(3):
                C = Ks_l[i] @ A_l[j]
                if np.abs(C).max() >= 2.0 ** 53:
                    raise ValueError("dgemm exceeded exact range of 2^53")
                s = i + j
                groups[s] = C if s not in groups else groups[s] + C
        tot = {s: np.rint(v).astype(np.int64).tolist() for s, v in groups.items()}
        m, d = KS.shape[0], self.dim
        out = []
        for r in range(m):
            row = [0] * d
            for s, arr in tot.items():
                sh = L * s
                ar = arr[r]
                for c in range(d):
                    row[c] += ar[c] << sh
            out.append(row)
        return out

    def _candidates(self, simf_row: np.ndarray) -> tuple[np.ndarray, float]:
        """Candidate neighbours + selection safety margin.

        Selection proceeds on the float64 approximation (error bounded by ~1e-13 in
cosine units), but the margin is returned: the difference between the
        k-th and CAND-th result. As long as it is orders of magnitude larger than
        the approximation error, no point outside the candidate set can enter
        the exact top twenty. When the margin is smaller than 1e-9,
        `encode_int_fast` falls back to full sorting.
        """
        c = min(self.CAND, self.n_train)
        idx = np.argpartition(-simf_row, c - 1)[:c]
        vals = np.sort(simf_row[idx])[::-1]
        return idx, float(vals[self.k - 1] - vals[-1])

    def _exact_sims(self, bn: list[int], cand: np.ndarray) -> list[tuple[int, int]]:
        """Exact SIM_j for given candidates (int64 limbs + assembly into int)."""
        L = self.LIMB_SIM
        if not self._sim_int64_ok:          # limbs don't fit, count in Python int
            cl = cand.tolist()
            return [(sum(b * r for b, r in zip(bn, self.zq[j].tolist())), j) for j in cl]
        bnv = np.asarray(bn, dtype=np.int64)
        bh = bnv >> L
        bl = bnv - (bh << L)
        zh = self.zqh[cand]
        zl = self.zql[cand]
        hh = zh @ bh
        hl = zh @ bl
        lh = zl @ bh
        ll = zl @ bl
        hh_l, hl_l, lh_l, ll_l = (a.tolist() for a in (hh, hl, lh, ll))
        cl = cand.tolist()
        return [((hh_l[i] << (2 * L)) + ((hl_l[i] + lh_l[i]) << L) + ll_l[i], cl[i])
                for i in range(len(cl))]

    # ------------------------------------------------- path 3: math.fsum
    def encode_fsum(self, texts) -> np.ndarray:
        """float64 + `math.fsum` everywhere there is a sum. Returns raw vectors."""
        data = self.fs.data
        indptr = self.indptr.tolist()
        indices = self.indices
        alpha = self.fs.state["alpha"]
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for t_i, text in enumerate(texts):
            keys, phi = features_fsum(text, self.idf, self.cfg)
            buf = np.zeros(self.cfg["n_buckets"], dtype=np.float64)
            if keys:
                buf[np.asarray(keys, dtype=np.int64)] = phi
            prod = buf[indices] * data          # products are exactly rounded
            ks = [math.fsum(prod[indptr[j]:indptr[j + 1]].tolist())
                  for j in range(self.n_train)]
            acc = [math.fsum(k * a for k, a in zip(ks, alpha[:, c].tolist()))
                   for c in range(self.dim)]
            if self.method == "ridge":
                out[t_i] = _l2_fsum(acc)
                continue
            bn = _l2_fsum(acc)
            sims = [(math.fsum(b * zz for b, zz in zip(bn, self.zf[j].tolist())), j)
                    for j in range(self.n_train)]
            sims.sort(key=lambda p: (-p[0], p[1]))
            nb = [j for _, j in sims[:self.k]]
            d = [1.0 - sims[i][0] for i in range(self.k)]
            h = d[-1] * 1.0000001
            R = self.fs.state["R"]
            if h <= 0.0:
                corr = R[nb[0]].tolist()
            else:
                w = []
                for dj in d:
                    u = dj / h
                    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
                    v = 1.0 - u * u * u          # explicit multiplications, not pow()
                    w.append(v * v * v)
                sw = math.fsum(w)
                if sw <= 0.0:
                    corr = R[nb[0]].tolist()
                else:
                    rows = [R[j].tolist() for j in nb]
                    corr = [math.fsum(wi * r[c] for wi, r in zip(w, rows)) / sw
                            for c in range(self.dim)]
            out[t_i] = _l2_fsum([a + c for a, c in zip(acc, corr)])
        return out


def _l2_fsum(v) -> list[float]:
    nrm = math.sqrt(math.fsum(x * x for x in v))
    if nrm == 0.0:
        return [0.0] * len(v)
    return [x / nrm for x in v]


def _load_frozen(path, csr_cache: pathlib.Path | None):
    """FrozenSurrogate with optional CSR array cache (feature reconstruction takes ~85 s)."""
    path = pathlib.Path(path)
    if csr_cache is None:
        return FrozenSurrogate(path)
    csr_cache = pathlib.Path(csr_cache)
    if csr_cache.exists():
        z = np.load(csr_cache, allow_pickle=False)
        fs = FrozenSurrogate.__new__(FrozenSurrogate)
        zz = np.load(path, allow_pickle=False)
        fs.spec = json.loads(bytes(zz["spec"]).decode("utf-8"))
        fs.cfg = fs.spec["cfg"]
        fs.cfg["char_ngrams"] = tuple(fs.cfg["char_ngrams"])
        fs.cfg["word_ngrams"] = tuple(fs.cfg["word_ngrams"])
        fs.idf = zz["idf"]
        from .freeze import unpack_strings
        fs.texts = unpack_strings(zz["texts"])
        fs.ids = unpack_strings(zz["ids"])
        fs.indptr, fs.indices, fs.data = z["indptr"], z["indices"], z["data"]
        fs.state = {"method": fs.spec["method"], "Y": zz["Y"], **fs.spec["params"]}
        for key in ("alpha", "R", "U", "lam_eig", "z"):
            if key in zz.files:
                fs.state[key] = zz[key]
        fs._buf = np.zeros(fs.cfg["n_buckets"], dtype=np.float64)
        return fs
    fs = FrozenSurrogate(path)
    np.savez(csr_cache, indptr=fs.indptr, indices=fs.indices, data=fs.data)
    return fs


def quantized_to_float(V6: np.ndarray) -> np.ndarray:
    """Vector archival (integers at scale 1e-6) to float64."""
    return V6.astype(np.float64) / float(OUT_SCALE)
