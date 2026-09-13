# The bit-for-bit contract

What is **guaranteed**, what is only **measured**, and how to check either.

The claim: *for a given frozen model file and a given text, the archived vector
— L2-normalised and quantised to 6 decimals — is the same on every IEEE-754
machine.* This document states the conditions under which that is a theorem and
the conditions under which it is merely an observation with a known margin.

Measurements below were made on `model/surogat_v2.npz` (25 622 pairs, method
`hybrid_z`, `k = 20`, `lambda = 1e-4`) over the 1200-text validation set, with
Python 3.12, numpy 2.5, Unicode UCD 15.0.0, macOS/arm64, `OMP_NUM_THREADS=2`.
The fresh test set was not touched for this purpose.

---

## 1. The result in one sentence

Quantisation to 6 decimals is **not** a guarantee of cross-platform agreement —
it only lowers the probability of disagreement. The guarantee comes from an
explicit definition of the computation in **integer arithmetic**
(`embsurrogate/reference.py`), against which the numpy path is a fast
approximation: measurably good, but an approximation.

## 2. Measured numbers

| measurement | value |
|---|---|
| max \|numpy − reference\| before quantisation (1200 texts) | 1.14e-13 |
| texts whose 6th decimal differs (of 1200) | 0 |
| smallest margin to a rounding boundary (460 800 coordinates) | 1.23e-12 |
| margin / path difference (headroom) | 11× |
| max \|numpy − `math.fsum`\| (120 texts) | 1.51e-14 |
| smallest gap between the k-th and (k+1)-th neighbour | 1.99e-16 |
| texts with a different neighbour **set** than the reference | 2 / 1200 |
| texts with a different neighbour **order** | 44 / 1200 |
| global rho: numpy / reference / ridge baseline | 0.7900 / 0.7900 / 0.7709 |
| per-query rho: numpy / reference / ridge baseline | 0.7790 / 0.7790 / 0.7685 |

Cost per text (one core, 25 622 training points): numpy **0.10 s**, integer
arithmetic vectorised **0.50 s**, `math.fsum` **5.6 s**, integer arithmetic in
pure Python **3.4 s**. Full table: `wyniki/kontrakt_precyzja.csv` of the study.

The two rows about neighbour sets are the reason `dual._local` selects
neighbours with `np.lexsort((index, -similarity))` rather than
`argpartition`/`argsort`: with a smallest gap of 2e-16, the identity of the
`k`-th neighbour is a coin flip at float64 resolution, and a different
neighbour set is a different function — an error of order 1e-3, not 1e-13.

## 3. Why quantisation alone is not enough

An archived vector of 1200 texts × 384 dimensions is 460 800 numbers. If two
implementations differ by `eps`, a mismatch appears whenever some coordinate
lies closer than `eps` to a rounding boundary of `0.5e-6`. Under a uniform
fractional part, the expected count of such coordinates is `2·n·eps/quant`:
**922 predicted versus 936 measured** at `eps = 1e-9`, so the uniform model
holds and can be used to reason further.

The same model predicts the number of digit flips,
`E[flips] = n·E|difference|/quant`. In the first run, 10 coordinates flipped,
implying `E|difference| ≈ 2.2e-11`, consistent with the observed maximum of
9.3e-10 (a typical deviation 43× below the maximum is the expected shape of a
rounding-error distribution). After the path difference was reduced to 1.1e-13,
the expected number of flips falls below one — and zero were measured.

The conclusion is quantitative, not rhetorical: **risk grows linearly with the
number of coordinates and falls linearly with the quantum, and no fixed
quantisation zeroes it.** The present 11× headroom means one coordinate per
1200 texts is an order of magnitude away from flipping. That works; it is not
proof, and it should not be presented as proof.

## 4. What is guaranteed bit for bit

`embsurrogate.reference` returns the vector as integers (value = `n / 10^6`).
Its result is a function of the text, the model file and the fixed scales alone,
under eight assumptions — each verifiable without running the code:

* **A1** Python integer semantics: arbitrary precision, no overflow, **associative**
  addition. Summation order stops being a question, so thread count and BLAS
  version cannot matter.
* **A2** `math.isqrt` — the exact integer part of a square root (Python ≥ 3.8), with
  no floating-point approximation.
* **A3** `decimal.Decimal.ln` at `prec = 60` — correctly rounded per the General
  Decimal Arithmetic specification, which makes `tf = 1 + ln(cnt)` independent
  of libm.
* **A4** `hashlib.sha256` — FIPS 180-4 (feature bucket and sign).
* **A5** IEEE-754 binary64 for multiplication by a power of two, `rint`, division
  and `sqrt` — all **required** by the standard. Hence `round(2^s · a)` on the
  frozen arrays is exact and identical everywhere.
* **A6** Neighbour ordering by the integer key `(-similarity, index)`: a tie means
  exact equality, not "almost equal".
* **A7** The **frozen arrays** in the `.npz` (sha256 in the manifest) are the
  oracle — not features recomputed from the text. Recomputation agrees to
  1.9e-15 with identical bucket indices, but it passes through `np.log`; see E2.
* **A8** The Unicode version. `features.normalize` calls NFKC and `casefold`, which
  depend on the UCD table compiled into the interpreter (15.0.0 here).
  Irrelevant for ASCII, material for a multilingual corpus.
  **Open recommendation: record `unicodedata.unidata_version` in the model
  manifest.** It is currently not written there (the manifest stores the Python,
  numpy and platform versions), and it is the one input to the computation that
  cannot be recovered from the file. This has not been changed in the code
  because doing so would alter the manifest format of already published models.

Under A1–A8 agreement is **guaranteed**, not measured: there is not a single
operation left in that computation whose result the standard leaves open.

### Scales

Real value `x` is represented by `round(2^s · x)`. Measured sufficiency
(criterion: the same 6 digits as the numpy path):

| scale | swept values | mismatching texts (of 60) |
|---|---|---|
| `s1` (features and kernel) | 24 / 32 / 40 / 48 | 60 / 3 / 0 / 0 |
| `s2 = s3` (coefficients) | 24 / 32 / 40 / 48 | 60 / 12 / 1 / 0 |
| `s_u`, `s_w` (tricube) | (30, 60) → (54, 108) | 10 / 10 → 0 / 10 |

`s1 = 40` suffices (48 is used, for headroom), `s2 = s3 = 48`, and — the
surprise of the measurement — `s_u = 54`, `s_w = 108`. The original `s_u = 30`
produced an error of 4.8e-10, three orders of magnitude above the coefficient
quantisation, and accounted for **all** mismatches of the first run. The
separating experiment: raising only the coefficients to `s2 = 54` left 10/10
mismatching (4.8e-10), while raising only tricube to `s_u = 54` gave 0/10
(6.9e-14). Mechanism: the tricube weight enters a weighted mean of residuals,
so its relative error ~3·2^(−s_u) propagates straight to the output vector.
Upper bound on the scales: `round(2^s · a)` must stay below `2^53` for the
float64 → integer conversion to be exact, giving `s ≤ 54` for `z` (max 0.297)
and `s ≤ 49` for `idf` (max 11.15).

The sweep of `s1` and `s2` was run with the *old* tricube scales
(`s_u = 30`), so the common 4.8e-10 error is present in every row: those rows
measure sensitivity to `s1`/`s2` *at a fixed* tricube error, and "0/60" means
"no worse than tricube", not "exact". The ordering of the conclusions is
unaffected, because the tricube error is common to all rows while the
differences between rows are orders of magnitude larger.

## 5. What is only empirical

* **E1 — agreement of the fast numpy path.** 0 / 1200 mismatching, difference
  1.1e-13 against a margin of 1.2e-12 (11× headroom). Measured on **one**
  machine with **one** BLAS. The contribution of summation order can, however,
  be bounded separately: `math.fsum` is correctly rounded and therefore
  order-independent, and numpy differs from it by 1.5e-14 — a factor of 3.3e7,
  i.e. seven and a half orders of magnitude below the 5e-7 threshold. *Summation order in BLAS is not the
  threat to the sixth digit; ties in k-NN and functions outside the standard
  are.*
* **E2 — functions outside IEEE-754 on the fast path.** The standard requires
  exact rounding only for `+ - * / sqrt fma`; `log` and `pow` are recommended,
  not required. Both occur on the numpy path: `np.log(cnt)` in featurisation and
  the cube in the tricube kernel. Measured here: `math.log` and `np.log` are
  correctly rounded for all `cnt` in 1..300 (0 deviations), and `u**3` differs
  from `u*u*u` by at most 1.1e-16 (1 ulp) — which is why the code uses the
  multiplication form. The two-step value `1.0 + log(cnt)` differs from a
  correctly rounded `1 + ln(cnt)` for 9 of 300 arguments; that is the effect of
  two roundings, deterministic as long as `log` is correctly rounded, which
  another platform need not provide. Consequence: recomputing features from text
  is diagnostics, not the source of truth (A7).
* **E3 — ties in k-NN.** With a smallest measured gap of 2.0e-16, the choice of
  the last neighbour is arbitrary at float64 resolution for some queries. The
  reference settles it by rule A6 and `dual._local` now uses the same key; the
  measurements of 44/1200 different orders and 2/1200 different sets were taken
  *before* that key was in place, and they are the evidence for why it is needed.

## 6. How to check it

On a frozen model with a manifest, control texts and a golden file:

```bash
export OMP_NUM_THREADS=2
python -m embsurrogate.replay_check model.npz \
    --probes probes.jsonl --expected expected.npy
```

This verifies, separately because each can break on its own: (1) the sha256 of
the file against the manifest; (2) that features recomputed from the archived
texts are bit-for-bit identical to the frozen ones (this is where a changed
feature version, a different `PYTHONHASHSEED` or a different Unicode
normalisation shows up); (3) that the vectors equal the golden file and that two
calls in one process agree.

Against the normative definition — the only check that catches a different BLAS
or a different libm:

```bash
python -m embsurrogate.replay_check model.npz \
    --probes probes.jsonl --reference --n-ref 20 \
    --csr-cache /tmp/csr_cache.npz
```

The comparison is made on **quantised integers**, so the test has no tolerance
of its own. It also verifies that the difference between the paths is an order
of magnitude below the margin to the rounding boundary, and fails if it is not —
even when the digits happen to come out identical, because then the agreement is
accidental. Exit code 1 on any difference.

The repository ships a small frozen model
(`tests/data/surrogate_tiny.npz`, `N = 900`) precisely so this can be run
without the research data, and so CI can run it on an operating system other
than the one that built it: see `.github/workflows/ci.yml` and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## 7. Things that would void the contract

* Editing anything in `features.DEFAULT_CONFIG` or in the feature code — this
  changes `Phi`, hence the model. The guard is `FEATURE_VERSION`
  (`klon-features-2`), stored in the model and checked on load; a mismatch
  raises rather than returning silently wrong vectors.
* Recompressing or rewriting the `.npz` (`np.savez` is used without compression
  for deterministic bytes; the manifest holds the sha256 of the file and of each
  array).
* Changing the neighbour-selection key, the tricube formulation, or the order of
  the final normalisation and quantisation (normalise first, then round).
* An interpreter with a different Unicode version, for non-ASCII text (A8).
