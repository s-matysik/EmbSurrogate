---
title: Method
---

# Method

What the surrogate computes, why each component has the form it has, and which
alternatives were measured and rejected. Notation: `t` is a text, `d` the
teacher's output dimension (384 in the reported study), `N` the number of
training pairs, `D = 2^18` the number of feature buckets.

The estimator is implemented once, in `embsurrogate/dual.py`, and every entry
point (`Surrogate`, `run_experiment`, `run_big`, `FrozenSurrogate`) calls that
one implementation, so no two code paths can drift apart.

---

## 1. Problem statement

Given pairs `(t_i, y_i)`, `i = 1..N`, with `y_i = teacher(t_i)` and
`||y_i|| = 1`, find `f` such that `f(t) ≈ teacher(t)` for unseen `t`, subject to
three constraints that come from the archival purpose rather than from
statistics:

1. `f` must be expressible as a single file plus a short program, with numpy as
   the only run-time dependency;
2. `f` must be a *function* in the strict sense, the same text must give the
   same vector on every machine and in every year (see
   [DETERMINISM.md](DETERMINISM.md));
3. the fit may use the teacher's inputs and outputs only, no weights, no
   gradients, no tokenizer, no logits.

Only the teacher's *direction* matters (sentence encoders return normalised
vectors and are consumed through cosine similarity), so `f` returns
L2-normalised vectors and is evaluated with cosine-based measures.

## 2. Feature map

`Phi: text → R^D` is a fixed, unlearned map (`embsurrogate/features.py`):

1. **Normalisation**: NFKC, `casefold()` (not `lower()`: it handles German ß
   and Greek final sigma correctly), every digit folded to `0`, whitespace
   collapsed, input truncated at 20 000 characters.
2. **Features**: character n-grams of length 3 and 4 over the normalised text
   with word boundaries marked explicitly by `\x02`, so a trigram spanning the
   start of a word differs from the same characters inside a word.
3. **Hashing**: `SHA-256(feature)`; the first 8 bytes give the bucket
   (`(h >> 1) mod D`) and the last bit gives the sign (signed hashing, which
   cancels the first-order bias from collisions). SHA-256 rather than Python's
   `hash()` because the latter is salted per process (`PYTHONHASHSEED`) and is
   not stable across interpreter versions, it would make the archive
   unreadable.
4. **Weights**: sublinear tf, `1 + log(tf)`, times idf
   `log((1+n)/(1+df)) + 1` computed on buckets (as in scikit-learn), then L2
   normalisation. **The idf table is the only learned part of the feature map**
   and is frozen into the model file.

Two consequences matter. First, since rows are L2-normalised, the linear kernel
`K(t, t') = Phi(t)·Phi(t')` *is* the cosine between the hashed
representations, the kernel needs no separate definition. Second, the tf
accumulation is performed over **sorted** feature keys
(`features._ordered_counts`), removing the only order-dependent step in the map.

**Rejected: richer feature sets.** Adding 5-grams and word 1 to 2-grams measurably
*hurts*. Each additional feature family increases the part of a new text's
vector that lies **outside the span of the training set**: the measured median
of that out-of-span fraction is 0.72 for `chars 3-5 + words 1-2` versus 0.58 for
`chars 3-4`, while the nonlinearity floor contributes only ~12.7 % of the error.
More expressive features therefore buy less than they lose. (Recorded in the
`features.DEFAULT_CONFIG` comments with the measurement they come from.)

## 3. Dual form

With `X` the `N × D` feature matrix, `K = X Xᵀ` (`N × N`), and `Y` the `N × d`
matrix of teacher vectors, kernel ridge regression solves

```
alpha = (K + lambda·N·I)^(-1) Y                     # N x d
f_ridge(t) = normalise( k_*(t)ᵀ alpha ),  k_*(t)_i = Phi(t)·Phi(t_i)
```

solved by Cholesky factorisation, not by forming an inverse
(`dual._solve_ridge`).

Why dual rather than primal: the primal weight matrix would be `D × d`
= 262144 × 384 ≈ 1 GB of float64, while the dual coefficients are `N × d`. Just
as importantly, in dual form kernel ridge, Nadaraya-Watson and LOESS are the
same family differing only in the shape of the weights, which makes the variant
comparison in `run_experiment.py` a fair one.

Variants implemented (`dual.METHODS`): `knn1` (copy the nearest neighbour, a
lower reference), `nw` (Nadaraya-Watson, tricube weights over `k` neighbours),
`loess` (local linear regression in Nyström coordinates), `ridge`, `rbf`
(Gaussian kernel on the angular distance, `exp(-gamma(2-2cos)))`, `hybrid`
(ridge trend + local correction in feature space) and `hybrid_z` (the published
choice, below).

## 4. `hybrid_z`: residual correction in the output space

`hybrid_z` predicts

```
b(t)      = k_*(t)ᵀ alpha                                  # ridge trend
z_i       = normalise( y_i − e_i )                         # LOO prediction for i
s(t)_i    = normalise(b(t)) · z_i                          # similarity in OUTPUT space
f(t)      = normalise( b(t) + Σ_{i∈nbh_k(t)} w_i R_i / Σ w_i )
R_i       = y_i − (K alpha)_i                              # training residual
w_i       = tricube( (1 − s_i) / h ),  h = (1 − s_min)·1.0000001
```

Two design decisions carry the result.

**The neighbourhood is searched after the first stage, in the output space.** In
n-gram space the cosine between two unrelated texts is typically 0.2 to 0.3, so a
`k`-neighbourhood there is mostly noise; after the ridge stage, neighbours sit
at cosine 0.6 to 0.8. The mechanism: texts the first stage maps close together are
texts on which it makes a similar mistake, so their residuals carry information
about the mistake being made on the new text. Proximity in feature space does
not imply that. Measured effect on the fresh test set: global rho rises from
0.7808 (`ridge`) to 0.8075 (`hybrid_z`), and recall@10 from 0.4217 to 0.4402.

**The training coordinates must be leave-one-out.** `e_i = alpha_i / (G^{-1})_ii`
with `G = K + lambda·N·I` gives the LOO residual in closed form, so `z_i` is the
prediction for `t_i` *as if it had not been in the training set*. Using the
in-sample prediction instead makes `z_i ≈ y_i` at small `lambda`: a luxury no
test text has, and the correction then learns the leak rather than the signal.
For large `N` the LOO diagonal is obtained from the Cholesky factor:
`diag(G^{-1})_i = Σ_k (L^{-1})_{ki}²` (`dual.solve_ridge_loo_f32`).

**Tricube.** `tricube(u) = (1 − u³)³` on `[0, 1]`, zero outside, computed as
`t = 1 - u*u*u; t*t*t`: as multiplications, never `u ** 3`. IEEE-754 requires
exact rounding for `+ - * / sqrt fma` but not for `pow`, so `u ** 3` (which numpy
routes to libm) may legally differ by 1 ulp between platforms; measured
difference on the development machine was 1.1e-16 (1 ulp), and the scale
analysis in [DETERMINISM.md](DETERMINISM.md) shows the tricube weight is the
single most sensitive quantity in the whole path.

**Neighbour selection is by an explicit key** `(-similarity, training index)`,
implemented with `np.lexsort`. This is not cosmetic: the measured smallest gap
between the `k`-th and `(k+1)`-th neighbour is 1.99e-16, i.e. a tie at float64
resolution. Before the key was fixed, 44 of 1200 queries had a different
neighbour *order* and 2 of 1200 a different neighbour *set* than the reference
implementation. A different set is a different function, an error of order
1e-3, not 1e-13.

Hyper-parameters were chosen on a validation set disjoint from the test set:
`lambda = 1e-4`; `k = 100` (validation rho: 0.7930 / 0.8037 / 0.8050 / 0.8031
for k = 20 / 50 / 100 / 200).

## 5. Local regression of degree 1 (LOESS) and Nyström coordinates

For `loess` and `hybrid(degree=1)` the local fit needs coordinates, which come
from the eigendecomposition of the kernel matrix (kernel PCA),
`K = U Λ Uᵀ`, with the query mapped in by the Nyström extension
`z_* = k_*ᵀ U Λ^(-1/2)`. `numpy.linalg.eigh` on a symmetric matrix is
deterministic in the ordering of eigenvalues, and the sign of each eigenvector
is fixed explicitly (largest-magnitude coordinate made positive) because the
sign is otherwise arbitrary. The local design matrix is centred **on the query**,
so the intercept of the local fit *is* the prediction at that point.

On this problem degree 1 did not beat degree 0: the published configuration uses
`degree = 0` (a tricube-weighted mean of residuals). The comparison grid is in
`run_experiment.GRID`.

## 6. Fitting at large N

For `N = 43 093` the Gram matrix is 14.8 GB in float64, and
`numpy.linalg.inv` would need a second copy. `dual.solve_ridge_loo_f32` instead
works in float32 (7.4 GB), factorises **in place** (`cho_factor(...,
overwrite_a=True)`), inverts the triangular factor in place (LAPACK `strtri`),
zeroes the unused triangle blockwise (LAPACK leaves input residue there, which
a full-column sum would otherwise pick up), and returns `alpha`,
`diag(G^{-1})` and the LOO predictions in float64. Peak memory is one matrix.

The precision loss is harmless because the matrix is well conditioned:
`G = K + lambda·N·I` with `lambda·N` of order 1 gives a condition number around
1e2 to 1e3, so the float32 relative error (~1e-7) yields ~1e-4 on `alpha`. This
does not weaken the archival contract: `alpha` is **frozen as data**, and the
inference path recomputes in float64 and quantises to 6 decimals. Training
precision decides *which* model is stored, not whether the stored model is
reproducible.

## 7. Evaluation

`embsurrogate/evaluate.py` deliberately separates two things that are often
conflated:

* **pointwise fidelity**: `cos(f(t), teacher(t))`: how close the individual
  vector is;
* **geometric fidelity**: whether relations between texts survive: Spearman rho
  of the pairwise-similarity matrices, recall@k of nearest neighbours, NDCG@k,
  the error in the reconstructed cosine. This is what embeddings are used for,
  and it is the criterion for "the behaviour was cloned".

`report_full` additionally guards three artefacts: (i) the common component that
all sentence embeddings share, hence the centred cosine and R², given that the
mean training vector alone scores ≈ 0.35; (ii) global rho being dominated by
unrelated pairs, hence per-query rho, recall@k and NDCG@k; (iii) AUC being
degenerate under class imbalance at high thresholds, hence average precision
and sensitivity at FPR = 1e-4. Confidence intervals are bootstrapped **over
texts**, because pairs are not independent.

The floor for every comparison is `baseline_mean`: give every text the mean
training vector.

## 8. What limits the fidelity

A ceiling experiment (`run_experiment.py --ceiling`) replaces the n-gram
features with the embeddings of a *different* neural encoder
(`distiluse-base-multilingual-cased-v1`) and runs the same kernel ridge on top.
At `N = 4800` pairs its best configuration (`lambda = 1e-4`) reaches global rho
0.7942, median cosine 0.7596 and recall@10 0.4318 (`wyniki/multi/sufit.csv`;
the `lambda = 1e-5` row is within noise at rho 0.7940, recall@10 0.4335),
against rho 0.7501 for n-gram features at the same `N`
(`wyniki/krzywa_uczenia_T0.csv`). The gap is attributable to the feature space,
not to the regression: with the same estimator, better inputs do better.

Consistently with that, adding pairs helps only slowly, the fitted scaling law
is `1 − rho = 0.624 · N^(−0.110)` on the last three points of the learning curve
(0.653 · N^(−0.114) over all five). The n-gram feature space, not the amount of
supervision, is what holds the fidelity near rho ≈ 0.81.

That law is a property of **this feature space**, not of the method. Measured on
the same split, the encoder route is flat in N, one encoder gives rho 0.8304 at
43 093 pairs, the same as at 8 000; and rises with the number of independent
encoders instead: two give 0.8498 with R² 0.6152, against 0.8075 and 0.5348 for
n-grams at equal N. A non-linear expansion of that space (Nyström, 2000 anchors,
RBF) does not help once the residual correction is present: 0.8453 against
0.8498, the same competition between the two mechanisms seen with the product
kernel.

The map itself matters, and not only its inputs. Keeping a vector index usable
across model versions is established practice, and the adapters that do it
include an orthogonal Procrustes map (Vejendla, *Drift-Adapter*, EMNLP 2025,
doi:10.18653/v1/2025.emnlp-main.805); that setting differs from this one in
that both models can still be run. Measured on the same split, the orthogonality
constraint is expensive: 0.5498 against 0.8061 for the ridge before the residual
correction, 0.8045 against 0.8304 after it, and, the part that decides the
default, Procrustes cannot use a second encoder at all, giving 0.7855 against
0.8498, because `QᵀQ = I` leaves no freedom to weight two descriptions against
each other. The ridge is therefore the shipped map, with Procrustes kept only as
a measured baseline.

Two further directions are implemented but not part of the published
configuration: `probe_table.py` (query the teacher with *single words* to read
its static table directly, then pool, the Model2Vec-style construction, here
built from input/output pairs only) and `product_kernel.py` /
`regularisation.py` (product kernels and alternative regularisation sweeps).

## Size of the artefact

A deposit a repository will not accept is not a deposit, so the size of the
bundle is a property of the method, not a packaging detail. The first bundle
format was 558 MB for 43,093 pairs, of which 529 MB sat in four float64 arrays:
teacher vectors, dual coefficients, residuals, out-of-sample coordinates.

| change | saving | cost |
|---|---|---|
| residuals not stored (`R = lam N a` for a solved system) | 132 MB | none; exact in closed form, 1.7e-5 iterative |
| vectors as `int32` at the archival unit of 1e-6 | 66 MB | none; that precision is the contract already |
| coefficients `float32`, coordinates `float16` | 165 MB | rho, cos and R2 unchanged; recall@10 +0.0001 (see below) |
| texts compressed | 15 MB | none |
| vectors linked to the deposit instead of copied | 48 MB | `replay` then needs `--vectors` |

The third row is the one that had to be measured rather than argued. On the
43,093-pair model rho, the median cosine and R2 are identical to four decimals
across all five variants; recall@10 moves by one unit in the fourth decimal
once the coordinates are stored in float16, which includes the default:

| stored as | rho | median cos | R2 | recall@10 | size |
|---|---|---|---|---|---|
| float64, residuals in file | 0.8075 | 0.8121 | 0.5348 | 0.4402 | 558 MB |
| residuals derived | 0.8075 | 0.8121 | 0.5348 | 0.4402 | 426 MB |
| + coefficients and coordinates float32 | 0.8075 | 0.8121 | 0.5348 | 0.4402 | 226 MB |
| + coordinates float16 | 0.8075 | 0.8121 | 0.5348 | 0.4403 | 193 MB |
| + coefficients float16 | 0.8075 | 0.8121 | 0.5348 | 0.4403 | 160 MB |

float16 coefficients add nothing to that 0.0001, which is why the default is
float32 rather than float16: the conservative choice is free at this scale, and
the measurement says the aggressive one would also have been safe. The prediction
is a weighted sum over tens of thousands of archived vectors, so per-coefficient
rounding averages out; rho is additionally a rank statistic.

What this means in practice, for the review corpus of 3,345 Cochrane records:

| artefact | size |
|---|---|
| `all-MiniLM-L6-v2` checkpoint (the lightest usable encoder) | 87 MB |
| `allenai-specter` checkpoint (encoder for scholarly text) | 420 MB |
| teacher vectors alone | 10.3 MB |
| self-contained bundle | 14.4 MB |
| bundle with vectors linked to the deposit | 10.6 MB |

The bundle is about 1.4x the vector deposit and roughly an order of magnitude
smaller than the lightest model that could have produced it. That ratio is the
reason to deposit a surrogate rather than a checkpoint, and it does not depend
on any fidelity claim.

### A fixed overhead worth knowing about

The bucket weights (`idf`) are 2^18 float32 values, about 1 MB, regardless of
corpus size. For a 43,093-pair bundle that is 0.6% of the file; for a
300-record bundle it is a third of it. The weights are rounded to float32
*before* fitting, so the coefficients are not fitted against one precision and
replayed against another.
