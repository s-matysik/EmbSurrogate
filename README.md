# embsurrogate — a deterministic, archivable surrogate of a text-embedding model

`embsurrogate` reconstructs the map *text → vector* of a sentence-embedding
model from nothing but **pairs (text, vector)** produced by that model, and
freezes the reconstruction into a single file that can be replayed years later
with numpy as the only dependency.

The purpose is **archival**, not compression or theft of a model: when the
version of the teacher model used to build a corpus index disappears — a
Hugging Face revision is deleted, an API endpoint is retired, a vendor changes
a "stable" model in place — the vectors already stored in the index remain, but
there is no longer any way to embed a *new* document into the same space. A
frozen surrogate restores that ability: it is a deterministic function that
maps new text into the old vector space with measured, and deliberately
modest, fidelity.

* No access to the teacher's weights, gradients, tokenizer or logits is
  assumed — only its inputs and outputs.
* Inference from a frozen model imports **numpy only**; scipy is needed to
  *fit* a surrogate, never to *read* one.
* Every step that could depend on the machine (hash seeds, summation order,
  neighbour ties, transcendental functions) is pinned, and the contract is
  enforced by a test suite plus an integer-arithmetic reference implementation.

---

## Table of contents

1. [Problem](#problem)
2. [What the package computes](#what-the-package-computes)
3. [Installation](#installation)
4. [Quick start](#quick-start)
5. [Measured fidelity](#measured-fidelity)
6. [How fidelity scales with the number of pairs](#how-fidelity-scales-with-the-number-of-pairs)
7. [API](#api)
8. [Command-line tools](#command-line-tools)
9. [Repository layout](#repository-layout)
10. [Determinism and testing](#determinism-and-testing)
11. [Limitations](#limitations)
12. [Citing](#citing)
13. [License](#license)

---

## Problem

A retrieval system built on sentence embeddings stores two things: the vectors
of its documents and, implicitly, the *function* that produced them. The
vectors are durable — they are just numbers in a file. The function is not: it
lives in a model checkpoint whose availability, revision pinning and numerical
behaviour are outside the archive's control. When the function is gone, the
stored index becomes read-only: existing documents can still be compared with
each other, but nothing new can be added to the same space, so the index cannot
be extended, re-chunked or audited against new material.

This package addresses the narrow, well-posed part of that problem: given a
sample of the teacher's behaviour — pairs (text, vector) collected while the
teacher was still available — fit a *surrogate* function that (i) is
deterministic and dependency-light enough to survive as an archival artefact,
and (ii) reproduces the teacher's **geometry** well enough to keep rank-based
operations meaningful. How well "well enough" is, is measured below and is not
oversold: a surrogate vector is not a drop-in replacement for a teacher vector.

## What the package computes

Three components, each chosen for determinism first and accuracy second:

**Features.** `Phi: text → R^D`, `D = 2^18 = 262144`. Character n-grams of
length 3 and 4 over the NFKC-normalised, case-folded, digit-folded text with
explicit word boundaries, hashed with **SHA-256** (not Python's randomised
`hash()`), signed hashing from the last bit of the same digest, sublinear
tf-idf weights, L2 normalisation. The idf table is the only data-dependent part
and is frozen into the model file. Because rows are L2-normalised, the linear
kernel on features *is* the cosine.

**Estimator.** Kernel ridge regression in **dual form**: a prediction is a
combination of the *training vectors*, with coefficients
`alpha = (K + lambda·N·I)^-1 Y`, so the model stores an `N × d` coefficient
matrix rather than a `D × d` primal matrix (which would be ~1 GB at `D = 2^18`).

**Residual correction (`hybrid_z`).** The ridge trend is corrected by a locally
weighted average of training residuals with a tricube kernel — but the
neighbourhood is searched in the **output space**, using the first-stage
prediction, not in n-gram space. In n-gram space two unrelated texts sit at
cosine 0.2–0.3, so "neighbourhood" there is largely fictitious; after the first
stage, neighbours sit at cosine 0.6–0.8. The training coordinates for that
search are **leave-one-out** predictions (`y_i − e_i`), not in-sample ones: at
small `lambda` the in-sample prediction is nearly `y_i`, a situation a test text
never enjoys, and the correction would otherwise learn the leak.

The mathematics, including why each variant was rejected, is in
[docs/METHOD.md](docs/METHOD.md).

## Installation

Python 3.11 or 3.12.

```bash
# inference only (numpy)
pip install .

# fitting new surrogates (adds scipy)
pip install ".[train]"

# running the test suite
pip install ".[test]" && pytest -q

# full research pipeline: grid searches, teacher encoding, figures
pip install ".[experiments]"
```

From the repository root; the package is also importable from `src/` without
installation (`PYTHONPATH=src`). The published source is at `https://github.com/s-matysik/EmbSurrogate`.

## Quick start

Ten lines, run against the small frozen model that ships with the tests:

```python
import numpy as np
from embsurrogate.predict import FrozenSurrogate

model = FrozenSurrogate("tests/data/surrogate_tiny.npz")  # numpy only, no network
V = model.encode(["umowa najmu lokalu użytkowego",        # new, unseen texts
                  "kernel ridge regression in dual form"])
print(V.shape, np.linalg.norm(V, axis=1))                 # (2, 384) [1. 1.]

Q = model.encode(["lease of commercial premises"])        # query the archive
sims = Q @ np.asarray(model.state["Y"]).T                 # against teacher vectors
print(model.ids[int(sims.argmax())], float(sims.max()))   # nearest archived text
```

`encode` returns L2-normalised vectors quantised to 6 decimals — that
quantisation *is* the archival unit (see
[docs/DETERMINISM.md](docs/DETERMINISM.md)); pass `decimals=None` for the raw
float64.

Two runnable scripts:

* [`examples/quickstart.py`](examples/quickstart.py) — load the frozen model,
  embed new texts, retrieve from the archive, verify reproducibility.
* [`examples/fit_from_pairs.py`](examples/fit_from_pairs.py) — a complete
  end-to-end run on synthetic pairs: fit → evaluate → freeze → reload → replay,
  with no external data or network access.
* [`examples/build_tiny_fixture.py`](examples/build_tiny_fixture.py) — how the
  test fixture in `tests/data/` was built from a teacher dataset.

## Measured fidelity

All numbers below are transcribed from `wyniki/v3/metryki.json` and
`wyniki/krzywa_uczenia_T0.csv` of the study that accompanies this package (the
research outputs themselves are not part of the repository — see
[Repository layout](#repository-layout)).

Setup: teacher `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`,
revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, `d = 384`, normalised
outputs; **N = 43 093** training pairs of multilingual encyclopaedic prose;
`lambda = 1e-4`, `k = 100` neighbours in the correction. Two disjoint held-out
sets of 1200 texts each: a *validation* set, on which hyper-parameters were
chosen, and **T0**, a fresh test set used once. T0 is the column to read.

| Metric (T0, 1200 fresh texts) | `hybrid_z` | `ridge` only |
|---|---|---|
| cosine to the teacher vector, median | **0.8121** | 0.7986 |
| cosine after removing the mean vector, median | 0.7727 | — |
| R² (fraction of variance explained) | **0.5348** | — |
| Spearman rho of pairwise similarities (all pairs) | **0.8075** [0.7941, 0.8204] | 0.7808 |
| Spearman rho per query, median | 0.7967 | — |
| recall@10 (shared nearest neighbours) | **0.4402** | 0.4217 |
| NDCG@10 (relevance = teacher cosine) | 0.8698 | — |
| Pearson r of pairwise similarities | — | 0.8136 |
| MAE of the reconstructed cosine | — | 0.1357 |
| cosine 5th / 95th percentile | — | 0.6246 / 0.9116 |
| cosine minimum | — | 0.4121 |

Detection of *related pairs*, i.e. asking the surrogate to decide whether two
texts are similar according to the teacher:

| Threshold on the teacher cosine | share of positives | average precision | sensitivity at FPR = 1e-4 |
|---|---|---|---|
| > 0.6 | 0.61 % | 0.6015 | **0.1125** |
| > 0.8 | 0.030 % | 0.3413 | 0.3488 |

The 95 % confidence interval for global rho comes from a bootstrap over
**texts** (not pairs — pairs are not independent). On the validation set the
same configuration gives rho = 0.8050, recall@10 = 0.4298, median cosine
0.8008, R² = 0.5215, so the fresh-test numbers are not an artefact of
selection. The number of neighbours was chosen on validation
(rho: k=20 → 0.7930, k=50 → 0.8037, **k=100 → 0.8050**, k=200 → 0.8031).

A floor is worth keeping in mind when reading the median cosine: assigning
*every* text the mean training vector already scores ≈ 0.35, which is why the
centred cosine and R² are reported alongside it.

## How fidelity scales with the number of pairs

Global Spearman rho on T0 as the training pool grows
(`wyniki/krzywa_uczenia_T0.csv`):

| N (training pairs) | rho on T0 | k |
|---|---|---|
| 4 800 | 0.7501 | 20 |
| 9 600 | 0.7740 | 20 |
| 16 801 | 0.7868 | 20 |
| 25 622 | 0.7929 | 20 |
| 43 093 | **0.8075** | 100 |

A power law fitted to the last three points of that table gives
`1 − rho = 0.624 · N^(−0.110)`; over all five points,
`1 − rho = 0.653 · N^(−0.114)` (R² = 0.989 in log-log). Either way the exponent
is around −0.11: **an order of magnitude more pairs buys roughly 0.05 of rho.**
Extrapolating the three-point fit, `N = 10^6` pairs would reach rho ≈ 0.86, and
rho = 0.9 is decades of corpus away. The practical reading is that the ceiling
is set by the feature space, not by the amount of supervision; a note in
`docs/METHOD.md` records what an alternative feature space buys.

> Note on provenance: the accompanying Polish method report quotes
> `1 − rho = 0.513 · N^(−0.090)` for this fit. That constant pair does not
> reproduce from the learning-curve files as archived (no window of
> `krzywa_uczenia_T0.csv` or `krzywa_uczenia_A1.csv` yields it); the fits above
> were recomputed from the saved tables and should be used.

## API

```python
import embsurrogate as es
```

**Archival inference (numpy only)**

| Object | Purpose |
|---|---|
| `embsurrogate.predict.FrozenSurrogate(path)` | load a frozen `.npz`; validates the feature version and, if features are not embedded, re-derives them from the archived texts and checks their sha256 |
| `FrozenSurrogate.encode(texts, decimals=6)` | text → L2-normalised vectors, quantised; `decimals=None` for raw float64 |
| `FrozenSurrogate.kstar(texts)` | query × training cosine matrix, computed without scipy |
| `FrozenSurrogate.texts`, `.ids`, `.state["Y"]` | the archive itself: training texts, their identifiers, and the teacher vectors stored for them |
| `embsurrogate.features.hashed_row(text, cfg)` | one text → (bucket indices, tf values) |
| `embsurrogate.features.csr_arrays(texts, idf, cfg)` | full feature map without scipy → CSR arrays |
| `embsurrogate.dual.predict_dual(state, Ks)` | the estimator itself, given a kernel block |
| `embsurrogate.reference.ReferenceModel(path)` | integer-arithmetic reference implementation: the normative definition of a vector |

**Fitting (requires scipy)**

| Object | Purpose |
|---|---|
| `embsurrogate.surrogate.Surrogate(method, ...)` | `fit(texts, Y, ids)` / `predict(texts)`; methods `knn1`, `nw`, `loess`, `ridge`, `rbf`, `hybrid`, `hybrid_z` |
| `embsurrogate.features.fit_transform(texts, cfg)` | features + idf in one call |
| `embsurrogate.dual.fit_dual(method, K, Ytr, **params)` | the closed-form solutions, shared by every entry point |
| `embsurrogate.dual.solve_ridge_loo_f32(G32, Ytr, lam)` | float32 in-place Cholesky + LOO predictions for large N |
| `embsurrogate.freeze.build_explicit(...)` | serialise a fitted state into `.npz` + manifest with per-array sha256 |
| `embsurrogate.evaluate.report` / `report_full` | the fidelity metrics used in the tables above |

**Research scripts** (`run_experiment`, `run_big`, `run_primal`,
`teacher_sbert`, `probe_table`, `product_kernel`, `teachers`,
`regularisation`, `figures`) reproduce the study: variant grids, large-N fitting,
teacher encoding, alternative kernels, figures. They expect the research data
layout and additional dependencies (`pandas`, `scikit-learn`, `matplotlib`,
`sentence-transformers`).

**Legacy name.** The package was called `klon` before publication. `import klon`
still works, emits a `DeprecationWarning` and forwards to `embsurrogate`, so
previously frozen models and old scripts — including
`python -m klon.replay_check` — keep running unchanged. Strings stored *inside*
model files (`klon-features-2`, `klon-surogat-2` and the manifest keys) were
deliberately **not** renamed: they are data, and changing them would invalidate
existing archives.

## Command-line tools

```bash
# embed new texts with a frozen model
python -m embsurrogate.predict model.npz --text "umowa najmu lokalu"
python -m embsurrogate.predict model.npz --jsonl new.jsonl --out V.npy

# verify a frozen model: manifest checksums, feature determinism, golden vectors
python -m embsurrogate.replay_check model.npz \
    --probes probes.jsonl --expected expected.npy

# ... and against the integer reference implementation (slower, normative)
python -m embsurrogate.replay_check model.npz \
    --probes probes.jsonl --reference --n-ref 20

# freeze a fitted configuration into an archival file
python -m embsurrogate.freeze data/ds_multi best.json model.npz
```

Installed console scripts: `embsurrogate-predict`,
`embsurrogate-replay-check`, `embsurrogate-freeze`.

## Repository layout

```
src/embsurrogate/     the package (inference path is numpy-only)
src/klon/             deprecated alias of the above, kept for old archives
tests/                pytest suite, including the cross-platform golden check
tests/data/           one small frozen model (13.7 MB) + control texts + golden vectors
examples/             quick start, end-to-end fit, fixture builder
docs/METHOD.md        mathematics and the variants that were rejected
docs/DETERMINISM.md   the bit-for-bit contract: what is guaranteed, what is measured
docs/REPRODUCIBILITY.md  how to rebuild every artefact and every number
.github/workflows/ci.yml  the cross-platform determinism check
```

Deliberately **not** in the repository: corpora and teacher vectors (`data/`),
full-size frozen models (`model/`, 330–560 MB each) and experiment outputs
(`wyniki/`). They are research data, not software; keeping them out is what
holds the repository under 20 MB. `.gitignore` encodes this rule, with an
exception for the small fixture under `tests/data/`.

## Determinism and testing

The claim under test is narrow and falsifiable: *for a given frozen model and a
given text, the vector quantised to 6 decimals is the same on every
IEEE-754 machine.* The suite (`pytest -q`, 25 tests) covers:

* feature determinism within one process and across two processes;
* independence from `PYTHONHASHSEED` (0 and 12345) — hashing is SHA-256, so the
  interpreter's randomised `hash()` never enters;
* equality of the scipy and numpy feature paths, bit for bit;
* neighbour **tie resolution**: an artificial case with exactly equal
  similarities must select by the documented key `(-similarity, index)`, and the
  test is shown to be sensitive to swapping one tied neighbour for another;
* agreement of the fast numpy path with `reference.py`, the integer
  implementation, compared on quantised integers (no tolerance of its own);
* the **golden test**: the frozen fixture must reproduce
  `tests/data/expected_tiny.npy` exactly;
* refusal to load a model built on a different `feature_version`
  (`pytest.raises`).

CI runs that suite on `ubuntu-latest` and `macos-latest` with Python 3.11 and
3.12, prints the numpy/scipy/OS/BLAS versions it used, and re-runs the golden
check and the integer-reference check as separate steps. The fixture was built
on macOS/arm64, so a green Linux job is a genuine cross-platform verification
rather than a self-check.

What is *guaranteed* versus merely *measured* is spelled out in
[docs/DETERMINISM.md](docs/DETERMINISM.md). In short: quantisation to 6
decimals does not by itself prove cross-platform equality — it lowers the
probability of a mismatch, and the margin to the nearest rounding boundary is
what bounds the risk (measured 1.23e-12 against a path difference of 1.14e-13,
an 11-fold margin, on the 25 622-pair model). The guarantee comes from the
integer reference implementation, of which the numpy path is a fast, measurably
faithful approximation.

## Limitations

Read this section before using the package for anything.

1. **A surrogate vector cannot be inserted into an index built by the teacher.**
   The median cosine to the teacher vector is **0.81** on fresh texts. Vectors
   in a retrieval index are routinely separated by less than that, so mixing
   surrogate vectors into a teacher-built index silently changes what the index
   means. Use the surrogate to rebuild or to query, not to top up.
2. **Precise similarity decisions are out of reach.** At a decision threshold
   of teacher cosine 0.6, sensitivity at FPR = 1e-4 is **0.11**: nine out of ten
   genuinely related pairs are missed if false positives must be that rare.
   Ranking is usable (rho ≈ 0.81, NDCG@10 ≈ 0.87); thresholding is not.
3. **For texts already archived, use the stored vectors, not the surrogate.**
   The teacher's own vectors sit in the frozen file (`state["Y"]`, aligned with
   `.texts` and `.ids`); look them up by id. `encode()` always recomputes a
   *surrogate* vector, including for training texts, so calling it on an
   archived text throws away exact information for an approximation.
4. **Only half the variance is explained** (R² = 0.53), and the median cosine is
   inflated by a common component shared by all sentence embeddings — after
   removing the mean vector it falls to 0.77. A trivial baseline that returns
   the mean training vector already reaches ≈ 0.35.
5. **recall@10 ≈ 0.44**: fewer than half of the teacher's ten nearest
   neighbours appear among the surrogate's ten.
6. **More data will not fix this.** The exponent of the scaling law is ≈ −0.11;
   the limit is the n-gram feature space, not the number of pairs.
7. **Domain and language follow the pairs.** The reported numbers come from
   multilingual encyclopaedic prose with a 384-dimensional teacher whose
   `max_seq_length` is 128. Nothing here transfers automatically to code,
   chat logs, or long documents; refit and re-measure.
8. **Model files are large and their size grows linearly with N.** A frozen
   model stores the training texts, teacher vectors, dual coefficients,
   residuals and LOO coordinates: ~330 MB at N = 25 622, ~560 MB at
   N = 43 093. The fixture in this repository is a deliberately small one
   (N = 900, 13.7 MB).
9. **Inference is not free**: ~0.1 s per text against a 25 622-point model on one
   core (the integer reference path: ~0.5 s), because every query is compared
   with every training point.
10. **Legal and ethical scope.** The method needs only inputs and outputs of the
    teacher, which is exactly what many terms of service restrict. This package
    is intended for archiving *your own* embedding spaces, where you are
    entitled to the teacher's outputs; it is not a licence-laundering device.

## Citing

If you use this software, please cite the software itself and the accompanying
article. Both the article DOI and the software DOI are to be filled in on
acceptance:

```bibtex
@article{Matysik2026embsurrogate,
  author  = {Matysik, Sebastian},
  title   = {embsurrogate: a deterministic, archivable surrogate of a
             text-embedding model fitted from input--output pairs},
  journal = {SoftwareX},
  year    = {2026},
  doi     = {<DOI>},
  url     = {https://github.com/s-matysik/EmbSurrogate}
}
```

Machine-readable metadata: [`CITATION.cff`](CITATION.cff) (same placeholders).

## License

MIT — see [`LICENSE.txt`](LICENSE.txt). Copyright (c) 2026 Sebastian Matysik.
