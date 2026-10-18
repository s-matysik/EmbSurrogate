---
title: Reproducibility
---

# Reproducibility

Three separable questions, in increasing order of what they require:

| question | needs | time |
|---|---|---|
| **A.** Does the archived function reproduce? | this repository only | seconds |
| **B.** Do the fidelity numbers reproduce? | teacher model + corpus | hours |
| **C.** Does the whole study reproduce? | the above + the research data layout | ~a day |

---

## A. Verify the archived function (no external data)

Everything needed is in the repository: a frozen model of 900 pairs, 20 control
texts held out of it, and the golden vectors.

```bash
pip install ".[test]"
export OMP_NUM_THREADS=2

pytest -q                                      # 63 tests

python -m embsurrogate.replay_check tests/data/surrogate_tiny.npz \
    --probes tests/data/probes_tiny.jsonl \
    --expected tests/data/expected_tiny.npy

python -m embsurrogate.replay_check tests/data/surrogate_tiny.npz \
    --probes tests/data/probes_tiny.jsonl \
    --expected tests/data/expected_tiny.npy --reference --n-ref 20
```

The second command compares the vectors with the file that was produced on the
machine where the fixture was built; the third compares the fast numpy path with
the integer reference implementation. Exit code 1 on any difference. This is the
same sequence CI runs on `ubuntu-latest` and `macos-latest` with Python 3.11 and
3.12, which is what turns "reproducible" from a claim into a test, since the
fixture was built on macOS/arm64.

Rebuilding the fixture from a teacher dataset (it is deterministic, the pool is
the first `--n` pairs in file order, the control texts are the ones immediately
after, no random seed is involved):

```bash
python examples/build_tiny_fixture.py data/ds_multi --n 900 --probes 20
```

`--n 900` is the size that keeps the `.npz` under 15 MB for a 384-dimensional
teacher: the file stores four `N × d` float64 arrays (teacher vectors, dual
coefficients, residuals, LOO coordinates) plus the `2^18`-entry idf table.

## B. Reproduce the fidelity numbers

You need the teacher and a corpus. The teacher used in the study:

```
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
revision e8f8c211226b894fcb81acc59f3b34ba3efd5f42
dim 384, normalised outputs, max_seq_length 128
```

Pin the revision. Teacher output is the input of everything downstream, and an
unpinned model is exactly the failure mode this package exists to survive.

```bash
pip install ".[experiments]"

# 1. pairs (text, vector) from your corpus
python -m embsurrogate.teacher_sbert corpus.jsonl data/ds_multi \
    --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# 2. compare variants on a small split; writes metryki.csv, najlepszy.json,
#    krzywa_uczenia.csv (add --ceiling for the neural-feature reference)
python -m embsurrogate.run_experiment data/ds_multi wyniki/multi

# 3. fit at large N in a fresh process (float32 in-place Cholesky + LOO)
python -m embsurrogate.run_big data/ds_big3 model/surrogate_v3.npz wyniki/v3 \
    --lam 1e-4 --k 100

# 4. freeze a chosen configuration into an archival file + manifest
python -m embsurrogate.freeze data/ds_multi wyniki/multi/najlepszy.json \
    model/surrogate.npz
```

The dataset layout expected by `run_experiment.load_dataset` is a directory
with `meta.jsonl` (one JSON object per line, fields `id` and `text`) and
`vectors.npy` (row `i` is the teacher vector of line `i`), plus an optional
`model_info.json` recording which teacher produced it.

Splits are derived from fixed seeds rather than stored (`run_experiment.split`,
`run_big.indices`), so they are reproducible from the code: the validation set
is the 1200 texts of `split(6000, 0.2, seed=0)`, **T0** is indices
18001 to 19200 (used once, after all hyper-parameter choices), and the training
pool is the remainder, permuted with seeds 1, 2, 3 in nested order so that the
learning-curve subsets are prefixes of the same list.

Published reference values, for comparison (`wyniki/v3/metryki.json`,
`N = 43 093`, `lambda = 1e-4`, `k = 100`): on **T0**, `hybrid_z` gives median
cosine 0.8121, R² 0.5348, global rho 0.8075 (95 % CI 0.7941 to 0.8204), per-query
rho 0.7967, recall@10 0.4402, NDCG@10 0.8698; `ridge` alone gives 0.7986 /
0.7808 / 0.4217. On the validation set: rho 0.8050, recall@10 0.4298.

Expect the *last digits* to move if you use a different BLAS, a different
thread count or a different scipy: the training solver is float32 and its
result is a model, not an archived vector. What must not move is the replay of a
given frozen file, that is question A.

## C. Reproduce the study

Beyond B, the study also contains: alternative kernels
(`product_kernel.py`), regularisation sweeps (`regularisation.py`), a
single-word probe table (`probe_table.py`), a comparison of several teachers
(`teachers.py`), the primal/conjugate-gradient fit (`run_primal.py`) and the
figure script (`figures.py`). These read and write the research directory layout
(`data/`, `model/`, `wyniki/`) that is **not** part of this repository; they are
included so the published numbers can be traced to code, not as a turnkey
pipeline.

## Environment

The reported numbers were produced with:

```
Python 3.12.14, numpy 2.5.3, scipy 1.18.1
Unicode UCD 15.0.0
macOS 15.1, arm64
OMP_NUM_THREADS=2
```

CI additionally covers Python 3.11 and 3.12 on `ubuntu-latest` and
`macos-latest` with whatever numpy/scipy wheels pip resolves, and prints those
versions in the job log before running the checks.

## Determinism caveats worth repeating

* Recomputing features from the archived texts is a *diagnostic*; the frozen
  arrays, with their sha256 in the manifest, are the oracle.
* Quantisation to 6 decimals bounds but does not eliminate cross-platform risk;
  the margin to the rounding boundary is the measure of that risk, and
  `--reference` is what reports it.
* `unicodedata.unidata_version` is the one input that cannot be recovered from
  the model file and is currently not recorded in the manifest, note it
  alongside any model you archive.

Details and measurements: [DETERMINISM.md](DETERMINISM.md).
