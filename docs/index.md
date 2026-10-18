---
title: embsurrogate
---

# embsurrogate

A systematic review ranks ten thousand abstracts with a text-embedding model and
reports the model's name. That is enough to describe the analysis and not enough
to repeat it. **embsurrogate** turns the embedding step into a deposit built once
while the model still exists: archived records return their stored vector exactly
at the archival precision the manifest names, new records are embedded by an
explicit formula fitted on the archived pairs alone, and every record is labelled
exact or approximate.

```bash
pip install embsurrogate
embsurrogate archive --texts corpus.jsonl --vectors vectors.npy --out bundle/
embsurrogate verify  --bundle bundle/ --strict
```

## Start here

| If you want to | Read |
|---|---|
| know what to run, in your own situation | [Use cases](USE_CASES.html) |
| understand what the surrogate computes and why | [Method](METHOD.html) |
| check what is guaranteed and what is only measured | [Determinism contract](DETERMINISM.html) |
| reproduce the numbers in the paper | [Reproducibility](REPRODUCIBILITY.html) |

## Four situations, one decision

Which of these you are in decides almost everything about how the package should
be used, and `archive` measures the choice on your own corpus rather than
asserting it:

| Situation | What to do | Fidelity |
|---|---|---|
| the corpus has not changed | archive the vectors; replay returns them | exact |
| new records, a suitable encoder still runs | map from that encoder into the archived space | rank correlation 0.83 to 0.85 |
| new records, no encoder available | the dependency-free n-gram route | 0.81 |
| the model licence permits redistributing weights | deposit the weights in int8 | 0.9999 |

The n-gram route is in every bundle as the fallback, and it is the only one the
determinism contract covers, because a live neural encoder is not bit-reproducible
across versions, which is the premise of the whole problem.

## What the package will not do

* reproduce screening decisions one by one at a low false-positive rate: at a
  false-positive rate of 1e-4 the surrogate recovers about one relevant pair in nine;
* handle paraphrase, negation or word order on the dependency-free route, which is
  a lexical-similarity machine;
* replace archiving the vectors. It contains them.

## Source and citation

Code: [github.com/s-matysik/EmbSurrogate](https://github.com/s-matysik/EmbSurrogate)
(MIT). Continuous integration verifies the determinism contract on six
combinations of operating system and interpreter, reading a bundle built
elsewhere. Citation metadata is in `CITATION.cff` at the repository root.
