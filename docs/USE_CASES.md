# What a researcher actually does with this

Every number below is measured, and each says which regime it belongs to. The
package is small; knowing which of these four situations you are in is most of
using it correctly.

## 1. You are submitting a review that used embeddings

Run `archive` once, while the model you used is still installed, and put the
bundle in the supplementary material. For a review of 3,345 abstracts the
vectors alone are 10.3 MB and the whole bundle is 14.4 MB, so the extra cost of
being reproducible is about four megabytes. If you deposit the vectors
separately, `--link-vectors` points at them instead of copying: 10.6 MB.

What that buys: anyone can re-run your selection and get your vectors back,
exactly at the archival precision in the manifest, and `verify` says so with an
exit code rather than a promise.

If your corpus may not be redistributed, which is the normal case for abstracts
from the major bibliographic databases, run `deposit` instead. It keeps
identifiers, digests and a fixed matrix, no source text, and at 13.3 MB is both
smaller and more faithful than the text-carrying bundle (rank correlation 0.8006
against 0.7711); keeping the residual correction raises that to 0.8105 at
25.6 MB. The replicator fetches the abstracts from the licensed source under
their own rights, exactly as they already re-run your search strategy.

## 2. You are the reviewer or the reader

```
embsurrogate verify --bundle bundle/ --strict
```

Checks every digest, rebuilds the features from the texts, confirms the
reference path, exits non-zero on any difference. Then `replay` reproduces the
vectors and writes a provenance file saying, per record, whether it was stored
or approximated and by which route. You do not have to trust the author's claim
about which records are exact; the file says.

## 3. You are updating the review a year later

This is the only situation the surrogate exists for. New records get vectors on
the same scale as the archived ones without the original model. What to expect,
all measured on a held-out set:

| route | rank correlation with the withdrawn model |
|---|---|
| any encoder still runs and suits your language | 0.83, two encoders 0.85 |
| no encoder, dependency-free n-grams | 0.81 at 43,093 pairs, 0.77 at 8,000 |
| model licence allows depositing weights (int8) | 0.9999 at 135 MB |

`archive` fits both of the first two on held-out records and keeps the encoder
route only when it wins on *your* corpus, recording in the manifest why it was
refused when it loses. A mismatched encoder is worse than none: an English model
on a Polish corpus gives 0.631.

**One thing to get right.** Do not mix stored and approximated vectors in one
ranking. Measured over five reviews, archived records fall from 65.9% of the top
of the list to 22.5% when the two populations meet, because the regression pulls
vectors toward the corpus centroid and a screening query is itself a centroid.
Approximating everything restores 61.9% and gives higher sensitivity
(0.598 against 0.478). `replay --mode consistent` does this; the default warns
when a run mixes populations.

## 4. You are screening the new records

Use `select`, not a one-shot ranking. Over five Cochrane reviews, sensitivity at
a 10% reading budget is 0.64 for the original model, 0.67 for the surrogate and
0.69 for a lexical baseline with no model at all; reaching 95% of the relevant
papers takes reading 34.7%, 32.4% and 31.0% of the pool respectively. Two
consequences worth stating plainly:

* the labelling loop matters more than the representation, so a tool that ranks
  once and stops is leaving most of the saving on the table;
* the lexical baseline is not worse here, and `select` computes it in the same
  call, so you can see for your own corpus whether the embedding step earns its
  place rather than assuming it does.

Seeds beat questions: a one-shot ranking from the review question alone reaches
0.32 and 0.36; from the centroid of 25 already-included papers, 0.44 and 0.52.

## When not to use this

* **The corpus has not changed.** Archive the vectors; reproduction is exact and
  free, and the surrogate adds nothing.
* **You may redistribute the model weights.** Deposit them in int8 and get
  0.9999. The surrogate is for when you cannot.
* **You need per-record decisions at a low false-positive rate.** At 10^-4 false
  positives the surrogate recovers about one relevant pair in nine. It
  reproduces rankings and topic structure, not individual screening decisions.
* **You need paraphrase, negation or word order.** The dependency-free route is
  a lexical-similarity machine; it reads "no evidence of benefit" much as it
  reads "evidence of benefit".

## Outside literature reviews

Nothing in the method knows what a review is. It needs a corpus with stored
vectors that keeps growing: patent portfolios, e-discovery, bibliometrics,
grant-portfolio analysis. The caveat is honest rather than formal: fidelity was
measured on biomedical abstracts and Polish Wikipedia, so for another domain the
number you should trust is the one `archive` measures on your own corpus and
writes into the manifest.

## Trying it on a small sample first

Before committing a real corpus, run the whole cycle on a few hundred texts:

```
bash examples/try_it.sh corpus.jsonl 600
bash examples/try_it.sh corpus.jsonl 600 sentence-transformers/all-MiniLM-L6-v2
```

It takes the first 600 records, holds the last 10% out, computes teacher
vectors, builds a bundle, verifies it, replays both populations and prints the
fidelity `archive` measured on your own corpus. A real run on 540 archived plus
60 held-out Wikipedia entries, with `all-MiniLM-L6-v2` as the teacher:

| what | result |
|---|---|
| bundle | 2.9 MB (`manifest.json`, `pairs.npz`, `surrogate.npz`) |
| `verify --strict` | all seven checks pass, reference agreement 5.6e-17 |
| replay of the 540 archived texts | 540 exact, 0 approximated, `fully_reproducible: true` |
| replay of the 60 held-out texts | 60 approximated, route recorded per record |
| fidelity, n-gram route | rank correlation 0.41, median cosine 0.78 |
| fidelity, with an encoder named at archive time | rank correlation 0.59 |
| text-free deposit | 2.1 MB, `contains_source_texts: false` |

**Read those two fidelity numbers as a property of the sample size, not of the
method.** At 540 pairs the dependency-free route is far below the 0.81 it
reaches at 43,093, exactly as the scaling law says it must be; the encoder route
is much less sensitive to N, which is why it wins here by 0.18 and `archive`
keeps it. What a small sample does test, and tests completely, is the part that
either works or does not: the checksums, the exact lookups, the honest labelling
of every approximated record, and whether the fidelity your corpus actually
yields is enough for what you intend to do with it.
