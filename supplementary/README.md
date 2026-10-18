# Supplementary data for the embsurrogate software paper

Everything a referee or a replicator needs to check a number printed in the
manuscript, plus the baseline that makes the determinism claim testable. The
code itself is not here: it lives at https://github.com/s-matysik/EmbSurrogate
and is versioned there. What this record adds is the measurements.

## What is in each file


### fidelity

* `01_fidelity/krzywa_uczenia_T0.csv` : Fig. 2a, scaling law: rho on the single-use held-out set by number of pairs
* `01_fidelity/metryki_v3_43k.json` : Section 3.1 headline metrics at N=43,093
* `01_fidelity/metryki_v3_po_naprawie.json` : the same after the tie-break fix (Section 2.4)
* `01_fidelity/stabilnosc_rozkladu.csv` : Fig. 2b, error decomposition against training size

### encoder route

* `02_encoder_route/wzmocnienie.csv` : Section 3.1: one encoder, two encoders, RBF expansion
* `02_encoder_route/most_alpha.csv` : kernel weight alpha chosen on validation
* `02_encoder_route/prokrustes.csv` : orthogonal Procrustes baseline (Drift-Adapter family)
* `02_encoder_route/alternatywa_inny_model.csv` : mapping from a mismatched encoder

### screening

* `03_screening/przesiew_wiele.csv` : Section 3.2: five CLEF topics, two teachers, two query modes
* `03_screening/tabela_per_temat.csv` : per-topic summary of the above
* `03_screening/aktywne_uczenie.csv` : Section 3.2: the labelling loop
* `03_screening/aktywne_podsumowanie.json` : reading budget to 95% recall

### deposit

* `04_deposit/rozmiary_artefaktow.csv` : Table 2: what a researcher can deposit
* `04_deposit/model_kwantyzacja.csv` : Section 1: int8 weight deposit, rho 0.9999
* `04_deposit/depozyt_bez_tekstow.csv` : Section 3.4: four deposit variants
* `04_deposit/szew.csv` : mixing stored and approximated vectors in one ranking

### quality case

* `05_quality_case/case_jakosc.json` : Section 3.5: 2,000 quality-management abstracts

### determinism

* `06_determinism/kontrakt_precyzja.csv` : Section 2.4: margin to the rounding boundary

### negative results

* `07_negative_results/kompromis_cechy.csv` : feature richness sweep
* `07_negative_results/C1_tokenizer.csv` : features aligned to the teacher tokeniser
* `07_negative_results/jadra_C3_C5.csv` : learned diagonal metric and product kernel
* `07_negative_results/kompozycja_B.csv` : structural features, with the identical-feature control

### teachers

* `08_teachers/nauczyciele.csv` : Fig. 3: three teachers
* `08_teachers/nauczyciele_zgodnosc.csv` : agreement between teachers

### verification baseline

* `tests/data/surrogate_tiny.npz` : a small frozen bundle, 900 training pairs of
  Polish Wikipedia entries, the one the continuous-integration job reads on a
  machine that did not build it
* `tests/data/expected_tiny.npy` : the vectors it must return, to the archival
  precision named in its manifest
* `tests/data/probes_tiny.jsonl` : the 20 control texts those vectors belong to
* `tests/data/surrogate_tiny.manifest.json` : digests and the feature-version
  marker the loader checks

Reproducing the check needs nothing but the package and numpy:

```
pip install embsurrogate
embsurrogate verify --bundle tests/data --strict
```

## What is deliberately absent

The large frozen models of the research runs (the 43,093-pair bundle is 160 MB in
the current format, 532 MB in the first one) are rebuildable from the repository
and the public corpora: `examples/fetch_wikipedia.py` and `examples/fetch_clef.py`
retrieve the texts, `examples/embed_corpus.py` recomputes the teacher vectors, and
`embsurrogate archive` rebuilds the bundle. The quality-management corpus of
Section 3.5 is retrieved from Crossref the same way. Source texts are not
redistributed here for the same reason the paper gives for the text-free deposit:
abstracts from the major bibliographic databases are under publisher copyright.

## Where the verification baseline is

In this copy the baseline is not duplicated: it is the fixture the test suite
already carries, at `tests/data/` in the repository root. The archived copy of
this record (see the paper's data statement) carries it as
`09_verification_baseline/` so that the record stands alone.

## Integrity

`SHA256SUMS.txt` lists a digest for every file above. Verify with
`shasum -a 256 -c SHA256SUMS.txt` from the root of this record.
