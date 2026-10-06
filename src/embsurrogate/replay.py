"""Reproduce vectors from a bundle, after the embedding model is gone.

    embsurrogate replay --bundle bundle/ --texts corpus.jsonl \
        --out vectors_replayed.npy --provenance provenance.csv

Two cases, and the whole point of this module is that it never silently mixes
them:

* **the text was archived** -- the vector returned is the one the model
  produced, read from the bundle, identical bit for bit. If the raw data is the
  same, every vector is exact and any selection computed from them is exactly
  the original selection. No approximation is involved and no fidelity caveat
  applies.
* **the text is new** -- the vector is produced by the surrogate, and the
  fidelity measured when the bundle was built (recorded in its manifest) is what
  you get. The record is marked as such in the provenance file.

Matching is on the exact text after the same normalisation the features use, so
a record that differs by a stray whitespace is treated as new rather than
silently matched to a different text. The provenance file has one row per input
record with `exact` or `surrogate`, which is the audit trail a reviewer can
check: a replay that claims full reproducibility must have zero `surrogate`
rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

from . import archive as A
from . import features as F
from .freeze import unpack_strings


class Bundle:
    """An archive bundle: stored pairs plus the fitted surrogate."""

    def __init__(self, path: pathlib.Path, verify: bool = True,
                 vectors: pathlib.Path | None = None):
        self.path = pathlib.Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text(
            encoding="utf-8"))
        if self.manifest.get("format") != A.BUNDLE_FORMAT:
            raise SystemExit(f"unknown bundle format: {self.manifest.get('format')}")
        fv = self.manifest["feature_version"]
        from .predict import SUPPORTED_FEATURE_VERSIONS
        if fv not in SUPPORTED_FEATURE_VERSIONS:
            raise SystemExit(
                f"bundle built on features {fv}, this code implements "
                f"{sorted(SUPPORTED_FEATURE_VERSIONS)} — results would differ")
        if verify:
            for name in ("pairs.npz", "surrogate.npz"):
                got = hashlib.sha256((self.path / name).read_bytes()).hexdigest()
                if got != self.manifest["sha256"][name]:
                    raise SystemExit(f"{name} does not match its checksum — "
                                     "the bundle is damaged or was modified")
        p = np.load(self.path / "pairs.npz", allow_pickle=False)
        s = np.load(self.path / "surrogate.npz", allow_pickle=False)
        self.ids = unpack_strings(p["ids"])
        self.texts = unpack_strings(p["texts"])
        # Vectors are stored as integers at the archival unit, or linked by
        # checksum to the deposit. Either way they are converted here and used
        # only for exact lookups; the prediction path does not touch them.
        if "Y_micro" in p.files:
            self.Y = p["Y_micro"].astype(np.float64) / 1e6
        elif vectors is not None:
            Yl = np.load(vectors).astype(np.float64)
            got = hashlib.sha256(np.ascontiguousarray(
                np.round(Yl * 1e6).astype(np.int32)).tobytes()).hexdigest()
            if got != self.manifest["sha256"]["vectors"]:
                raise SystemExit("the supplied vector file is not the one this "
                                 "bundle was built from (checksum differs)")
            self.Y = np.round(Yl, 6)
        else:
            raise SystemExit(
                "this bundle links its vectors instead of storing them; pass "
                "--vectors with the file named in the manifest")
        self.idf = s["idf"].astype(np.float64)
        self.coef = s["coef"].astype(np.float64)
        self.z = s["z"].astype(np.float64)
        # residuals are a scalar multiple of the coefficients for a solved system
        self.R = float(s["lam_N"]) * self.coef
        self.alpha_mix = float(s["alpha_mix"])
        self.V = s["V"].astype(np.float64) if "V" in s.files else None
        self.cfg = dict(F.DEFAULT_CONFIG)
        for kk, vv in self.manifest["cfg"].items():
            self.cfg[kk] = tuple(vv) if isinstance(vv, list) else vv
        self.k = int(self.manifest["params"]["k"])
        self._index = {F.normalize(t, self.cfg): i for i, t in enumerate(self.texts)}
        self._X = None
        self._table = None
        self._bridge = None
        # Which route produced the surrogate vectors of the LAST call. One run
        # uses one route for every new text, so this belongs to the run rather
        # than to the record, and the provenance file repeats it per row only
        # for the convenience of whoever reads that file alone.
        self._route = "stored"

    # ------------------------------------------------------------------ utils

    def _key(self, text: str) -> str:
        """Matching key: the text after exactly the normalisation the features use.

        Deliberately not a looser match. If a record differs from the archived
        one by anything the featuriser can see, it is a different input and must
        be marked as new, not quietly paired with a vector that belongs to a
        different text.
        """
        return F.normalize(text, self.cfg)

    @property
    def X(self):
        """Training-side n-gram features, recomputed from the archived texts."""
        if self._X is None:
            self._X = F.transform(self.texts, self.idf, self.cfg)
        return self._X

    def bridge(self):
        """Encoder route, if the bundle carries one AND the encoder loads.

        Deliberately silent on failure: a bundle must stay readable on a machine
        where the named model cannot be fetched, which is the situation the
        package exists for. The caller learns which route ran from the
        provenance file, never by having to guess.
        """
        if self._bridge == "none":
            return None
        if self._bridge is None:
            spec = (self.manifest.get("bridge") or {}).get("encoder")
            f = self.path / "bridge.npz"
            if not spec or spec == "callable" or not f.exists():
                self._bridge = "none"
                return None
            try:
                enc = A.encoder_fn(spec)
            except Exception:
                self._bridge = "none"
                return None
            s = np.load(f, allow_pickle=False)
            self._bridge = ({"coef": s["coef"].astype(np.float64),
                             "z": s["z"].astype(np.float64),
                             "Z": s["Z"].astype(np.float64),
                             "R": float(s["lam_N"]) * s["coef"].astype(np.float64),
                             "alpha_bridge": float(s["alpha_bridge"])}, enc, spec)
        return self._bridge if self._bridge != "none" else None

    def table(self):
        if self._table is None and self.manifest.get("static_table"):
            from .static_table import StaticTable
            self._table = StaticTable(self.path / self.manifest["static_table"])
        return self._table

    @property
    def fidelity(self) -> dict:
        return self.manifest["fidelity_new_texts"]

    # ------------------------------------------------------------- reproduce

    def vectors_for(self, texts: list[str], mode: str = "archival"
                    ) -> tuple[np.ndarray, list[str]]:
        """Returns vectors and, per record, whether they are exact or surrogate.

        Two modes, and picking the wrong one is a measurable error rather than a
        matter of taste.

        * `archival` (default) returns the ORIGINAL vector for every archived
          text and a surrogate vector only for texts that are new. Use it when
          the question is what the original pipeline computed: recomputing a
          finished selection, auditing published numbers, re-deriving a figure.

        * `consistent` returns a surrogate vector for EVERY text, archived ones
          included, using the out-of-sample coordinate each archived text would
          have received had it not been in the training set. Use it whenever
          archived and new records are compared with one another -- ranked
          against a query, clustered, deduplicated.

        The reason `archival` is wrong for ranking is measured, not conjectural.
        Ridge shrinks predictions towards the mean direction of the corpus: on
        a review corpus the surrogate vectors sit at cosine 0.78 to the corpus
        centroid against 0.65 for the teacher's. A screening query is itself a
        centroid of seed documents, so approximated records score systematically
        higher against it than exact ones do -- not because they are more
        relevant, but because they are smoother. Mixing the two populations in
        one ranking over five Cochrane reviews pushed the exact records out of
        the top 10%: they held 66% of it under the teacher and 23% under mixed
        vectors, against 62% when the surrogate was applied consistently.
        """
        if mode not in ("archival", "consistent"):
            raise ValueError("mode must be 'archival' or 'consistent'")
        hits = [self._index.get(self._key(t)) for t in texts]
        out = np.zeros((len(texts), self.Y.shape[1]))
        if mode == "consistent":
            prov = ["surrogate-oos" if h is not None else "surrogate"
                    for h in hits]
            new = list(range(len(texts)))
            for i, h in enumerate(hits):
                if h is not None:
                    out[i] = self.z[h]
                    new.remove(i)
            if new:
                out[new] = A.predict([texts[i] for i in new], self.idf,
                                     self.coef, self.R, self.z, self.X, self.V,
                                     self.cfg, self.alpha_mix, self.k,
                                     self.table())
            return out, prov
        prov = ["exact" if h is not None else "surrogate" for h in hits]
        for i, h in enumerate(hits):
            if h is not None:
                out[i] = self.Y[h]
        new = [i for i, h in enumerate(hits) if h is None]
        if new:
            br = self.bridge()
            if br is not None:
                state, enc, spec = br
                out[new] = A.predict_bridge([texts[i] for i in new], self.idf,
                                            state, self.X, self.cfg, enc, self.k)
                self._route = f"encoder:{spec}"
            else:
                out[new] = A.predict([texts[i] for i in new], self.idf,
                                     self.coef, self.R, self.z, self.X, self.V,
                                     self.cfg, self.alpha_mix, self.k,
                                     self.table())
                self._route = "n-gram"
        return out, prov


def run(bundle: pathlib.Path, texts_path: pathlib.Path, out: pathlib.Path,
        provenance: pathlib.Path | None = None, decimals: int | None = 6,
        vectors: pathlib.Path | None = None, mode: str = "archival") -> dict:
    b = Bundle(bundle, vectors=vectors)
    ids, texts = A.read_texts(texts_path)
    Y, prov = b.vectors_for(texts, mode=mode)
    if decimals is not None:
        Y = np.round(Y, decimals)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, Y)
    n_exact = prov.count("exact")
    if provenance is not None:
        import csv
        with provenance.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "source", "route"])
            for i, p in zip(ids, prov):
                w.writerow([i, p, "stored" if p == "exact" else b._route])
    report = {
        "records": len(texts),
        "mode": mode,
        "exact": n_exact,
        "surrogate": len(texts) - n_exact,
        "fully_reproducible": n_exact == len(texts),
        "vector_sha256": hashlib.sha256(
            np.ascontiguousarray(Y).tobytes()).hexdigest(),
        "route_for_new_texts": (b._route if b._route != "stored"
                                else "none (no new texts)"),
        "fidelity_for_new_texts": {
            kk: round(float(vv), 4) for kk, vv in b.fidelity.items()
            if kk in ("rho_globalne", "cos_mediana", "recall@10", "R2")},
        # A run that mixes the two populations and then ranks them together is
        # the one case where the default is the wrong choice; say so here rather
        # than leave it to the documentation.
        "warning": ("this run mixes original and surrogate vectors; if these "
                    "records will be ranked, clustered or deduplicated against "
                    "one another, re-run with --mode consistent, because ridge "
                    "shrinkage makes surrogate vectors score higher against a "
                    "centroid query than original ones do")
        if (mode == "archival" and 0 < n_exact < len(texts)) else "",
        "bundle": {"n_pairs": b.manifest["n_pairs"],
                   "teacher": b.manifest.get("teacher", ""),
                   "feature_version": b.manifest["feature_version"]},
    }
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="embsurrogate replay")
    ap.add_argument("--bundle", type=pathlib.Path, required=True)
    ap.add_argument("--texts", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--provenance", type=pathlib.Path, default=None)
    ap.add_argument("--vectors", type=pathlib.Path, default=None,
                    help="the deposited vector file, for a bundle that links "
                         "its vectors rather than storing them")
    ap.add_argument("--mode", choices=("archival", "consistent"),
                    default="archival",
                    help="archival: original vectors for archived texts "
                         "(reproduce what the pipeline computed). consistent: "
                         "surrogate vectors for every text (compare archived "
                         "and new records with one another)")
    ap.add_argument("--raw", action="store_true",
                    help="write raw float64 instead of values rounded to 6 places")
    a = ap.parse_args(argv)
    rep = run(a.bundle, a.texts, a.out, a.provenance,
              decimals=None if a.raw else 6, vectors=a.vectors, mode=a.mode)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0 if rep["records"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
