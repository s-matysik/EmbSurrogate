"""A deposit that carries no source texts.

    embsurrogate deposit --bundle bundle/ --out deposit/ \
        --encoder sentence-transformers/distiluse-base-multilingual-cased-v1

A bundle holds the archived texts, which makes it exact and also makes it
undepositable for most review corpora: abstracts from the major bibliographic
databases are under publisher copyright and their licences forbid
redistribution. A deposit that a repository will not accept is not a deposit,
and the same is true of one the licence will not allow.

The texts turn out to be needed for three things, none of which requires them
in the clear:

* **matching an archived record** -- a record identifier (PMID, DOI) with a
  SHA-256 digest of the normalised text as a version check is enough. Whoever
  reproduces the selection fetches the abstracts from the licensed source under
  their own entitlement, exactly as they already re-run the search strategy;
  the digest then tells them whether they have the same version of the record.
* **the kernel between a new text and the archive** -- this is what the texts
  were really for, and it disappears in primal form. For the encoder route the
  kernel is `Z Z'`, so the ridge prediction `Z_q Z' a` can be written
  `Z_q W` with `W = Z' a`: a matrix of (encoder dimension) x (teacher
  dimension), which does not grow with the corpus. Measured: 512 x 384, 0.8 MB,
  identical to the dual prediction to 3.4e-15.
* **the residual correction** -- needs the out-of-sample coordinates `z` and
  the residuals `R`, which live in the teacher's output space. They are arrays
  of numbers about each record, not its text.

Measured on 8,000 pairs, the text-free deposit is also the more faithful one:
rank correlation 0.8006 for `W` alone and 0.8105 with the correction, against
0.7711 for the n-gram route that needs the texts -- at 13.3 MB against 36.7 MB.
What it costs is the determinism contract: this route calls a named neural
encoder at replay time, so it is reproducible only as far as that encoder is,
which is the very assumption the n-gram route exists to avoid. Deposit both if
the licence allows the texts; deposit this one when it does not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np

from . import archive as A
from . import dual as D
from . import evaluate as E
from . import features as F

DEPOSIT_FORMAT = "embsurrogate-deposit-1"


def _digests(texts, cfg) -> np.ndarray:
    """SHA-256 of each text after exactly the normalisation the features use."""
    return np.array([hashlib.sha256(F.normalize(t, cfg).encode("utf-8")).digest()
                     for t in texts], dtype="S32")


def build(bundle: pathlib.Path, out: pathlib.Path, encoder,
          lam: float = 1e-4, k: int = 100, with_correction: bool = True,
          holdout: float = 0.2, seed: int = 0) -> dict:
    """Writes the text-free deposit from a bundle the author holds locally."""
    from .replay import Bundle
    b = Bundle(bundle)
    texts, Y = list(b.texts), np.asarray(b.Y, dtype=np.float64)
    enc = A.encoder_fn(encoder)
    Z = enc(texts)

    # fidelity for texts that were NOT archived, on the same protocol as the
    # bundle's own number so the two are comparable
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(texts))
    n_ho = max(1, int(holdout * len(texts)))
    ho, tr = perm[:n_ho], perm[n_ho:]
    st_tr = D.fit_dual("hybrid_z", Z[tr] @ Z[tr].T, Y[tr], lam=lam, k=k, degree=0)
    W_tr = Z[tr].T @ st_tr["alpha"]
    b_ho = Z[ho] @ W_tr
    yh = D.l2norm(b_ho)
    if with_correction:
        yh = D.l2norm(b_ho + D._local(D.l2norm(b_ho) @ st_tr["z"].T,
                                      st_tr["R"], k, degree=0))
    measured = {kk: float(vv) for kk, vv
                in E.report_full(Y[ho], yh, Y[tr].mean(0)).items()
                if isinstance(vv, (int, float))}

    st = D.fit_dual("hybrid_z", Z @ Z.T, Y, lam=lam, k=k, degree=0)
    W = Z.T @ st["alpha"]
    out.mkdir(parents=True, exist_ok=True)
    payload = {"W": W.astype(np.float32),
               "ids": np.array(list(b.ids), dtype=object).astype("U64"),
               "sha256_texts": _digests(texts, b.cfg),
               "Y_micro": np.round(Y * 1e6).astype(np.int32)}
    if with_correction:
        payload["z"] = st["z"].astype(np.float16)
        payload["R"] = st["R"].astype(np.float16)
    np.savez_compressed(out / "deposit.npz", **payload)

    manifest = {
        "format": DEPOSIT_FORMAT,
        "contains_source_texts": False,
        "matching": "record id, with sha256 of the normalised text as a "
                    "version check; the texts come from the licensed source",
        "encoder": encoder if isinstance(encoder, str) else "callable",
        "deterministic": False,
        "W_shape": [int(W.shape[0]), int(W.shape[1])],
        "n_records": len(texts),
        "with_correction": bool(with_correction),
        "feature_version": F.FEATURE_VERSION,
        "cfg": {kk: (list(vv) if isinstance(vv, tuple) else vv)
                for kk, vv in b.cfg.items()},
        "fidelity_new_texts": {
            "protocol": f"{int(holdout*100)}% of the corpus held out, seed {seed}",
            "n_holdout": int(n_ho), **measured},
        "sha256": {"deposit.npz": hashlib.sha256(
            (out / "deposit.npz").read_bytes()).hexdigest()},
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


class Deposit:
    """Reads a text-free deposit and embeds texts supplied by the user."""

    def __init__(self, path: pathlib.Path, encoder=None):
        self.path = pathlib.Path(path)
        self.manifest = json.loads(
            (self.path / "manifest.json").read_text(encoding="utf-8"))
        z = np.load(self.path / "deposit.npz", allow_pickle=False)
        self.W = z["W"].astype(np.float64)
        self.ids = [str(x) for x in z["ids"]]
        self.digests = z["sha256_texts"]
        self.Y = z["Y_micro"].astype(np.float64) / 1e6
        self.z = z["z"].astype(np.float64) if "z" in z.files else None
        self.R = z["R"].astype(np.float64) if "R" in z.files else None
        self.cfg = {kk: (tuple(vv) if isinstance(vv, list) else vv)
                    for kk, vv in self.manifest["cfg"].items()}
        self.k = 100
        self._enc = A.encoder_fn(encoder or self.manifest["encoder"])
        self._by_digest = {bytes(d): i for i, d in enumerate(self.digests)}
        self._by_id = {x: i for i, x in enumerate(self.ids)}

    def vectors_for(self, texts, ids=None):
        """Archived records return their stored vector; others go through W."""
        dg = _digests(texts, self.cfg)
        hit = [self._by_digest.get(bytes(d)) for d in dg]
        if ids is not None:
            hit = [h if h is not None else self._by_id.get(str(i))
                   for h, i in zip(hit, ids)]
        out = np.zeros((len(texts), self.W.shape[1]))
        prov = ["exact" if h is not None else "surrogate" for h in hit]
        for i, h in enumerate(hit):
            if h is not None:
                out[i] = self.Y[h]
        new = [i for i, h in enumerate(hit) if h is None]
        if new:
            bq = self._enc([texts[i] for i in new]) @ self.W
            yh = D.l2norm(bq)
            if self.z is not None:
                yh = D.l2norm(bq + D._local(yh @ self.z.T, self.R, self.k,
                                            degree=0))
            out[new] = yh
        return out, prov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="embsurrogate deposit")
    ap.add_argument("--bundle", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--no-correction", action="store_true")
    a = ap.parse_args(argv)
    man = build(a.bundle, a.out, a.encoder,
                with_correction=not a.no_correction)
    mb = sum(p.stat().st_size for p in a.out.iterdir() if p.is_file()) / 1e6
    f = man["fidelity_new_texts"]
    print(json.dumps({"deposit": str(a.out), "MB": round(mb, 2),
                      "contains_source_texts": False,
                      "W_shape": man["W_shape"],
                      "rho": round(f["rho_globalne"], 4),
                      "cos_median": round(f["cos_mediana"], 4)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
