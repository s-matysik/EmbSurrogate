"""Inference from a frozen surrogate — dependency: numpy only.

    python3 -m klon.predict model/surogat.npz --text "umowa najmu lokalu"
    python3 -m klon.predict model/surogat.npz --jsonl data/nowe.jsonl --out V.npy

The model is a single .npz file; there is no reference here to the network, to
the teacher model, or to scipy. Vectors come out normalized and by default
quantized to 6 decimal places.

NOTE ON REPRODUCIBILITY. This path is FAST, not NORMATIVE. Quantization
to 6 places reduces the risk of divergence between machines, but does not eliminate it:
the validation vector has 1200 x 384 coordinates, and the expected number of those
lying closer than eps to the rounding boundary grows linearly with their count
(measured: 936 coordinates closer than 1e-9). The normative definition of the vector
is `klon/reference.py` — computation in integer arithmetic, without
floating-point summation and without functions outside the IEEE-754 standard. Agreement of this
path with the reference is checked by `python3 -m klon.replay_check ... --reference`.
Numbers and the "guaranteed vs measured" distinction: wyniki/kontrakt_wnioski.md."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

from . import dual
from . import features as F
from .freeze import unpack_strings

DECIMALS = 6


class FrozenSurrogate:
    """Loaded .npz file — ready for vector computation."""

    def __init__(self, path: pathlib.Path):
        z = np.load(path, allow_pickle=False)
        self.spec = json.loads(bytes(z["spec"]).decode("utf-8"))
        if self.spec["feature_version"] != F.FEATURE_VERSION:
            raise RuntimeError(
                f"model built on features {self.spec['feature_version']}, "
                f"a kod ma {F.FEATURE_VERSION} — results would be inconsistent")
        self.cfg = self.spec["cfg"]
        self.cfg["char_ngrams"] = tuple(self.cfg["char_ngrams"])
        self.cfg["word_ngrams"] = tuple(self.cfg["word_ngrams"])
        self.idf = z["idf"]
        self.texts = unpack_strings(z["texts"])
        if "indptr" in z.files:
            self.indptr, self.indices, self.data = z["indptr"], z["indices"], z["data"]
        else:
            # features are not stored in the file — we reconstruct them from the texts
            # and
            # verify the checksums recorded at freezing time
            self.indptr, self.indices, self.data = F.csr_arrays(
                self.texts, self.idf, self.cfg)
            exp = self.spec.get("sha256_cech")
            if exp:
                import hashlib
                got = {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
                       for k, v in (("indptr", self.indptr), ("indices", self.indices),
                                    ("data", self.data))}
                if got != exp:
                    raise RuntimeError(
                        "features reconstructed from texts do not match the model — "
                        "different feature code version or corrupted texts")
        self.ids = unpack_strings(z["ids"])
        self.state = {"method": self.spec["method"], "Y": z["Y"], **self.spec["params"]}
        for key in ("alpha", "R", "U", "lam_eig", "z"):
            if key in z.files:
                self.state[key] = z[key]
        self._buf = np.zeros(self.cfg["n_buckets"], dtype=np.float64)

    def kstar(self, texts) -> np.ndarray:
        """Similarities query x training, calculated without scipy."""
        n_tr = len(self.indptr) - 1
        out = np.empty((len(texts), n_tr), dtype=np.float64)
        for i, t in enumerate(texts):
            idx, val = F.hashed_row(t, self.cfg)
            val = val * self.idf[idx]
            nrm = np.sqrt(float(val @ val))
            if nrm > 0.0:
                val = val / nrm
            out[i] = dual.csr_matvec_numpy(
                self.indptr, self.indices, self.data, idx, val,
                self.cfg["n_buckets"], self._buf)
        return out

    def encode(self, texts, decimals: int | None = DECIMALS) -> np.ndarray:
        Y = dual.predict_dual(self.state, self.kstar(list(texts)))
        if decimals is None:
            return Y
        # Quantization BEFORE the final normalization would be an error: we normalize,
        # then round — the same as the export of teacher vectors in the repo
        return np.round(Y, decimals)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=pathlib.Path)
    ap.add_argument("--text", action="append", default=[])
    ap.add_argument("--jsonl", type=pathlib.Path)
    ap.add_argument("--field", default="text")
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--decimals", type=int, default=DECIMALS)
    args = ap.parse_args(argv)

    texts = list(args.text)
    if args.jsonl:
        with args.jsonl.open(encoding="utf-8") as f:
            texts += [json.loads(l)[args.field] for l in f if l.strip()]
    if not texts:
        ap.error("provide --text or --jsonl")

    m = FrozenSurrogate(args.model)
    V = m.encode(texts, decimals=args.decimals)
    if args.out:
        np.save(args.out, V)
        print(f"{V.shape} -> {args.out}")
    else:
        for t, v in zip(texts, V):
            print(json.dumps({"text": t[:60], "embedding": [round(float(x), args.decimals)
                                                            for x in v[:8]] + ["..."]},
                             ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
