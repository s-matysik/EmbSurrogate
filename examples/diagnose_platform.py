"""Locate the stage at which the feature path diverges between platforms.

    python -m examples.diagnose_platform [model.npz]

Prints stored versus recomputed digests of the three CSR arrays, plus the
values that depend on the platform's maths library and on the order of
summation, in hexadecimal float form so two CI logs can be compared directly.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys
import unicodedata

import numpy as np

from embsurrogate import features as F
from embsurrogate.freeze import unpack_strings


def main(argv):
    p = pathlib.Path(argv[0] if argv else "tests/data/surrogate_tiny.npz")
    z = np.load(p, allow_pickle=False)
    spec = json.loads(bytes(z["spec"]).decode("utf-8"))
    cfg = dict(spec["cfg"])
    cfg["char_ngrams"] = tuple(cfg["char_ngrams"])
    cfg["word_ngrams"] = tuple(cfg["word_ngrams"])
    texts = unpack_strings(z["texts"])
    idf = z["idf"]

    print(f"platform={sys.platform} python={sys.version.split()[0]} "
          f"numpy={np.__version__} unicodedata={unicodedata.unidata_version}")

    indptr, indices, data = F.csr_arrays(texts, idf, cfg)
    sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    stored = spec.get("sha256_cech", {})
    for name, arr in (("indptr", indptr), ("indices", indices), ("data", data)):
        s = stored.get(name, "-")
        print(f"{name:8s} stored={s[:16]} recomputed={sha(arr)[:16]} "
              f"match={s == sha(arr)}")
    print(f"structure: rows={len(indptr) - 1} nnz={len(indices)} "
          f"indices_sum={int(indices.sum())} indptr_tail={int(indptr[-1])}")

    # Which operations on the feature path are allowed to differ across
    # platforms: log() from the maths library (IEEE 754 does not require it to
    # be correctly rounded) and the reduction order inside a dot product.
    idx0, val0 = F.hashed_row(texts[0], cfg)
    v = val0 * idf[idx0]
    n_dot = float(np.sqrt(v @ v))
    n_fsum = math.sqrt(math.fsum(float(x) * float(x) for x in v))
    print(f"row0: nnz={len(idx0)} tf_sum={float(val0.sum()).hex()}")
    print(f"norm_dot ={n_dot.hex()}")
    print(f"norm_fsum={n_fsum.hex()} dot_equals_fsum={n_dot == n_fsum}")
    print("first3_values=" + " ".join(float(x).hex() for x in (v / n_dot)[:3]))
    print("libm: 1+log(2)=" + (1.0 + float(np.log(2.0))).hex()
          + " 1+log(3)=" + (1.0 + float(np.log(3.0))).hex())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
