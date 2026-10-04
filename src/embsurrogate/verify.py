"""Check that a bundle still reproduces what its manifest claims.

    embsurrogate verify --bundle bundle/ [--strict]

Exit code 0 means every check passed. This is the command a reviewer runs, and
the one a continuous-integration job runs on a machine that did not build the
bundle -- which is where the project's one real cross-platform defect was found
(a reduction in the feature normalisation whose summation order depends on the
architecture; see docs/DETERMINISM.md).

Four checks, in increasing strength:

1. the archived arrays match the checksums recorded at build time;
2. the features recomputed from the archived texts match their checksums, so
   the featuriser in this installation behaves as it did when the bundle was
   built -- this is the check that catches a platform difference;
3. the archived vectors come back exactly for archived texts, which is the
   claim that "the same raw data gives the same result" rests on;
4. with `--strict`, the surrogate path is additionally compared against the
   integer-arithmetic reference implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

from . import features as F


def check(bundle: pathlib.Path, strict: bool = False, sample: int = 64) -> dict:
    from .replay import Bundle

    out = {"bundle": str(bundle), "checks": {}}
    b = Bundle(bundle, verify=True)            # raises on a checksum mismatch
    out["checks"]["array_checksums"] = True
    out["n_pairs"] = b.manifest["n_pairs"]
    out["feature_version"] = b.manifest["feature_version"]

    man_texts = hashlib.sha256(
        "\u0000".join(b.texts).encode("utf-8")).hexdigest()
    out["checks"]["texts_intact"] = man_texts == b.manifest["sha256"]["texts"]

    # Features recomputed HERE must match the digests recorded when the bundle
    # was built. This is the strongest check in the file: it is what failed on
    # Linux while passing on the machine that built the fixture, and it located
    # a reduction whose summation order depends on the architecture.
    dig = b.manifest.get("feature_digest")
    if dig:
        got = dict(zip(("indptr", "indices", "data"),
                       (hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
                        for x in F.csr_arrays(b.texts, b.idf, b.cfg))))
        out["checks"]["features_match_build_time_digests"] = got == dig
        out["feature_digest_mismatch"] = [kk for kk in dig if got[kk] != dig[kk]]
    idx = np.linspace(0, len(b.texts) - 1, min(sample, len(b.texts))).astype(int)
    sub = [b.texts[i] for i in idx]
    indptr, indices, data = F.csr_arrays(sub, b.idf, b.cfg)
    Xs = F.transform(sub, b.idf, b.cfg)
    out["checks"]["numpy_and_scipy_paths_agree"] = bool(
        np.array_equal(indices, Xs.indices) and np.array_equal(data, Xs.data))

    # the exactness claim: archived texts return the archived vectors
    Y, prov = b.vectors_for(sub)
    exact = np.array_equal(Y, b.Y[idx])
    out["checks"]["archived_texts_exact"] = bool(exact and all(
        p == "exact" for p in prov))

    # a text that is certainly not archived must be routed to the surrogate
    probe = ["zzz not an archived record " + "q" * 40]
    _, prov2 = b.vectors_for(probe)
    out["checks"]["new_text_routed_to_surrogate"] = prov2 == ["surrogate"]

    if strict:
        # `features_fsum` is the reference featuriser: correctly rounded
        # summation, no BLAS, written to be read rather than to be fast. The
        # fast path is allowed to differ from it only in the last bits.
        from . import reference as REF
        worst = 0.0
        for t in sub[:8]:
            row = F.transform([t], b.idf, b.cfg)
            o = np.argsort(row.indices)
            keys, vals = REF.features_fsum(t, b.idf, b.cfg)
            if not np.array_equal(row.indices[o], np.asarray(keys)):
                worst = float("inf")
                break
            worst = max(worst, float(np.max(np.abs(
                row.data[o] - np.asarray(vals)))))
        out["reference_max_difference"] = worst
        out["checks"]["reference_implementation_agrees"] = worst < 1e-9

    out["fidelity_for_new_texts"] = {
        kk: round(float(vv), 4) for kk, vv in b.fidelity.items()
        if kk in ("rho_globalne", "cos_mediana", "R2", "recall@10")}
    out["passed"] = all(out["checks"].values())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="embsurrogate verify")
    ap.add_argument("--bundle", type=pathlib.Path, required=True)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args(argv)
    rep = check(a.bundle, strict=a.strict)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
