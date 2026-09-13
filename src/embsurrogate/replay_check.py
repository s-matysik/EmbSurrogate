"""Reproducibility test: whether the frozen surrogate gives exactly the same result.

    python3 -m klon.replay_check model/surogat.npz --probes data/probes.jsonl
    python3 -m klon.replay_check model/surogat.npz --probes data/probes.jsonl \\
        --expected model/oczekiwane.npy        # verification mode

Three things are checked, because each can break separately:

1. INTEGRITY — the sha256 of the file and of every array matches the manifest.
2. FEATURE DETERMINISM — features computed now from the training texts saved
   in the model are bit for bit identical to those frozen. This is where a
   change of feature code version, a different PYTHONHASHSEED, or a different
   Unicode normalization shows up.
3. RESULT REPEATABILITY — vectors for the control texts are identical
   to those saved (after quantization to 6 places) and identical between two
   independent calls within the same process.
4. AGREEMENT WITH THE REFERENCE (--reference mode) — the fast numpy path gives the
   same 6 digits as `klon.reference`, i.e. the integer implementation,
   in which there is neither floating-point summation nor any function outside
   the IEEE-754 standard. Points 1-3 check whether THIS machine computes repeatably;
   point 4 checks whether it computes what is defined in the contract —
   and only it will catch a divergence caused by a different BLAS library or a different libm.

    python3 -m klon.replay_check model/surogat_v2.npz --reference --n-ref 20

The reference path costs ~2 s per text (numpy ~0.5 s), so in --reference mode
we take 20 control texts by default; --n-ref 0 = all.

Exit code 0 = everything matches, 1 = ANY difference (details on stdout)."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

from . import features as F
from .predict import FrozenSurrogate


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def check_reference(model_path: pathlib.Path, m, probes: list[str], decimals: int = 6,
                    csr_cache: pathlib.Path | None = None) -> tuple[bool, list[str]]:
    """Comparison of the numpy fast path with the integer implementation.

    INTEGERS after quantization are compared (the archival vector is
    n/10^decimals), not floats — so the test has no tolerance of its own.
    Additionally, the difference before quantization and the margin to the
    rounding boundary are reported: if this margin approaches the difference
    between the paths, agreement is no longer guaranteed, even when it
    happens to come out right this time.
    """
    from . import reference as REF

    log: list[str] = []
    ref = REF.ReferenceModel(model_path, csr_cache=csr_cache)
    scale = 10 ** decimals
    V6, Vhi = ref.encode_int_fast(probes, chunk=25, scales=(scale, 10 ** 18))
    Vnp = m.encode(probes, decimals=None)
    V6np = np.rint(np.round(Vnp, decimals) * scale).astype(np.int64)
    V6ref = V6.astype(np.int64)
    same = bool(np.array_equal(V6np, V6ref))
    bad = int((V6np != V6ref).any(axis=1).sum())
    log.append(f"[{'OK ' if same else 'ERROR'}] numpy vs integer reference: "
               f"{len(probes) - bad}/{len(probes)} texts matching in all "
               f"{V6ref.shape[1]} digits")
    hi = np.array([[float(x) / 10 ** 18 for x in row] for row in Vhi])
    dmax = float(np.abs(Vnp - hi).max())
    half, mod = 5 * 10 ** (18 - decimals - 1), 10 ** (18 - decimals)
    dist = min(abs((int(x) % mod) - half) for row in Vhi for x in row) / mod
    log.append(f"[INFO] max |numpy - reference| before quantization = {dmax:.2e}; "
               f"smallest margin to rounding boundary = {dist:.4f} "
               f"units of 1e-{decimals} ({dist / 10 ** decimals:.2e})")
    if dmax > 0.1 * dist / 10 ** decimals:
        same = False
        log.append("[ERROR] difference between paths is not an order of magnitude smaller "
                   "from the rounding margin — the agreement is coincidental")
    log.append(f"[INFO] smallest gap between kth and (k+1)th neighbour = "
               f"{ref.diag['odstep_k_kp1_min']:.2e} (a tie at this point would change "
               f"neighbour set, and with it the result)")
    return same, log


def check(model_path: pathlib.Path, probes: list[str],
          expected: pathlib.Path | None, decimals: int = 6,
          reference: bool = False, n_ref: int = 20,
          csr_cache: pathlib.Path | None = None) -> tuple[bool, list[str]]:
    log: list[str] = []
    ok = True

    man_path = model_path.with_suffix(".manifest.json")
    m = FrozenSurrogate(model_path)

    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        got = hashlib.sha256(model_path.read_bytes()).hexdigest()
        same = got == man["sha256_pliku"]
        ok &= same
        log.append(f"[{'OK ' if same else 'ERROR'}] sha256 pliku modelu: {got[:16]}...")
    else:
        log.append("[WARNING] no manifest next to model — skipping integrity check")

    indptr, indices, data = F.csr_arrays(m.texts, m.idf, m.cfg)
    same_feat = (_sha(indptr) == _sha(m.indptr) and _sha(indices) == _sha(m.indices)
                 and _sha(data) == _sha(m.data))
    ok &= same_feat
    log.append(f"[{'OK ' if same_feat else 'ERROR'}] training features reconstructed "
               f"bit for bit ({len(m.texts)} texts, nnz={len(data)})")

    V1 = m.encode(probes, decimals=decimals)
    V2 = m.encode(probes, decimals=decimals)
    same_run = np.array_equal(V1, V2)
    ok &= same_run
    log.append(f"[{'OK ' if same_run else 'ERROR'}] two calls in the same process "
               f"agree ({V1.shape})")

    if expected is not None:
        if expected.exists():
            V0 = np.load(expected)
            same_exp = np.array_equal(V0, V1)
            ok &= same_exp
            d = float(np.max(np.abs(V0 - V1))) if V0.shape == V1.shape else float("nan")
            log.append(f"[{'OK ' if same_exp else 'ERROR'}] agreement with the stored golden file "
                       f"{expected.name} (max |difference| = {d:.2e})")
            Vraw1 = m.encode(probes, decimals=None)
            log.append(f"[INFO] raw float64 vs quantized: max |difference| = "
                       f"{float(np.max(np.abs(Vraw1 - V1))):.2e} "
                       f"(this is only the magnitude of the rounding itself to {decimals} decimals; "
                       f"quantization REDUCES, but does not eliminate the risk of inconsistency "
                       f"between machines — the measure of that risk is the margin to the "
                       f"rounding boundary from --reference mode, see "
                       f"docs/DETERMINISM.md sec. 3)")
        else:
            np.save(expected, V1)
            log.append(f"[INFO] utworzono wzorzec {expected.name}, sha256={_sha(V1)[:16]}...")

    if reference:
        sub = probes if n_ref in (0, None) else probes[:n_ref]
        ok_ref, log_ref = check_reference(model_path, m, sub, decimals, csr_cache)
        ok &= ok_ref
        log.extend(log_ref)
    return ok, log


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=pathlib.Path)
    ap.add_argument("--probes", type=pathlib.Path)
    ap.add_argument("--field", default="text")
    ap.add_argument("--expected", type=pathlib.Path)
    ap.add_argument("--decimals", type=int, default=6)
    ap.add_argument("--reference", action="store_true",
                    help="compare the numpy path with the integer implementation")
    ap.add_argument("--n-ref", type=int, default=20,
                    help="how many texts in --reference mode (0 = all)")
    ap.add_argument("--csr-cache", type=pathlib.Path, default=None,
                    help="npz file with CSR arrays (reconstructing features takes ~85 s)")
    args = ap.parse_args(argv)

    if args.probes:
        with args.probes.open(encoding="utf-8") as f:
            probes = [json.loads(l)[args.field] for l in f if l.strip()]
    else:  # without a control file, we take every tenth training text
        probes = FrozenSurrogate(args.model).texts[::10]

    ok, log = check(args.model, probes, args.expected, args.decimals,
                    reference=args.reference, n_ref=args.n_ref,
                    csr_cache=args.csr_cache)
    print("\n".join(log))
    print("\nRESULT:", "REPRODUCIBLE" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
