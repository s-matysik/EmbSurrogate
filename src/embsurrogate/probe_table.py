"""Designed probe set: single words instead of (only) sentences.

Motivation. If we have the right to feed the model arbitrary inputs before it disappears,
then natural texts are a poor choice: each of them mixes several dozen
words and we end up with one equation for many unknowns. By giving the model
a SINGLE WORD we get directly its image in the output space —
i.e. a row of the static table. A set of N single-word probes gives a system
with an identity matrix instead of a dense one.

Hence the two-stage construction of the surrogate:

    step 1 (table)   W[w] = teacher("w")            for w from the dictionary
    step 2 (pooling)  base(t) = normalization( sum_w weight(w) * W[w] )
    step 3 (correction)  surrogate(t) = normalization( base(t) + residual(t) )

where the residual is fitted with kernel regression on pairs from an ordinary
corpus. Step 2 corresponds to what static distillates of sentence
models do (Model2Vec and related ones), with the difference that here we do not look
inside the model — everything comes from input/output pairs.

    python3 -m klon.probe_table data/ds_multi/meta.jsonl data/sondy.jsonl --top 20000"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

import numpy as np

from . import features as F


def vocabulary(texts, top: int = 20000, min_count: int = 2) -> list[tuple[str, int]]:
    """Most frequent words (after normalization as in featurization)."""
    c: Counter = Counter()
    for t in texts:
        c.update(F._WORD_RE.findall(F.normalize(t)))
    items = [(w, n) for w, n in c.most_common() if n >= min_count]
    return items[:top]


def sif_weights(counts: np.ndarray, a: float = 1e-3) -> np.ndarray:
    """SIF weights (Arora et al. 2017): a / (a + p(w)) — dampen frequent words."""
    p = counts / counts.sum()
    return a / (a + p)


def pool(texts, vocab_index: dict[str, int], W: np.ndarray, weights: np.ndarray,
         cfg: dict | None = None) -> np.ndarray:
    """Text -> weighted sum of table rows, normalized."""
    cfg = cfg or F.DEFAULT_CONFIG
    out = np.zeros((len(texts), W.shape[1]), dtype=np.float64)
    for i, t in enumerate(texts):
        acc = np.zeros(W.shape[1], dtype=np.float64)
        for w in F._WORD_RE.findall(F.normalize(t, cfg)):
            j = vocab_index.get(w)
            if j is not None:
                acc += weights[j] * W[j]
        out[i] = acc
    n = np.linalg.norm(out, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return out / n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("meta", type=pathlib.Path, help="meta.jsonl ze zbioru treningowego")
    ap.add_argument("out", type=pathlib.Path, help="jsonl z sondami do zaembedowania")
    ap.add_argument("--top", type=int, default=20000)
    ap.add_argument("--min-count", type=int, default=2)
    args = ap.parse_args(argv)

    texts = [json.loads(l)["text"] for l in args.meta.open(encoding="utf-8") if l.strip()]
    vocab = vocabulary(texts, args.top, args.min_count)
    with args.out.open("w", encoding="utf-8") as f:
        for w, n in vocab:
            f.write(json.dumps({"id": f"probe-{w}", "source": "sonda-slowo",
                                "title": w, "text": w, "count": n},
                               ensure_ascii=False) + "\n")
    print(f"sond: {len(vocab)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
