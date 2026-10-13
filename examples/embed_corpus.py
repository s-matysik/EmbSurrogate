"""Produce the teacher vectors `archive` takes as input.

    python examples/embed_corpus.py corpus.jsonl vectors.npy \
        --model sentence-transformers/all-MiniLM-L6-v2

This is the one step that needs the embedding model, and the only step that ever
will: run it while the model is still installed. The output is an N x d float32
array in the order the texts appear in the JSONL, which is the contract
`archive` expects. The model identifier and its revision are printed so they can
go into the methods section, since the name alone is not a reproducible
reference.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def read_texts(path: pathlib.Path, field: str = "text") -> list[str]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line)[field])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("texts", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--field", default="text")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args(argv)

    from sentence_transformers import SentenceTransformer       # imported late:
    # only this script needs it, and the archival path must stay numpy-only.

    texts = read_texts(a.texts, a.field)
    m = SentenceTransformer(a.model, device="cpu")
    Y = m.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                 batch_size=a.batch, show_progress_bar=False).astype(np.float32)
    np.save(a.out, Y)
    rev = getattr(getattr(m, "model_card_data", None), "base_model_revision", None)
    print(json.dumps({"texts": len(texts), "dimension": int(Y.shape[1]),
                      "model": a.model, "revision": rev,
                      "out": str(a.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
