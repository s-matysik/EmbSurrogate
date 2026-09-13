"""Teacher model: SBERT instead of bge-m3 from Ollama.

In the repository, `embed.py` calls a local Ollama (host on the Tailscale network). For the
surrogate experiment we need a teacher that is reachable everywhere and
versionable, so we take models from sentence-transformers. The output
format is compatible with `build_dataset.py`: vectors.npy + meta.jsonl in the
same order.

    python3 -m klon.teacher_sbert data/korpus.jsonl data/ds_minilm \\
        --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --limit 6000

It also saves model_info.json with the commit (revision) of the model from HuggingFace —
without pinning the revision, "the same model" can produce different vectors after a year."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np


def load_corpus(path: pathlib.Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=pathlib.Path)
    ap.add_argument("outdir", type=pathlib.Path)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-chars", type=int, default=0,
                    help="0 = bez limitu; model i tak utnie do swojego okna")
    args = ap.parse_args(argv)

    from sentence_transformers import SentenceTransformer

    rows = load_corpus(args.corpus, args.limit or None)
    texts = [r["text"][: args.max_chars] if args.max_chars else r["text"] for r in rows]
    print(f"texts: {len(texts)}  model: {args.model}")

    t0 = time.time()
    model = SentenceTransformer(args.model)
    dim = model.get_sentence_embedding_dimension()
    V = model.encode(
        texts,
        batch_size=args.batch,
        convert_to_numpy=True,
        normalize_embeddings=True,      # cosine = dot product
        show_progress_bar=False,
    ).astype(np.float32)
    dt = time.time() - t0

    args.outdir.mkdir(parents=True, exist_ok=True)
    np.save(args.outdir / "vectors.npy", V)
    with (args.outdir / "meta.jsonl").open("w", encoding="utf-8") as f:
        for r, t in zip(rows, texts):
            f.write(json.dumps({
                "id": r["id"], "source": r.get("source", ""),
                "title": r.get("title", ""), "chars": len(t), "text": t,
            }, ensure_ascii=False) + "\n")

    info = {
        "model": args.model,
        "dim": int(dim),
        "n": int(V.shape[0]),
        "normalized": True,
        "max_seq_length": int(getattr(model, "max_seq_length", 0)),
        "encode_seconds": round(dt, 1),
    }
    try:  # revision from HF cache — without it the model version cannot be pinned
        from huggingface_hub import HfApi
        info["revision"] = HfApi().model_info(args.model).sha
    except Exception as exc:                       # no network / no package
        info["revision"] = None
        info["revision_error"] = str(exc)[:200]
    (args.outdir / "model_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"gotowe: {V.shape} w {dt:.1f}s -> {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
