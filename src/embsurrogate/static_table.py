"""Static token table distilled from the teacher: one vector per token.

    python -m embsurrogate.static_table build --out bundle/table.npz


At the time the archive is built the teacher still exists, so it may be asked
anything. We push EVERY token of its own vocabulary through it separately and
store the result as a table of numbers. The table is archival in exactly the
same sense as the coefficient matrix: a float array plus the tokenizer file
(sentencepiece, a few megabytes), and composing a text vector is a lookup, a
weighted mean and a normalisation -- no network, no randomness, no dependence
on the version of the modelling library.

Three ingredients that an earlier attempt with single-word probes lacked (rho
0.27 on its own, and nothing added to the ridge):

1. **tokens of the tokenizer, not words of the corpus** -- the vocabulary
   covers any text, because unseen words break into sub-words;
2. **PCA reduction** -- measured: it is worth +0.07 rho;
3. **weights falling with token frequency** -- approximated by the rank in the
   vocabulary (Zipf's law). We also have the corpus, so measured inverse
   document frequency was tried as well, and lost: 0.456 against 0.479. The
   proxy beat the measurement, because corpus IDF overvalues tokens that are
   rare in this corpus but common in the language.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

TEACHER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ------------------------------------------------------------ build the table

def build_table(out: pathlib.Path, teacher: str = TEACHER,
                batch: int = 1024, n_components: int = 256) -> dict:
    """Pushes every vocabulary token through the teacher and stores the table."""
    from sentence_transformers import SentenceTransformer

    tm = SentenceTransformer(teacher)
    tok = tm.tokenizer
    vocab = tok.get_vocab()                      # token -> id
    inv = [None] * (max(vocab.values()) + 1)
    for t, i in vocab.items():
        inv[i] = t
    ids = [i for i, t in enumerate(inv) if t is not None]
    texts = [tok.convert_tokens_to_string([inv[i]]).strip() or inv[i] for i in ids]

    E = np.zeros((len(inv), tm.get_sentence_embedding_dimension()), dtype=np.float32)
    for s in range(0, len(ids), batch):
        chunk = texts[s:s + batch]
        V = tm.encode(chunk, convert_to_numpy=True, normalize_embeddings=False,
                      batch_size=batch, show_progress_bar=False)
        E[ids[s:s + batch]] = V
        if (s // batch) % 20 == 0:
            print(f"  {s}/{len(ids)}", flush=True)

    mu = E.mean(0, keepdims=True)
    X = (E - mu).astype(np.float64)
    # PCA through an SVD of the centred matrix: deterministic, no random start
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    P = Vt[:n_components].T                      # projection onto the leading directions
    Z = X @ P
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, E=E.astype(np.float64), mu=mu.astype(np.float64), P=P,
             Z=Z, S=S[:n_components], tokens=np.array(
                 [t if t else "" for t in inv], dtype=object), allow_pickle=True)
    info = {"teacher": teacher, "tokens": int(len(inv)),
            "teacher_dim": int(E.shape[1]), "pca": int(n_components),
            "variance_first_component": float(S[0] ** 2 / (S ** 2).sum()),
            "variance_retained": float((S[:n_components] ** 2).sum() / (S ** 2).sum())}
    out.with_suffix(".info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


# ------------------------------------------------------------- use the table

class StaticTable:
    """Composing a text vector: lookup, weighted mean, normalisation."""

    def __init__(self, path: pathlib.Path, teacher: str = TEACHER,
                 space: str = "pca", weighting: str = "zipf",
                 idf: np.ndarray | None = None):
        from transformers import AutoTokenizer

        z = np.load(path, allow_pickle=True)
        self.tok = AutoTokenizer.from_pretrained(teacher)
        self.E = z["E"] if space == "raw" else z["Z"]
        self.space, self.weighting = space, weighting
        V = self.E.shape[0]
        if weighting == "zipf":
            # the tokenizer vocabulary is roughly ordered by frequency, so the
            # logarithm of the rank plays the role of an inverse frequency
            self.w = np.log1p(np.arange(V, dtype=np.float64))
            self.w[0] = 0.0
        elif weighting == "idf":
            if idf is None:
                raise ValueError("idf weights must be computed on a corpus")
            self.w = idf
        elif weighting == "none":
            self.w = np.ones(V)
        else:
            raise ValueError(weighting)

    def ids(self, texts: list[str], max_len: int = 512) -> list[np.ndarray]:
        enc = self.tok(texts, add_special_tokens=False, truncation=True,
                       max_length=max_len)["input_ids"]
        return [np.asarray(e, dtype=np.int64) for e in enc]

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.E.shape[1]))
        for i, ii in enumerate(self.ids(texts)):
            if len(ii) == 0:
                continue
            w = self.w[ii]
            s = w.sum()
            out[i] = (w[:, None] * self.E[ii]).sum(0) / (s if s > 0 else 1.0)
        n = np.linalg.norm(out, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return out / n


def corpus_idf(table: StaticTable, texts: list[str]) -> np.ndarray:
    """Inverse document frequency of tokens, measured on the archived corpus."""
    V = table.E.shape[0]
    df = np.zeros(V)
    for ii in table.ids(texts):
        df[np.unique(ii)] += 1
    return np.log((1 + len(texts)) / (1 + df)) + 1.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=pathlib.Path, required=True)
    b.add_argument("--pca", type=int, default=256)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        print(json.dumps(build_table(a.out, n_components=a.pca),
                         indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
