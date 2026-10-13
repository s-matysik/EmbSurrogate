#!/usr/bin/env bash
# End-to-end on a small sample: build a deposit, verify it, replay both
# populations, and see the fidelity measured on YOUR corpus.
#
#     bash examples/try_it.sh corpus.jsonl            # 600 texts, no encoder
#     bash examples/try_it.sh corpus.jsonl 600 sentence-transformers/all-MiniLM-L6-v2
#
# corpus.jsonl: one JSON object per line with a "text" field. The last 10% are
# held out and never archived, so the replay step shows both an exact lookup and
# a genuine approximation. Takes a couple of minutes on a laptop CPU.
set -euo pipefail

CORPUS="${1:?usage: try_it.sh corpus.jsonl [n] [encoder-model]}"
N="${2:-600}"
ENCODER="${3:-}"
WORK="${WORK:-try_it_out}"
export PYTHONPATH="${PYTHONPATH:-src}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$WORK"
head -n "$N" "$CORPUS" > "$WORK/sample.jsonl"
HOLD=$(( N / 10 ))
head -n $(( N - HOLD )) "$WORK/sample.jsonl" > "$WORK/archived.jsonl"
tail -n "$HOLD" "$WORK/sample.jsonl" > "$WORK/new.jsonl"
echo "== sample: $(wc -l < "$WORK/archived.jsonl") archived, $(wc -l < "$WORK/new.jsonl") held out"

echo "== 1/5 teacher vectors (the only step needing the model)"
python examples/embed_corpus.py "$WORK/archived.jsonl" "$WORK/vectors.npy"

echo "== 2/5 archive: fit the surrogate and freeze the bundle"
if [ -n "$ENCODER" ]; then
  python -m embsurrogate.cli archive --texts "$WORK/archived.jsonl" \
      --vectors "$WORK/vectors.npy" --out "$WORK/bundle" --encoder "$ENCODER"
else
  python -m embsurrogate.cli archive --texts "$WORK/archived.jsonl" \
      --vectors "$WORK/vectors.npy" --out "$WORK/bundle"
fi

echo "== 3/5 verify: what a referee runs, on a machine that did not build it"
python -m embsurrogate.cli verify --bundle "$WORK/bundle" --strict

echo "== 4/5 replay the ARCHIVED texts: every record must come back exact"
python -m embsurrogate.cli replay --bundle "$WORK/bundle" \
    --texts "$WORK/archived.jsonl" --out "$WORK/again.npy" \
    --provenance "$WORK/prov_archived.csv"

echo "== 5/5 replay the HELD-OUT texts: these are the approximated ones"
python -m embsurrogate.cli replay --bundle "$WORK/bundle" \
    --texts "$WORK/new.jsonl" --out "$WORK/new.npy" \
    --provenance "$WORK/prov_new.csv"

if [ -n "$ENCODER" ]; then
  echo "== bonus: the text-free deposit, for a corpus you may not redistribute"
  python -m embsurrogate.cli deposit --bundle "$WORK/bundle" \
      --out "$WORK/deposit" --encoder "$ENCODER"
fi

echo "== how close the approximation is, against the teacher it never saw"
python examples/embed_corpus.py "$WORK/new.jsonl" "$WORK/new_teacher.npy" >/dev/null
python - "$WORK" <<'PY'
import json, pathlib, sys
import numpy as np
w = pathlib.Path(sys.argv[1])
yh, yt = np.load(w / "new.npy"), np.load(w / "new_teacher.npy").astype(np.float64)
cos = (yh * yt).sum(1)
S_h, S_t = yh @ yh.T, yt @ yt.T
iu = np.triu_indices(len(yt), 1)
r_h = S_h[iu].argsort().argsort(); r_t = S_t[iu].argsort().argsort()
rho = float(np.corrcoef(r_h, r_t)[0, 1])
man = json.loads((w / "bundle" / "manifest.json").read_text())
print(json.dumps({
    "held_out_texts": int(len(yt)),
    "cosine_to_teacher_median": round(float(np.median(cos)), 4),
    "similarity_rank_correlation": round(rho, 4),
    "fidelity_archive_measured_on_your_corpus": man.get("fidelity_new_texts"),
    "route_kept": (man.get("bridge") or "n-gram only"),
}, indent=2, default=str))
PY
echo "== files in $WORK: bundle/, prov_archived.csv, prov_new.csv"
