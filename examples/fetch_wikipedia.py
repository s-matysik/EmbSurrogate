"""Fetch the corpus the paper's fidelity experiment uses.

    python examples/fetch_wikipedia.py 45493 corpus.jsonl --contact you@example.org

Random articles from the Polish Wikipedia, lead section only, written as JSONL
with the fields `archive` expects: `{"id", "source", "title", "text"}`. The run
is resumable -- ids already in the output file are skipped -- because at one
request per second a corpus of this size takes hours.

Wikimedia asks every automated client to identify itself and to give a contact
address; pass it with `--contact`. Without one the request still goes through,
but rate limiting is stricter. No address is stored in this file.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import requests

API = "https://pl.wikipedia.org/w/api.php"
DELAY = 1.0          # measured: 0.1 s gets a 429 after ~160 records, 1.0 s is stable
MIN_CHARS = 200      # shorter entries are stubs with no body text
BATCH = 20           # the API returns at most 20 extracts per request


def _session(contact: str | None) -> requests.Session:
    who = "embsurrogate-example/1.0"
    if contact:
        who += f" ({contact})"
    s = requests.Session()
    s.headers.update({"User-Agent": who})
    return s


def _seen(path: pathlib.Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                out.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def fetch(n: int, out: pathlib.Path, contact: str | None = None,
          max_chars: int = 0) -> int:
    s = _session(contact)
    seen = _seen(out)
    added, fails = 0, 0
    with out.open("a", encoding="utf-8") as f:
        while added < n:
            try:
                r = s.get(API, params={
                    "action": "query", "format": "json",
                    "generator": "random", "grnnamespace": 0, "grnlimit": BATCH,
                    "prop": "extracts", "exintro": 1, "explaintext": 1,
                    "exlimit": BATCH}, timeout=30)
                if r.status_code == 429:
                    # the server says how long to wait; listen to it
                    wait = int(r.headers.get("Retry-After", 30))
                    print(f"  429: waiting {wait}s", file=sys.stderr)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                data = r.json()
                fails = 0
            except Exception as exc:                      # noqa: BLE001
                fails += 1
                if fails >= 10:
                    print(f"  10 consecutive failures, stopping: {exc}", file=sys.stderr)
                    break
                time.sleep(min(60, 5 * 2 ** (fails - 1)))
                continue
            for page in data.get("query", {}).get("pages", {}).values():
                pid = f"wiki-{page['pageid']}"
                text = (page.get("extract") or "").strip()
                if pid in seen or len(text) < MIN_CHARS:
                    continue
                if max_chars and len(text) > max_chars:
                    continue
                seen.add(pid)
                f.write(json.dumps({"id": pid, "source": "wikipedia-pl",
                                    "title": page.get("title", ""), "text": text},
                                   ensure_ascii=False) + "\n")
                added += 1
                if added >= n:
                    break
            f.flush()
            if added % 500 < BATCH:
                print(f"  {added}/{n}", flush=True)
            time.sleep(DELAY)
    return added


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("n", type=int)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--contact", default=None,
                    help="contact address for the User-Agent, as Wikimedia asks")
    ap.add_argument("--max-chars", type=int, default=0)
    a = ap.parse_args(argv)
    got = fetch(a.n, a.out, a.contact, a.max_chars)
    print(json.dumps({"written": got, "file": str(a.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
