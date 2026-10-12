"""Fetch the screening benchmark the paper's Section 3.2 uses.

    python examples/fetch_clef.py CD012661 data/clef --email you@example.org

CLEF eHealth TAR 2019 (Task 2): five Cochrane reviews with abstract-level
relevance judgements. Labels come from the CLEF-TAR/tar repository, texts from
PubMed through E-utilities, so the experiment needs no private corpus and
nothing is transcribed by hand. Writes three files per topic:

    <topic>_pool.json     records: key, title, abstract, year
    <topic>_labels.csv    key, decision (include/exclude)
    <topic>_info.json     the review question, counts, and where each came from

NCBI asks automated clients for a contact address; pass `--email` to supply one.
Without it the fetch still works, under stricter rate limiting. No address is
stored in this file.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

GH = "https://api.github.com/repos/CLEF-TAR/tar/contents"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PATHS = {  # topic -> (qrels directory, qrels file name)
    "CD012661": ("2019-TAR/Task2/Testing/Prognosis",
                 "full.test.prognosis.abs.2019.qrels"),
}
# These four share one qrels file under Intervention. They were chosen so that
# the pools fit in memory and the number of relevant papers still supports
# usable confidence intervals (prevalence 0.3% to 9%).
for _t in ("CD012069", "CD009069", "CD009642", "CD006468"):
    PATHS[_t] = ("2019-TAR/Task2/Testing/Intervention",
                 "full.test.intervention.abs.2019.qrels")


def _get(url: str, params: dict | None = None) -> bytes:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def fetch(topic: str, out: pathlib.Path, email: str | None = None,
          batch: int = 200, pause: float = 0.4) -> dict:
    folder, qrels_name = PATHS[topic]
    meta = json.loads(_get(f"{GH}/{folder}/qrels/{qrels_name}"))
    if meta.get("content"):
        qrels = base64.b64decode(meta["content"]).decode("utf-8", "replace")
    else:   # the contents API returns empty content above 1 MB, hence the blob
        blob = json.loads(_get(
            f"https://api.github.com/repos/CLEF-TAR/tar/git/blobs/{meta['sha']}"))
        qrels = base64.b64decode(blob["content"]).decode("utf-8", "replace")
    labels = {}
    for line in qrels.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == topic:
            labels[parts[2]] = int(parts[3])

    tmeta = json.loads(_get(f"{GH}/{folder}/topics/{topic}"))
    ttext = base64.b64decode(tmeta["content"]).decode("utf-8", "replace")
    title = ttext.split("Title:", 1)[1].split("Query:", 1)[0].strip()

    pmids = sorted(labels)
    recs, missing = [], 0
    for i in range(0, len(pmids), batch):
        p = {"db": "pubmed", "id": ",".join(pmids[i:i + batch]), "retmode": "xml"}
        if email:
            p["email"] = email
            p["tool"] = "embsurrogate-screening-example"
        root = ET.fromstring(_get(f"{EUTILS}/efetch.fcgi", p))
        for art in root.iter("PubmedArticle"):
            pid = art.findtext(".//PMID")
            ti = " ".join("".join(t.itertext()) for t in art.iter("ArticleTitle"))
            ab = " ".join("".join(t.itertext())
                          for t in art.iter("AbstractText")).strip()
            if not ab:          # a record with no abstract cannot be screened on text
                missing += 1
                continue
            recs.append({"key": pid, "title": ti.strip(), "abstract": ab,
                         "year": art.findtext(".//PubDate/Year") or ""})
        print(f"  {min(i + batch, len(pmids))}/{len(pmids)}", flush=True)
        time.sleep(pause)

    out.mkdir(parents=True, exist_ok=True)
    (out / f"{topic}_pool.json").write_text(
        json.dumps(recs, ensure_ascii=False), encoding="utf-8")
    with (out / f"{topic}_labels.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "decision"])
        for r in recs:
            w.writerow([r["key"], "include" if labels[r["key"]] else "exclude"])
    info = {"topic": topic, "review_question": title,
            "pmids_in_qrels": len(pmids), "fetched_with_abstract": len(recs),
            "without_abstract": missing,
            "relevant_among_fetched": sum(labels[r["key"]] for r in recs),
            "label_source": f"github CLEF-TAR/tar {folder}/qrels/{qrels_name}",
            "text_source": "PubMed E-utilities efetch"}
    (out / f"{topic}_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("topic", choices=sorted(PATHS))
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--email", default=None,
                    help="contact address for NCBI E-utilities, as they ask")
    a = ap.parse_args(argv)
    print(json.dumps(fetch(a.topic, a.out, a.email), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
