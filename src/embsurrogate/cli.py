"""Command line for the whole workflow.

    embsurrogate archive  --texts corpus.jsonl --vectors vectors.npy --out bundle/
    embsurrogate deposit  --bundle bundle/ --out deposit/ --encoder <model id>
    embsurrogate replay   --bundle bundle/ --texts corpus.jsonl --out vectors.npy
    embsurrogate select   --vectors vectors.npy --texts corpus.jsonl --seeds seeds.txt
    embsurrogate verify   --bundle bundle/

The order is the order of a project's life. `archive` runs once, while the
embedding model is still available, and produces a self-contained directory.
`replay` is what the next person runs, years later, on the raw data: it returns
the original vectors for records that were archived and surrogate vectors for
records that were not, and says in writing which is which. `select` turns
vectors into a reading order with baselines that need no model at all.
`verify` checks that a bundle still reproduces what it claims.
"""

from __future__ import annotations

import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = {"archive": "archive", "replay": "replay", "select": "select",
            "verify": "verify", "deposit": "deposit"}
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if argv else 2
    if argv[0] == "--version":
        from . import __version__
        print(__version__)
        return 0
    if argv[0] not in cmds:
        print(f"unknown command: {argv[0]}\n", file=sys.stderr)
        print(__doc__.strip(), file=sys.stderr)
        return 2
    name = cmds[argv.pop(0)]
    if name == "verify":
        from .verify import main as run
    else:
        from importlib import import_module
        run = import_module(f".{name}", __package__).main
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
