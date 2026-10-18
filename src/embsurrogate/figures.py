"""Summary figure: how much of the teacher model the surrogate reproduces.

    python3 -m klon.figures data/ds_multi wyniki/multi model/surogat.npz \\
        wyniki/rysunek_wiernosc.png

Four panels: (a) dependence on the number of pairs, (b) comparison of method families,
(c) reconstruction of the similarity matrix, (d) distribution of point-wise fidelity."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import evaluate as E
from .predict import FrozenSurrogate
from .run_experiment import load_dataset, split

C_SUR = "#1f4e79"      # n-gram surrogate
C_CEIL = "#c47c1b"     # reference: another neural model
C_FLOOR = "#8c8c8c"    # floor: average vector


def build(dataset: pathlib.Path, wyniki: pathlib.Path, model: pathlib.Path,
          out: pathlib.Path):
    texts, Y, _ = load_dataset(dataset)
    tr, te = split(len(texts), 0.2, 0)
    Yte = Y[te]
    fm = FrozenSurrogate(model)
    Yh = fm.encode([texts[i] for i in te], decimals=None)

    lc = pd.read_csv(wyniki / "krzywa_uczenia.csv")
    met = pd.read_csv(wyniki / "metryki.csv")
    ceil = pd.read_csv(wyniki / "sufit.csv")
    best_ceiling = ceil.sort_values("spearman_sim").iloc[-1]
    floor = met[met.metoda == "floor_sredni_wektor"].iloc[0]

    cos = E.cosine_rowwise(Yte, Yh)
    St, Sh = Yte @ Yte.T, Yh @ Yh.T
    iu = np.triu_indices(len(Yte), 1)
    rng = np.random.default_rng(0)
    sel = rng.choice(iu[0].size, 200_000, replace=False)
    st, sh = St[iu][sel], Sh[iu][sel]

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.32)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # (a) dependence on the number of pairs
    ax_a.plot(lc.n_treningowych, lc.spearman_sim, "o-", color=C_SUR, ms=4,
              label="geometria (rho)")
    ax_a.plot(lc.n_treningowych, lc.cos_mediana, "s--", color=C_SUR, ms=4,
              alpha=0.55, label="punktowo (cos)")
    ax_a.axhline(best_ceiling.spearman_sim, color=C_CEIL, lw=1.1, ls=":")
    ax_a.text(260, best_ceiling.spearman_sim + 0.012, "different transformer\n+ same regression",
              color=C_CEIL, fontsize=6, va="bottom")
    ax_a.set_xscale("log")
    ax_a.set_xticks([250, 1000, 4800]); ax_a.set_xticklabels(["250", "1k", "4.8k"])
    ax_a.xaxis.set_minor_locator(mpl.ticker.NullLocator())   # without intermediate labels
    ax_a.set_xlabel("(text, vector) pairs in the training set"); ax_a.set_ylabel("fidelity")
    ax_a.set_title("Fidelity grows with the number of pairs,\nbut ever more slowly", loc="left")
    ax_a.legend(frameon=False, fontsize=6, loc="lower right", borderaxespad=0.1,
                handlelength=1.4)
    ax_a.set_ylim(0.25, 0.95)

    # (b) family of methods
    fam = {"1-NN (copy the neighbour)": "knn1", "Nadaraya-Watson": "nw",
           "LOESS (local linear)": "loess", "kernel ridge": "ridge",
           "RBF kernel": "rbf", "ridge + local correction": "hybrid"}
    vals = {lab: met[met.metoda == m].spearman_sim.max() for lab, m in fam.items()}
    order = sorted(vals, key=vals.get)
    ypos = np.arange(len(order))
    ax_b.hlines(ypos, 0, [vals[k] for k in order], color=C_SUR, lw=1.0, alpha=0.45)
    ax_b.plot([vals[k] for k in order], ypos, "o", color=C_SUR, ms=5)
    ax_b.axvline(floor.spearman_sim if np.isfinite(floor.spearman_sim) else 0.0,
                 color=C_FLOOR, lw=1.0, ls="--")
    ax_b.axvline(best_ceiling.spearman_sim, color=C_CEIL, lw=1.1, ls=":")
    # method labels inside the panel, in the narrow two-column layout
    # the y-axis captions would overlap the neighbouring panel
    ax_b.set_yticks(ypos); ax_b.set_yticklabels([])
    for yy, lab in zip(ypos, order):
        ax_b.text(0.015, yy + 0.22, lab, fontsize=6, va="bottom", ha="left")
    ax_b.set_xlim(0, 0.95); ax_b.set_xlabel("Spearman rho of the similarity matrix")
    ax_b.set_title("Global regression beats local", loc="left")
    ax_b.text(0.30, -0.62, "floor: mean vector", color=C_FLOOR, fontsize=6,
              va="center")
    ax_b.set_ylim(-1.0, len(order) - 0.2)
    ax_b.text(best_ceiling.spearman_sim - 0.02, 0.15, "sufit", color=C_CEIL,
              fontsize=6, ha="right")

    # (c) similarity matrix
    hb = ax_c.hexbin(st, sh, gridsize=55, bins="log", cmap="Blues", mincnt=1,
                     linewidths=0)
    ax_c.plot([-0.1, 1.0], [-0.1, 1.0], color="k", lw=0.8, ls="--")
    ax_c.set_xlabel("teacher cos (text pair)")
    ax_c.set_ylabel("cos surogatu")
    ax_c.set_title("Pairwise similarities: consistent in\norder, overestimated in value",
                   loc="left")
    cb = fig.colorbar(hb, ax=ax_c, pad=0.02, fraction=0.045)
    cb.set_label("par (skala log)", fontsize=6)
    cb.ax.tick_params(labelsize=6)

    # (d) point-wise fidelity
    ax_d.hist(cos, bins=45, color=C_SUR, alpha=0.85)
    ax_d.axvline(np.median(cos), color="k", lw=1.0)
    ax_d.text(np.median(cos) + 0.01, ax_d.get_ylim()[1] * 0.92,
              f"mediana {np.median(cos):.2f}", fontsize=6)
    ax_d.axvline(float(floor.cos_mediana), color=C_FLOOR, lw=1.0, ls="--")
    ax_d.text(float(floor.cos_mediana) - 0.01, ax_d.get_ylim()[1] * 0.62, "floor",
              fontsize=6, color=C_FLOOR, ha="right")
    ax_d.axvline(float(best_ceiling.cos_mediana), color=C_CEIL, lw=1.1, ls=":")
    ax_d.text(float(best_ceiling.cos_mediana) + 0.01, ax_d.get_ylim()[1] * 0.62, "sufit",
              fontsize=6, color=C_CEIL)
    ax_d.set_xlabel("cos(surrogate(t), teacher(t))")
    ax_d.set_ylabel("number of test texts")
    ax_d.set_title("Single vector reproduced\nat 0.73 on average", loc="left")

    for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")):
        ax.text(-0.16, 1.16, letter, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="top")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    return fig, {"cos_mediana": float(np.median(cos)),
                 "n_test": int(len(te)), "n_train": int(len(tr))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=pathlib.Path)
    ap.add_argument("wyniki", type=pathlib.Path)
    ap.add_argument("model", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    args = ap.parse_args(argv)
    _, info = build(args.dataset, args.wyniki, args.model, args.out)
    print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
