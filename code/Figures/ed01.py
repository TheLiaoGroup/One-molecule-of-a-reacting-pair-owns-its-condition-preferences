"""Fig. S1 | The best condition on average is not each pair's best.

For each of the fifteen arrays (A1): the condition with the highest mean yield over pairs, and, over the pairs measured
under it, the mean of each pair's best measured yield minus its yield there (cost.opportunity, the estimator of the frozen
rule ac6798ca). Checked against fixed_condition_gap_15.json before anything is drawn. No random step.
Input with a measurement: A1 (DATASET_analysis.csv.gz, process-development rows dropped on reading).
    python 06_CODE/Figures/ed01.py             outputs in place
    python 06_CODE/Figures/ed01.py --out DIR   outputs into DIR
"""
import argparse
import json

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import ed_common as E
from ed_common import fs
from common import matrix
from cost import opportunity

N = 1
FROZEN = E.DATA / "Results/directional/fixed_condition_gap_15.json"
TITLE = "The best condition on average is not each pair's best"
NOTE = "Analysis screens; pairs measured under the screen's best-on-average condition"
TRANSFORM = ("For each array: the condition with the highest mean yield over pairs (first in identifier order if tied); over the "
             "pairs measured under it, the mean of (the pair's best measured yield - its yield under that condition), in percentage "
             "points. Median over the fifteen arrays.")
H, TOP, NAME_R, AX_X, AX_W, PITCH, GAP_G = 70.0, 10.0, 40.0, 41.5, 120.0, 3.3, 1.6


def compute(A1):
    ref = json.loads(FROZEN.read_text(encoding="utf-8"))
    rows = []
    for key, g in A1.groupby("screen"):
        pairs, conds, M = matrix(g)
        o = opportunity(M)
        rows.append(dict(screen=key, best_condition=conds[o["best_condition"]], pairs_measured=int(o["n_global_measured"]),
                         gap_pp=float(o["gap_pp"])))
    T = pd.DataFrame(rows).set_index("screen")
    per = ref["per_array_gap_pp"]
    assert set(T.index) == set(per) and all(abs(T.loc[k, "gap_pp"] - v) < E.TOL for k, v in per.items())
    med = float(np.median(T.gap_pp))
    assert abs(med - ref["median_gap_pp"]) < E.TOL and abs(T.gap_pp.min() - ref["min_gap_pp"]) < E.TOL
    assert abs(T.gap_pp.max() - ref["max_gap_pp"]) < E.TOL
    return T, med, max(abs(T.loc[k, "gap_pp"] - v) for k, v in per.items())


def draw(T, med, order, group):
    fig, cv = E.figure(H, "ed1")
    E.heading(fig, cv, 0, 0.6, None, TITLE, H, NOTE)
    ys, y_ = {}, PITCH / 2
    for i, k in enumerate(order):
        if i and group[k] != group[order[i - 1]]:
            y_ += GAP_G
        ys[k] = y_; y_ += PITCH
    hb = y_ + 0.4
    ax = fs.axes_mm(fig, AX_X, TOP, AX_W, hb, H); ax.set_gid("axes")
    ax.set_xlim(0, 38); ax.set_ylim(hb, 0)
    for k in order:
        v = T.loc[k, "gap_pp"]
        ax.add_patch(Rectangle((0, ys[k] - 1.05), v, 2.1, facecolor=fs.C["control"], edgecolor="none", gid=f"bar|{k}"))
        ax.text(v + 0.4, ys[k], f"{v:.1f}", ha="left", va="center", fontsize=fs.PT["tiny"], gid=f"value|{k}")
        cv.text(NAME_R, TOP + ys[k], fs.ARRAY_LABEL[k], ha="right", va="center", fontsize=fs.PT["tiny"], gid=f"name|{k}")
    ax.plot([med, med], [0, hb], color=fs.C["text"], lw=0.6, ls=(0, (2, 1.5)), zorder=0, gid="median")
    ax.text(med, -1.4, f"median {med:.1f}", ha="center", va="center", fontsize=fs.PT["tiny"], gid="median_label", clip_on=False)
    ax.set_xticks([0, 10, 20, 30]); ax.set_yticks([])
    ax.set_xlabel("Best measured yield minus yield under the best-on-average condition (percentage points)", labelpad=1.5)
    E.clean(ax, keep=("bottom",))
    return fig


def source_data(T, med, order, group):
    rows = [dict(row_type="array", screen=k, array=fs.ARRAY_LABEL[k], group=group[k], best_condition=T.loc[k, "best_condition"],
                 pairs_measured=int(T.loc[k, "pairs_measured"]), gap_pp=float(T.loc[k, "gap_pp"])) for k in order]
    rows.append(dict(row_type="median over arrays", gap_pp=med))
    d = pd.DataFrame(rows).astype({"pairs_measured": "Int64"})
    readme = pd.DataFrame([dict(sheet="ED1", estimand="fixed-condition gap", rows="one per array in the order of Figure 1F, then the "
                                "median", units="percentage points of yield; pairs", transformation=TRANSFORM),
                           dict(sheet="(all)", estimand="-", rows="-", units="-",
                                transformation="Measurements: DATASET_analysis.csv.gz (A1) of the data release. Per-array values are "
                                               "also in table S6. Input hashes: FigS1_SourceData_provenance.json.")])
    return dict(README=readme, ED1=d)


def main(out=None):
    A1, _ = E.public_units()
    T, med, dev = compute(A1)
    order, group = E.order()
    fig = draw(T, med, order, group)
    prov = dict(figure=f"Fig. S{N} | {TITLE}", generated_by=E.rel(__file__),
                inputs=dict(A1=dict(file=E.rel(E.RAW), sha256=E.sha_file(E.RAW), rows=int(len(A1))),
                            frozen=dict(file=E.rel(FROZEN), sha256=E.sha_file(FROZEN))),
                randomness="none", transformation=TRANSFORM, median_gap_pp=med, agreement_with_frozen=float(dev))
    p = E.write(N, fig, H, source_data(T, med, order, group), prov, out)
    print(f"Fig. S{N}: median {med:.2f} pp over {len(T)} arrays; max |diff| {dev:.1e}; wrote {', '.join(q.name for q in p.values())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    main(ap.parse_args().out)
