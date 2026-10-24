"""Fig. S2 | Pairs that share a molecule have more similar yield profiles in every array.

For the product-forming pairs of each of the fifteen arrays (A1): additive residuals (history.additive_residual, yield minus
pair and condition means) and the mean Pearson correlation of residual profiles between pairs that share only the role-1
molecule, only the role-2 molecule, or neither (history.pair_agreement). Checked against Anatomy_arrays.csv and
FigureS2_profile_correlations.csv before anything is drawn. No random step.
Input with a measurement: A1.
    python 06_CODE/Figures/ed02.py [--out DIR]
"""
import argparse

import numpy as np
import pandas as pd

import ed_common as E
from ed_common import fs
import anatomy
import history

N = 2
F1C = E.DATA / "Results/display_inputs/FigureS2_profile_correlations.csv"
TITLE = "Pairs that share a molecule have more similar yield profiles in every screen"
NOTE = "Analysis screens, product-forming pairs; additive residual yield profiles"
TRANSFORM = ("Product-forming pairs (best measured yield at least 10%) of each array. Additive residual = yield minus the pair's and "
             "the condition's means (fitted jointly). For every two pairs, the Pearson correlation of their residual profiles over "
             "shared conditions; mean over the pairs that share only the role-1 molecule, only the role-2 molecule, or neither.")
H, TOP, NAME_R, AX_X, AX_W, PITCH, GAP_G, ROLE_X = 78.0, 13.0, 40.0, 41.5, 92.0, 3.3, 1.6, 137.0
KIND = (("share1", "only the role-1 molecule", "o", fs.C["text"]), ("share2", "only the role-2 molecule", "s", fs.C["control"]),
        ("neither", "neither molecule", "o", fs.C["faint"]))


def compute(A1):
    AR = history.partner_arrays(A1)
    rows = []
    for key, A in AR.items():
        act = np.nanmax(A["M"], 1) >= anatomy.FORMS
        R = history.additive_residual(A["M"][act])[0]
        g1, g2, g0 = history.pair_agreement(R, A["a"][act], A["b"][act])
        rows.append(dict(screen=key, share1=float(g1), share2=float(g2), neither=float(g0), product_forming_pairs=int(act.sum())))
    T = pd.DataFrame(rows).set_index("screen")
    an = pd.read_csv(E.ANATOMY_ARRAYS).set_index("screen")
    f1 = pd.read_csv(F1C).set_index("screen")
    dev = max(float((T[c] - an.loc[T.index, f"r_{n}"]).abs().max()) for c, n in (("share1", "share1"), ("share2", "share2"), ("neither", "neither")))
    dev = max(dev, max(float((T[c] - f1.loc[T.index, c]).abs().max()) for c in ("share1", "share2", "neither")))
    assert len(T) == 15 and dev < E.TOL, dev
    med = {c: float(T[c].median()) for c in ("share1", "share2", "neither")}
    either = int(((T.share1 > T.neither) | (T.share2 > T.neither)).sum()); both = int(((T.share1 > T.neither) & (T.share2 > T.neither)).sum())
    return T, med, either, both, dev


def draw(T, med, either, both, order, group, roles):
    fig, cv = E.figure(H, "ed2")
    E.heading(fig, cv, 0, 0.6, None, TITLE, H, NOTE)
    for j, (c, lab, m, colr) in enumerate(KIND):                      # key
        x = AX_X + 33.0 * j
        cv.plot([x + 0.7], [TOP - 3.2], marker=m, ms=2.6, mew=0, mfc=colr, ls="none", gid=f"key|{c}")
        cv.text(x + 2.0, TOP - 3.2, f"pairs sharing {lab}", ha="left", va="center", fontsize=fs.PT["tiny"], gid=f"key_label|{c}")
    ys, y_ = {}, PITCH / 2
    for i, k in enumerate(order):
        if i and group[k] != group[order[i - 1]]:
            y_ += GAP_G
        ys[k] = y_; y_ += PITCH
    y_med = y_ + GAP_G + PITCH / 2 - PITCH / 2
    hb = y_med + PITCH / 2 + 0.4
    ax = fs.axes_mm(fig, AX_X, TOP, AX_W, hb, H); ax.set_gid("axes")
    ax.set_xlim(-0.2, 0.75); ax.set_ylim(hb, 0)
    ax.plot([0, 0], [0, hb], color=fs.C["faint"], lw=0.5, zorder=0, gid="zero")
    for k in order:
        for c, _, m, colr in KIND:
            ax.plot([T.loc[k, c]], [ys[k]], marker=m, ms=2.6, mew=0, mfc=colr, ls="none", zorder=3, gid=f"point|{k}|{c}")
        cv.text(NAME_R, TOP + ys[k], fs.ARRAY_LABEL[k], ha="right", va="center", fontsize=fs.PT["tiny"], gid=f"name|{k}")
        cv.text(ROLE_X, TOP + ys[k], f"1: {roles[k][0]}  ·  2: {roles[k][1]}", ha="left", va="center", fontsize=fs.PT["tiny"],
                color=fs.C["muted"], gid=f"roles|{k}")
    for c, _, m, colr in KIND:
        ax.plot([med[c]], [y_med], marker=m, ms=3.4, mew=0, mfc=colr, ls="none", zorder=3, gid=f"median|{c}")
    cv.text(NAME_R, TOP + y_med, "Median over screens", ha="right", va="center", fontsize=fs.PT["tiny"], fontweight="bold", gid="name|median")
    cv.text(ROLE_X, TOP + y_med, f"either above neither in {either} of 15; both in {both}", ha="left", va="center",
            fontsize=fs.PT["tiny"], color=fs.C["muted"], gid="counts")
    ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6]); ax.set_yticks([])
    ax.set_xlabel("Mean correlation of yield profiles between product-forming pairs", labelpad=1.5)
    E.clean(ax, keep=("bottom",))
    return fig


def source_data(T, med, either, both, order, group, roles):
    rows = [dict(row_type="array", screen=k, array=fs.ARRAY_LABEL[k], group=group[k], role1=roles[k][0], role2=roles[k][1],
                 product_forming_pairs=int(T.loc[k, "product_forming_pairs"]), share_role1_only=float(T.loc[k, "share1"]),
                 share_role2_only=float(T.loc[k, "share2"]), share_neither=float(T.loc[k, "neither"])) for k in order]
    rows.append(dict(row_type="median over arrays", share_role1_only=med["share1"], share_role2_only=med["share2"],
                     share_neither=med["neither"], arrays_either_above_neither=either, arrays_both_above_neither=both))
    d = pd.DataFrame(rows).astype({"product_forming_pairs": "Int64", "arrays_either_above_neither": "Int64",
                                   "arrays_both_above_neither": "Int64"})
    readme = pd.DataFrame([dict(sheet="ED2", estimand="mean correlation of additive residual yield profiles", rows="one per array in the "
                                "order of Figure 1F, then the medians", units="Pearson correlation; pairs", transformation=TRANSFORM),
                           dict(sheet="(all)", estimand="-", rows="-", units="-",
                                transformation="Measurements: DATASET_analysis.csv.gz (A1). Role names as in Figure 1F. Input hashes: "
                                               "FigS2_SourceData_provenance.json.")])
    return dict(README=readme, ED2=d)


def main(out=None):
    A1, _ = E.public_units()
    T, med, either, both, dev = compute(A1)
    order, group = E.order()
    roles = {k: (history.ROLE[k][0][0], history.ROLE[k][1][0]) for k in order}
    fig = draw(T, med, either, both, order, group, roles)
    prov = dict(figure=f"Fig. S{N} | {TITLE}", generated_by=E.rel(__file__),
                inputs=dict(A1=dict(file=E.rel(E.RAW), sha256=E.sha_file(E.RAW), rows=int(len(A1))),
                            frozen={n: dict(file=E.rel(p), sha256=E.sha_file(p)) for n, p in (("Anatomy_arrays.csv", E.ANATOMY_ARRAYS),
                                                                                               ("FigureS2_profile_correlations.csv", F1C))}),
                randomness="none", transformation=TRANSFORM, medians=med, arrays_either_above_neither=either,
                arrays_both_above_neither=both, agreement_with_frozen=float(dev))
    p = E.write(N, fig, H, source_data(T, med, either, both, order, group, roles), prov, out)
    print(f"Fig. S{N}: medians {med}; either {either}, both {both}; max |diff| {dev:.1e}; wrote {len(p)} files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    main(ap.parse_args().out)
