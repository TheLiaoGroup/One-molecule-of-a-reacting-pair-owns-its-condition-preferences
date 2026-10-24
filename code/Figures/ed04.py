"""Fig. S4 | Either half of a condition library names the same owner.

For each of the fifteen arrays (A1): the product-forming pairs (gate of the full array); the conditions ordered by the
SHA-256 of their identifiers and dealt alternately into two halves; both role shares (anatomy.pattern and anatomy.Shares,
the estimator of Figure 1F) re-estimated within each half and on the full library. Checked against phase3_halves.csv and
Anatomy_arrays.csv before anything is drawn. No random step.
Input with a measurement: A1.
    python 06_CODE/Figures/ed04.py [--out DIR]
"""
import argparse
import hashlib

import numpy as np
import pandas as pd

import ed_common as E
from ed_common import fs
import anatomy
import history

N = 4
HALVES = E.DATA / "Results/directional/phase3_halves.csv"
TITLE = "Either half of a condition library names the same owner"
NOTE = "Analysis screens, product-forming pairs; conditions split into two halves in a fixed pseudo-random order"
TRANSFORM = ("Product-forming pairs of each array (gate of the full array). Conditions ordered by the SHA-256 of their identifiers and "
             "dealt alternately into halves A and B. In each half and in the full library, both role shares by the estimator of Fig. "
             "1d (condition pattern after an intercept and slope on the condition means; share of a role = mean product of the "
             "patterns of pairs sharing only that molecule / mean pattern variance). Margin = share of the full-array owner minus "
             "share of its partner; a half names the other role when its margin is below zero.")
H, TOP, NAME_R, AX_X, AX_W, PITCH, GAP_G, ROLE_X = 76.0, 13.0, 40.0, 41.5, 92.0, 3.3, 1.6, 137.0
MARK = (("full", "full library", "D", fs.C["text"], 2.8), ("A", "half A", "o", fs.C["owner"], 2.6), ("B", "half B", "o", fs.C["partner"], 2.6))
sha = lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()


def shares(Y, a, b):
    E_ = anatomy.pattern(Y)[0]
    s = anatomy.Shares(E_, a, b).compute(np.ones(len(Y)))
    return float(s["1"]), float(s["2"])


def compute(A1, owner):
    AR = history.partner_arrays(A1)
    ref = pd.read_csv(HALVES).set_index(["array", "half"])
    an = pd.read_csv(E.ANATOMY_ARRAYS).set_index("screen")
    rows, dev = [], 0.0
    for key, A in AR.items():
        pf = np.nanmax(A["M"], 1) >= anatomy.FORMS
        Y, a, b, conds = A["M"][pf], A["a"][pf], A["b"][pf], A["conds"]
        s1, s2 = shares(Y, a, b)
        dev = max(dev, abs(s1 - an.loc[key, "share1"]), abs(s2 - an.loc[key, "share2"]))
        rows.append(dict(screen=key, part="full", conditions=len(conds), share1=s1, share2=s2))
        order_ = sorted(range(len(conds)), key=lambda j: sha(str(conds[j])))
        for half, cols in (("A", sorted(order_[0::2])), ("B", sorted(order_[1::2]))):
            h1, h2 = shares(Y[:, cols], a, b)
            r = ref.loc[(key, half)]
            dev = max(dev, abs(h1 - r.share1), abs(h2 - r.share2))
            assert int(r.conditions) == len(cols)
            rows.append(dict(screen=key, part=half, conditions=len(cols), share1=h1, share2=h2))
    T = pd.DataFrame(rows)
    T["owner_share"] = [r.share1 if owner[r.screen] == 1 else r.share2 for r in T.itertuples()]
    T["partner_share"] = [r.share2 if owner[r.screen] == 1 else r.share1 for r in T.itertuples()]
    T["margin"] = T.owner_share - T.partner_share
    assert dev < E.TOL, dev
    halves = T[T.part != "full"]
    agree = int((halves.margin >= 0).sum())
    both = int(halves.groupby("screen").margin.apply(lambda m: (m >= 0).all()).sum())
    assert (agree, both) == (int(ref.agrees.sum()), int(ref.groupby(level=0).agrees.all().sum()))
    bh = {k: sorted(AR[k]["conds"]) for k in ("Buchwald-Hartwig|1", "Buchwald-Hartwig|2")}
    overlap = len(set(bh["Buchwald-Hartwig|1"]) & set(bh["Buchwald-Hartwig|2"]))
    return T, agree, both, overlap, dev


def draw(T, agree, both, order, group, roles, owner):
    fig, cv = E.figure(H, "ed4")
    E.heading(fig, cv, 0, 0.6, None, TITLE, H, NOTE)
    for j, (c, lab, m, colr, ms) in enumerate(MARK):
        x = AX_X + 24.0 * j
        cv.plot([x + 0.7], [TOP - 3.2], marker=m, ms=ms, mew=0, mfc=colr, ls="none", gid=f"key|{c}")
        cv.text(x + 2.0, TOP - 3.2, lab, ha="left", va="center", fontsize=fs.PT["tiny"], gid=f"key_label|{c}")
    ys, y_ = {}, PITCH / 2
    for i, k in enumerate(order):
        if i and group[k] != group[order[i - 1]]:
            y_ += GAP_G
        ys[k] = y_; y_ += PITCH
    hb = y_ + 0.4
    ax = fs.axes_mm(fig, AX_X, TOP, AX_W, hb, H); ax.set_gid("axes")
    ax.set_xlim(-0.25, 0.65); ax.set_ylim(hb, 0)
    ax.plot([0, 0], [0, hb], color=fs.C["faint"], lw=0.5, zorder=0, gid="zero")
    G = T.set_index(["screen", "part"])
    for k in order:
        for (c, _, m, colr, ms), dy in zip(MARK, (0.0, -0.75, 0.75)):
            ax.plot([G.loc[(k, c), "margin"]], [ys[k] + dy], marker=m, ms=ms, mew=0, mfc=colr, ls="none", zorder=3, gid=f"point|{k}|{c}")
        cv.text(NAME_R, TOP + ys[k], fs.ARRAY_LABEL[k], ha="right", va="center", fontsize=fs.PT["tiny"], gid=f"name|{k}")
        cv.text(ROLE_X, TOP + ys[k], f"owner: {roles[k][owner[k] - 1]}", ha="left", va="center", fontsize=fs.PT["tiny"],
                color=fs.C["muted"], gid=f"roles|{k}")
    cv.text(E.RIGHT, 0.6 + 5.8, f"{agree} of 30 halves and {both} of 15 screens keep the owner", ha="right", va="center",
            fontsize=fs.PT["tiny"], color=fs.C["muted"], gid="counts")
    ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6]); ax.set_yticks([])
    ax.set_xlabel("Owner share minus partner share (full-screen owner)", labelpad=1.5)
    E.clean(ax, keep=("bottom",))
    return fig


def source_data(T, agree, both, overlap, order, group, roles, owner):
    G = T.set_index(["screen", "part"])
    rows = []
    for k in order:
        for part in ("full", "A", "B"):
            r = G.loc[(k, part)]
            rows.append(dict(screen=k, array=fs.ARRAY_LABEL[k], group=group[k], owner_role=roles[k][owner[k] - 1],
                             partner_role=roles[k][2 - owner[k]], part=part, conditions=int(r.conditions), share_role1=float(r.share1),
                             share_role2=float(r.share2), owner_share=float(r.owner_share), partner_share=float(r.partner_share),
                             margin=float(r.margin)))
    d = pd.DataFrame(rows)
    s = pd.DataFrame([dict(statistic="halves naming the full-array owner", value=agree, of=30),
                      dict(statistic="arrays in which both halves name it", value=both, of=15),
                      dict(statistic="conditions shared by the two C–N libraries (ligand and metal)", value=overlap, of=None)])
    readme = pd.DataFrame([dict(sheet="ED4", estimand="role shares of the condition pattern (Fig. 1F estimator)", rows="per array in the "
                                "order of Figure 1F: the full library, then halves A and B", units="shares of pattern variance",
                                transformation=TRANSFORM),
                           dict(sheet="ED4_counts", estimand="counts quoted in the legend", rows="one per count", units="counts",
                                transformation="From the ED4 sheet; the C–N overlap counts identical condition identifiers."),
                           dict(sheet="(all)", estimand="-", rows="-", units="-",
                                transformation="Measurements: DATASET_analysis.csv.gz (A1). Owner roles: Anatomy_arrays.csv. Input "
                                               "hashes: FigS4_SourceData_provenance.json.")])
    return dict(README=readme, ED4=d, ED4_counts=s.astype({"of": "Int64"}))


def main(out=None):
    A1, _ = E.public_units()
    owner = E.owner_roles()
    T, agree, both, overlap, dev = compute(A1, owner)
    order, group = E.order()
    roles = {k: (history.ROLE[k][0][0], history.ROLE[k][1][0]) for k in order}
    fig = draw(T, agree, both, order, group, roles, owner)
    prov = dict(figure=f"Fig. S{N} | {TITLE}", generated_by=E.rel(__file__),
                inputs=dict(A1=dict(file=E.rel(E.RAW), sha256=E.sha_file(E.RAW), rows=int(len(A1))),
                            frozen={n: dict(file=E.rel(p), sha256=E.sha_file(p)) for n, p in (("phase3_halves.csv", HALVES),
                                                                                               ("Anatomy_arrays.csv", E.ANATOMY_ARRAYS))}),
                randomness="none (halves by SHA-256 of the condition identifiers)", transformation=TRANSFORM,
                halves_agree=agree, arrays_both_agree=both, bh_condition_overlap=overlap, agreement_with_frozen=float(dev))
    p = E.write(N, fig, H, source_data(T, agree, both, overlap, order, group, roles, owner), prov, out)
    print(f"Fig. S{N}: {agree}/30 halves, {both}/15 arrays; BH overlap {overlap}; max |diff| {dev:.1e}; wrote {len(p)} files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    main(ap.parse_args().out)
