"""Fig. S6 | Partner molecules often have the deeper history.

For each of the fifteen arrays (A1) and two populations of pairs (all pairs; product-forming pairs): for every pair, the
number of other partners its owner molecule has within the population, and the same for its partner molecule (molecules by
canonical SMILES); the mean over the pairs of the array. This is the design-only count of the frozen exposure table
(ED_owner_partner_exposure_depth.csv), checked before anything is drawn. No yield beyond the product-forming gate enters;
no random step.
Input with a measurement: A1.
    python 06_CODE/Figures/ed06.py [--out DIR]
"""
import argparse

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

import ed_common as E
from ed_common import fs
import anatomy
import history

RDLogger.DisableLog("rdApp.*")
N = 6
FROZEN = E.DATA / "Results/directional/ED_owner_partner_exposure_depth.csv"
TITLE = "Partner molecules often have the deeper history"
TITLES = dict(a="All pairs", b="Product-forming pairs")
POP = dict(a="all pairs", b="product-forming pairs")
TRANSFORM = ("For each pair of the population (all pairs of the array, or its product-forming pairs), the number of other partners "
             "its owner molecule has within that population, and the same for its partner molecule (molecules by canonical "
             "SMILES); mean over the pairs. Owner = the role of Fig. 1F. 'Partner deeper' when the partner's mean is the larger.")
H, NAME_R, AX_W, PITCH, GAP_G = 74.0, 33.0, 52.0, 3.0, 1.4
PANEL_X = dict(a=0.0, b=92.0)
TOP = 15.0


def can(s, _c={}):
    if s not in _c:
        _c[s] = Chem.MolToSmiles(Chem.MolFromSmiles(str(s)), isomericSmiles=True)
    return _c[s]


def compute(A1, owner):
    pf = (A1.groupby(["screen", "subpair"])["yield"].max() >= anatomy.FORMS).rename("pf").reset_index()
    pf["m1"] = pf.subpair.str.split(" >> ").str[0].map(can); pf["m2"] = pf.subpair.str.split(" >> ").str[1].map(can)
    ref = pd.read_csv(FROZEN).set_index(["array", "population"])
    rows = []
    for key in sorted(pf.screen.unique()):
        own = owner[key]
        for pop, g in (("all pairs", pf[pf.screen == key]), ("product-forming pairs", pf[(pf.screen == key) & pf.pf])):
            d1 = g.m1.map(g.groupby("m1").m2.nunique()) - 1; d2 = g.m2.map(g.groupby("m2").m1.nunique()) - 1
            do, dp = (d1, d2) if own == 1 else (d2, d1)
            r = dict(screen=key, population=pop, pairs=len(g), owner_mean=float(do.mean()), partner_mean=float(dp.mean()))
            f = ref.loc[(key, pop)]
            assert int(f.targets) == len(g) and abs(round(r["owner_mean"], 3) - f.owner_mean_other_partners) < 1e-9
            assert abs(round(r["partner_mean"], 3) - f.partner_mean_other_partners) < 1e-9
            assert ("shallower" if r["owner_mean"] < r["partner_mean"] else "deeper" if r["owner_mean"] > r["partner_mean"] else "equal") == f.owner_side
            rows.append(r)
    T = pd.DataFrame(rows)
    T["ratio"] = T.partner_mean / T.owner_mean
    counts = {pop: int((g.owner_mean < g.partner_mean).sum()) for pop, g in T.groupby("population")}
    sh = T[(T.population == "product-forming pairs") & (T.owner_mean < T.partner_mean)]
    rng = (float(sh.ratio.min()), float(sh.ratio.max()))
    assert counts == {"all pairs": 9, "product-forming pairs": 7} and (round(rng[0], 1), round(rng[1], 1)) == (1.8, 4.3)
    return T, counts, rng


def draw(T, counts, order, group):
    fig, cv = E.figure(H, "ed6")
    fs.title(fig, E.LEFT, 0.6, TITLE, H).set_gid("title|main")
    for j, (side, colr) in enumerate((("owner", fs.C["owner"]), ("partner", fs.C["partner"]))):
        x = 120.0 + 22.0 * j
        cv.plot([x + 0.7], [0.6 + 2.3], marker="o", ms=2.6, mew=0, mfc=colr, ls="none", gid=f"key|{side}")
        cv.text(x + 2.0, 0.6 + 2.3, f"{side} molecule", ha="left", va="center", fontsize=fs.PT["tiny"], gid=f"key_label|{side}")
    ys, y_ = {}, PITCH / 2
    for i, k in enumerate(order):
        if i and group[k] != group[order[i - 1]]:
            y_ += GAP_G
        ys[k] = y_; y_ += PITCH
    hb = y_ + 0.4
    G = T.set_index(["screen", "population"])
    for L in ("a", "b"):
        x0 = PANEL_X[L]
        fs.letter(fig, x0, 7.0, L, H).set_gid(f"letter|{L}")
        fs.title(fig, x0 + 4.2, 7.35, TITLES[L], H).set_gid(f"title|{L}")
        ax = fs.axes_mm(fig, x0 + NAME_R + 1.5, TOP + 3.0, AX_W, hb, H); ax.set_gid(f"axes|{L}")
        ax.set_xlim(0, 14); ax.set_ylim(hb, 0)
        for k in order:
            r = G.loc[(k, POP[L])]
            ax.plot([r.owner_mean, r.partner_mean], [ys[k], ys[k]], color=fs.C["faint"], lw=0.6, zorder=1, gid=f"{L}|link|{k}")
            ax.plot([r.owner_mean], [ys[k]], marker="o", ms=2.6, mew=0, mfc=fs.C["owner"], ls="none", zorder=3, gid=f"{L}|owner|{k}")
            ax.plot([r.partner_mean], [ys[k]], marker="o", ms=2.6, mew=0, mfc=fs.C["partner"], ls="none", zorder=3, gid=f"{L}|partner|{k}")
            cv.text(x0 + NAME_R, TOP + 3.0 + ys[k], fs.ARRAY_LABEL[k], ha="right", va="center", fontsize=fs.PT["tiny"], gid=f"{L}|name|{k}")
        cv.text(x0 + NAME_R + 1.5 + AX_W, TOP + 0.6, f"owner shallower in {counts[POP[L]]} of 15", ha="right", va="center",
                fontsize=fs.PT["tiny"], color=fs.C["muted"], gid=f"{L}|count")
        ax.set_xticks([0, 5, 10]); ax.set_yticks([])
        ax.set_xlabel("Mean number of other partners" if L == "a" else "Mean number of other product-forming partners", labelpad=1.5)
        E.clean(ax, keep=("bottom",))
    return fig


def source_data(T, counts, rng, order, group, roles, owner):
    G = T.set_index(["screen", "population"])
    rows = [dict(screen=k, array=fs.ARRAY_LABEL[k], group=group[k], population=pop, owner_role=roles[k][owner[k] - 1],
                 partner_role=roles[k][2 - owner[k]], pairs=int(G.loc[(k, pop), "pairs"]), owner_mean_other_partners=float(G.loc[(k, pop), "owner_mean"]),
                 partner_mean_other_partners=float(G.loc[(k, pop), "partner_mean"]),
                 partner_over_owner=float(G.loc[(k, pop), "ratio"])) for pop in ("all pairs", "product-forming pairs") for k in order]
    s = pd.DataFrame([dict(statistic="arrays where the owner has the shallower history, all pairs", value=counts["all pairs"], of=15),
                      dict(statistic="arrays where the owner has the shallower history, product-forming pairs",
                           value=counts["product-forming pairs"], of=15),
                      dict(statistic="partner over owner in those arrays (product-forming), lowest", value=rng[0], of=None),
                      dict(statistic="partner over owner in those arrays (product-forming), highest", value=rng[1], of=None)])
    readme = pd.DataFrame([dict(sheet="ED6", estimand="design-only history depth (counts of other partners)", rows="per population, one "
                                "per array in the order of Figure 1F", units="mean count of other partners", transformation=TRANSFORM),
                           dict(sheet="ED6_counts", estimand="counts quoted in the legend", rows="one per statistic", units="counts; ratios",
                                transformation="From the ED6 sheet."),
                           dict(sheet="(all)", estimand="-", rows="-", units="-",
                                transformation="Measurements: DATASET_analysis.csv.gz (A1), used only for the product-forming gate (best "
                                               "measured yield at least 10%). Owner roles: Anatomy_arrays.csv. Input hashes: "
                                               "FigS6_SourceData_provenance.json.")])
    return dict(README=readme, ED6=pd.DataFrame(rows), ED6_counts=s.astype({"of": "Int64"}))


def main(out=None):
    A1, _ = E.public_units()
    owner = E.owner_roles()
    T, counts, rng = compute(A1, owner)
    order, group = E.order()
    roles = {k: (history.ROLE[k][0][0], history.ROLE[k][1][0]) for k in order}
    fig = draw(T, counts, order, group)
    prov = dict(figure=f"Fig. S{N} | {TITLE}", generated_by=E.rel(__file__),
                inputs=dict(A1=dict(file=E.rel(E.RAW), sha256=E.sha_file(E.RAW), rows=int(len(A1))),
                            frozen=dict(file=E.rel(FROZEN), sha256=E.sha_file(FROZEN)),
                            owner_roles=dict(file=E.rel(E.ANATOMY_ARRAYS), sha256=E.sha_file(E.ANATOMY_ARRAYS))),
                randomness="none", transformation=TRANSFORM, owner_shallower=counts, partner_over_owner_range=list(rng))
    p = E.write(N, fig, H, source_data(T, counts, rng, order, group, roles, owner), prov, out)
    print(f"Fig. S{N}: owner shallower {counts}; ratios {rng[0]:.2f}-{rng[1]:.2f}; wrote {len(p)} files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    main(ap.parse_args().out)
