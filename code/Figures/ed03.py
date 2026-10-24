"""Fig. S3 | No single molecule decides an owner.

From the product-forming pairs of the fifteen arrays (A1) and the frozen cohort file (DC1, DESIGN_COUNTS.csv), with the
Phase 1 estimators (array_diagnostics_data.molecule_T, anatomy.pattern and anatomy.Shares):
  a     T of the 38 molecules that own in at least two reactions and of the 15 that are never the owner;
  b     D = mean T as owner minus mean T as non-owner for the 29 molecules of the all-context set, with its interval;
  c     55 refits of both role shares with each of the 23 molecules of Fig. 3B removed from every array it uses;
  d     the 30 role shares of Fig. 2A recombined from molecule-level T.
The rank correlation of T between the two owner reactions of the 38 molecules is checked against the frozen summary (it is
reported in Supplementary Note 7, not drawn).
Everything is checked against the frozen Phase 1 outputs (phase1_T.csv, phase1_lomo.csv, phase1_check.csv,
phase1_summary.json) before anything is drawn. The interval of b comes from one generator (seed 20261004) that the frozen
run first used for four other intervals (23 molecules, amines, the balanced mean, the library contrast); those draws are
replayed, and the 10,000 resamples of b are written out (FigS3_all_context_draws.csv).
Inputs with a measurement: A1; the cohort file DC1.
    python 06_CODE/Figures/ed03.py [--out DIR]
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import ed_common as E
from ed_common import fs
from anatomy import Shares, pattern
from array_diagnostics_data import boot_ci, can, condition_patterns, molecule_T

N = 3
COHORTS = E.DATA / "Analysis/directional_cohorts/DESIGN_COUNTS.csv"
RES = E.DATA / "Results/directional"
APPROVED = {n: RES / n for n in ("phase1_T.csv", "phase1_lomo.csv", "phase1_check.csv", "phase1_summary.json")}
SEED, NB = 20261004, 10000
TITLE = "No single molecule decides an owner"
TITLES = dict(a="Owners carry more than non-owners", b="All contexts, clear owner or not",
              c="Removing one molecule keeps every owner", d="Molecule-level T rebuilds the shares")
TRANSFORM = dict(
    T="T of a molecule in one role of one array: mean product of the condition patterns (residual after an intercept and slope on "
      "the condition means, product-forming pairs) over its ordered pairs of pairs sharing a condition, divided by the array's "
      "mean pattern variance.",
    a="T over all owner contexts of the 38 molecules with at least two owner contexts and no non-owner context among arrays "
      "with a clear owner (DC1), and over all non-owner contexts of the 15 molecules with at least two non-owner contexts and no "
      "owner context.",
    b="The 29 molecules of the G1 and G1_all_only sets of DC1, all arrays: D = mean T as owner - mean T as non-owner. Interval: "
      "10,000 resamples of the 29 molecules with default_rng(20261004) after the four earlier Phase 1 intervals (the exact "
      "resamples are in FigS3_all_context_draws.csv); 2.5th and 97.5th percentiles of the mean.",
    c="For each of the 23 molecules of Fig. 3B and each array of its contexts, every pair containing it is removed and both role "
      "shares refitted; margin = owner share - partner share after the refit (owner as in Fig. 2A).",
    d="Recombined share of a role = sum over its molecules of n x T / sum of n, n = ordered pairs of pairs; published = Fig. 2A.")
H = 118.0


def per_molecule(rows, Tmap):
    """Phase 1: per molecule, mean T over its owner contexts minus mean T over its non-owner contexts."""
    out = []
    for m, h in rows.groupby("molecule"):
        o, n = h[h.context == "owner"], h[h.context == "non-owner"]
        if o.empty or n.empty:
            continue
        TO = np.mean([Tmap[(m, r.array, r.role)] for r in o.itertuples()])
        TN = np.mean([Tmap[(m, r.array, r.role)] for r in n.itertuples()])
        out.append(dict(molecule=m, stratum=h.stratum.iloc[0], T_O=TO, T_N=TN, D=TO - TN))
    return pd.DataFrame(out)


def compute(A1, DC):
    pub = pd.read_csv(E.ANATOMY_ARRAYS).set_index("screen")
    X, trows, check = {}, [], []
    for key in sorted(A1.screen.unique()):                                          # Gate 0 and T over the fifteen arrays
        X[key] = condition_patterns(A1, key)
        x = X[key]
        cls = Shares(x["E"], x["a"], x["b"]).compute(np.ones(len(x["a"])))
        for r, keys in ((1, x["a"]), (2, x["b"])):
            t = molecule_T(x["E"], keys)
            n = np.array([v[1] for v in t.values()], float); T = np.array([v[0] for v in t.values()])
            check.append(dict(screen=key, role=r, published=float(pub.loc[key, f"share{r}"]), recombined=float((n * T).sum() / n.sum()),
                              class_estimate=float(cls[str(r)])))
            trows += [dict(molecule=can(u), array=key, role=r, T=tv, n_ordered_pair_pairs=nn) for u, (tv, nn) in t.items()]
    TT, CH = pd.DataFrame(trows), pd.DataFrame(check)
    ref = pd.read_csv(APPROVED["phase1_T.csv"])
    J = TT.merge(ref, on=["molecule", "array", "role"], how="outer", suffixes=("", "_f"), indicator=True)
    assert (J._merge == "both").all() and float((J["T"] - J["T_f"]).abs().max()) < E.TOL
    CHf = pd.read_csv(APPROVED["phase1_check.csv"]).set_index(["array", "role"])
    assert len(CH) == 30 and all(abs(r.recombined - CHf.loc[(r.screen, r.role), "recombined"]) < E.TOL for r in CH.itertuples())
    CH["abs_diff"] = (CH.recombined - CH.published).abs()
    assert (CH.abs_diff < E.TOL).all()
    Tmap = {(r.molecule, r.array, r.role): r.T for r in TT.itertuples()}
    S = json.loads(APPROVED["phase1_summary.json"].read_text(encoding="utf-8"))

    # the frozen generator, replayed in its order: A (23), amines (19), balanced (19, 4), library contrast (8), then all contexts
    rng = np.random.default_rng(SEED)
    PRIM = DC[(DC.set == "G1") & DC.array_clear_owner]
    PM = per_molecule(PRIM, Tmap)
    am, ac = PM[PM.stratum == "amine"].D.to_numpy(), PM[PM.stratum == "acid"].D.to_numpy()
    assert (len(PM), len(am), len(ac)) == (23, 19, 4)
    ci_A, ci_am = boot_ci(PM.D, rng), boot_ci(am, rng)
    bal = 0.5 * am[rng.integers(0, 19, (NB, 19))].mean(1) + 0.5 * ac[rng.integers(0, 4, (NB, 4))].mean(1)
    TRIPLE = sorted(DC[DC.triple_context].molecule.unique())
    lib = np.array([Tmap[(m, "Buchwald-Hartwig|1", 1)] - Tmap[(m, "Buchwald-Hartwig|2", 1)] for m in TRIPLE])
    ci_lib = boot_ci(lib, rng)
    ALL = per_molecule(DC[DC.set.isin(["G1", "G1_all_only"])].assign(stratum=lambda x: x.stratum.fillna("")), Tmap)
    assert len(ALL) == 29
    idx = rng.integers(0, len(ALL), (NB, len(ALL)))
    means = ALL.D.to_numpy()[idx].mean(1)
    ci_all = [float(v) for v in np.percentile(means, [2.5, 97.5])]
    for mine, theirs in ((ci_A, S["A"]["ci"]), (ci_am, S["B"]["amine_ci"]), ([float(v) for v in np.percentile(bal, [2.5, 97.5])], S["B"]["balanced"]["ci"]),
                         (ci_lib, S["C"]["library_contrast_ci"]), (ci_all, S["descriptives"]["all_contexts"]["ci"])):
        assert np.allclose(mine, theirs, atol=1e-12, rtol=0), (mine, theirs)
    assert abs(ALL.D.mean() - S["descriptives"]["all_contexts"]["mean"]) < E.TOL and int((ALL.D > 0).sum()) == S["descriptives"]["all_contexts"]["positive"]

    # owner-owner and never-owner molecules, by the frozen rule
    rule = DC[DC.array_clear_owner].groupby("molecule").context.agg(list)
    OO_m = sorted(m for m, c in rule.items() if c.count("owner") >= 2 and "non-owner" not in c)
    NN_m = sorted(m for m, c in rule.items() if c.count("non-owner") >= 2 and "owner" not in c)
    OO = DC[DC.molecule.isin(OO_m) & DC.array_clear_owner & (DC.context == "owner")]
    NN = DC[DC.molecule.isin(NN_m) & DC.array_clear_owner & (DC.context == "non-owner")]
    pairs = []
    for m, h in OO.groupby("molecule"):
        h = h.sort_values(["n_pf_partners", "array"], ascending=[False, True]).head(2)
        r1, r2 = h.iloc[0], h.iloc[1]
        pairs.append(dict(molecule=m, array_1=r1["array"], T_1=Tmap[(m, r1["array"], r1["role"])], array_2=r2["array"],
                          T_2=Tmap[(m, r2["array"], r2["role"])]))
    PA = pd.DataFrame(pairs)
    rho = spearmanr(PA.T_1, PA.T_2)
    TB = pd.concat([pd.DataFrame([dict(group="owner in two or more reactions (38 molecules)", molecule=r.molecule, array=r.array,
                                       T=Tmap[(r.molecule, r.array, r.role)]) for r in OO.itertuples()]),
                    pd.DataFrame([dict(group="never the owner (15 molecules)", molecule=r.molecule, array=r.array,
                                       T=Tmap[(r.molecule, r.array, r.role)]) for r in NN.itertuples()])], ignore_index=True)
    d = S["descriptives"]
    assert (len(PA), NN.molecule.nunique()) == (d["owner_owner"]["n"], d["nonowner_nonowner"]["n"]) == (38, 15)
    assert abs(rho.statistic - d["owner_owner"]["spearman"]) < E.TOL
    assert abs(TB[TB.group.str.startswith("owner")]["T"].mean() - d["owner_owner"]["mean_T"]) < E.TOL
    assert abs(TB[TB.group.str.startswith("never")]["T"].mean() - d["nonowner_nonowner"]["mean_T"]) < E.TOL

    # leave one molecule out: every context array of the 23 molecules
    lomo = []
    for m, h in PRIM.groupby("molecule"):
        for arr in sorted(h.array):
            x = X[arr]
            Y = x["M"][x["act"]]
            raws = [u for u in set(x["a"]) | set(x["b"]) if can(u) == m]
            keep = ~(np.isin(x["a"], raws) | np.isin(x["b"], raws))
            s = Shares(pattern(Y[keep])[0], x["a"][keep], x["b"][keep]).compute(np.ones(int(keep.sum())))
            own = int(pub.loc[arr, "owner_role"])
            lomo.append(dict(molecule=m, array=arr, owner_role=own, share1_before=float(pub.loc[arr, "share1"]), share2_before=float(pub.loc[arr, "share2"]),
                             share1_after=float(s["1"]), share2_after=float(s["2"]), owner_after=1 if s["1"] >= s["2"] else 2,
                             pf_pairs_before=len(Y), pf_pairs_after=int(keep.sum())))
    LO = pd.DataFrame(lomo)
    LO["margin_after"] = np.where(LO.owner_role == 1, LO.share1_after - LO.share2_after, LO.share2_after - LO.share1_after)
    LO["largest_share_change"] = np.maximum((LO.share1_after - LO.share1_before).abs(), (LO.share2_after - LO.share2_before).abs())
    LOf = pd.read_csv(APPROVED["phase1_lomo.csv"]).set_index(["molecule", "array"])
    assert len(LO) == len(LOf) == 55 and all(abs(r.share1_after - LOf.loc[(r.molecule, r.array), "share1_after"]) < E.TOL
                                             and abs(r.share2_after - LOf.loc[(r.molecule, r.array), "share2_after"]) < E.TOL for r in LO.itertuples())
    assert (LO.owner_after == LO.owner_role).all()
    draws = pd.DataFrame(dict(draw=np.arange(1, NB + 1), molecules=[";".join(str(i + 1) for i in row) for row in idx], mean_D=means))
    return dict(CH=CH, PA=PA, rho=float(rho.statistic), TB=TB, ALL=ALL, ci_all=ci_all, LO=LO, draws=draws)


def swarm(vals, dy, dx):
    placed, out = [], {}
    for i in np.argsort(vals, kind="stable"):
        for off in [0.0] + [s_ * k * dx for k in range(1, 30) for s_ in (1, -1)]:
            if all(abs(vals[i] - pv) >= dy or abs(off - po) >= dx for pv, po in placed):
                placed.append((vals[i], off)); out[i] = off
                break
    return np.array([out[i] for i in range(len(vals))])


def draw(R):
    fig, cv = E.figure(H, "ed3")
    owner_c, partner_c, faint, muted, text_c = fs.C["owner"], fs.C["partner"], fs.C["faint"], fs.C["muted"], fs.C["text"]
    fs.title(fig, E.LEFT, 0.6, TITLE, H).set_gid("title|main")
    cv.text(E.LEFT, 6.4, "Product-forming pairs in the analysed screens; shared molecules across reactions", ha="left", va="center",
            fontsize=fs.PT["tiny"], color=muted, gid="note|main")
    pos = dict(a=(0, 12.0), b=(88.0, 12.0), c=(0, 66.0), d=(122.0, 66.0))
    for L, (x, y) in pos.items():
        fs.letter(fig, x, y, L, H).set_gid(f"letter|{L}")
        fs.title(fig, x + 4.2, y + 0.35, TITLES[L], H).set_gid(f"title|{L}")
    # a
    TB = R["TB"]
    bx = fs.axes_mm(fig, 14.0, 20.0, 46.0, 34.0, H); bx.set_gid("a|axes")
    bx.set_xlim(-0.6, 1.6); bx.set_ylim(-0.45, 1.65)                         # every context inside (-0.36 to 1.57)
    bx.axhline(0, color=faint, lw=0.5, zorder=0, gid="a|zero")
    for i, (g, colr) in enumerate((("owner", owner_c), ("never", partner_c))):
        sub = TB[TB.group.str.startswith(g)].reset_index(drop=True)
        off = swarm(sub["T"].to_numpy(), 0.042, 0.06)
        for j, r in sub.iterrows():
            bx.plot([i + off[j]], [r["T"]], marker="o", ms=1.8, mew=0, mfc=colr, ls="none", zorder=3, gid=f"a|point|{g}|{j}")
        m = float(sub["T"].mean())
        bx.plot([i - 0.38, i + 0.38], [m, m], color=text_c, lw=1.2, zorder=4, gid=f"a|mean|{g}")
        bx.text(i + 0.42, m, f"{m:.3f}", ha="left", va="center", fontsize=fs.PT["tiny"], gid=f"a|mean_value|{g}")
    bx.set_xticks([0, 1], ["owner in ≥2\nreactions (38)", "never the\nowner (15)"]); bx.set_yticks([0, 0.4, 0.8, 1.2, 1.6])
    bx.tick_params(axis="x", labelsize=fs.PT["tiny"])
    bx.set_ylabel("T", labelpad=1.5)
    E.clean(bx)
    # b
    ALL = R["ALL"].sort_values("D").reset_index(drop=True)
    cx = fs.axes_mm(fig, 102.0, 20.0, 52.0, 34.0, H); cx.set_gid("b|axes")
    cx.set_xlim(-0.6, 1.0); cx.set_ylim(-1, len(ALL) + 2.5)
    cx.axvline(0, color=faint, lw=0.5, zorder=0, gid="b|zero")
    for j, r in ALL.iterrows():
        cx.plot([r.D], [j], marker="o", ms=1.8, mew=0, mfc=owner_c if r.D > 0 else faint, ls="none", zorder=3, gid=f"b|point|{r.molecule}")
    mean_ = float(R["ALL"].D.mean()); lo, hi = R["ci_all"]
    yb = len(ALL) + 1.2
    cx.plot([lo, hi], [yb, yb], color=text_c, lw=0.8, zorder=4, gid="b|ci")
    cx.plot([mean_], [yb], marker="D", ms=3.0, mew=0, mfc=text_c, ls="none", zorder=5, gid="b|mean")
    cx.text(hi + 0.04, yb, f"{mean_:.3f} ({lo:.3f}–{hi:.3f})", ha="left", va="center", fontsize=fs.PT["tiny"], gid="b|mean_value")
    cx.text(1.0, 0.5, f"{int((R['ALL'].D > 0).sum())} of {len(ALL)} > 0", ha="right", va="center", fontsize=fs.PT["tiny"], color=muted,
            gid="b|positive")
    cx.set_xticks([-0.4, 0, 0.4, 0.8]); cx.set_yticks([])
    cx.set_xlabel("D = T as owner − T as non-owner", labelpad=1.5)
    E.clean(cx, keep=("bottom",))
    # c
    LO = R["LO"]
    arrays = [k for k in E.order()[0] if k in set(LO.array)]
    dx_ = fs.axes_mm(fig, 32.0, 74.0, 82.0, 4.0 * len(arrays), H); dx_.set_gid("c|axes")
    dx_.set_xlim(-0.05, 0.6); dx_.set_ylim(len(arrays) - 0.4, -0.6)            # every margin inside (0.12 to 0.54)
    dx_.axvline(0, color=faint, lw=0.5, zorder=0, gid="c|zero")
    for i, k in enumerate(arrays):
        sub = LO[LO.array == k].reset_index(drop=True)
        off = swarm(sub.margin_after.to_numpy(), 0.005, 0.08)
        for j, r in sub.iterrows():
            dx_.plot([r.margin_after], [i + off[j]], marker="o", ms=1.8, mew=0, mfc=owner_c, ls="none", zorder=3, gid=f"c|refit|{k}|{r.molecule}")
        own = int(sub.owner_role.iloc[0])
        full = sub.share1_before.iloc[0] - sub.share2_before.iloc[0] if own == 1 else sub.share2_before.iloc[0] - sub.share1_before.iloc[0]
        dx_.plot([full], [i], marker="|", ms=7, mew=1.0, mfc=text_c, mec=text_c, ls="none", zorder=2, gid=f"c|full|{k}")   # under the refits
        cv.text(30.0, 74.0 + 4.0 * (i + 0.6), f"{fs.ARRAY_LABEL[k]} ({len(sub)})", ha="right", va="center", fontsize=fs.PT["tiny"], gid=f"c|name|{k}")
    cv.text(4.2, 66.0 + 5.8, f"{len(LO)} refits, no owner changed; largest change in a role share {LO.largest_share_change.max():.3f}; "
            f"smallest owner margin {LO.margin_after.min():.3f}", ha="left", va="center", fontsize=fs.PT["tiny"], color=muted, gid="c|summary")
    dx_.set_xticks([0, 0.2, 0.4, 0.6]); dx_.set_yticks([])
    dx_.set_xlabel("Owner share − partner share after the refit (bar: full screen)", labelpad=1.5)
    E.clean(dx_, keep=("bottom",))
    # d
    CH = R["CH"]
    ex = fs.axes_mm(fig, 134.0, 74.0, 34.0, 30.0, H); ex.set_gid("d|axes")
    lim = (-0.15, 0.5)
    ex.set_xlim(*lim); ex.set_ylim(*lim)
    ex.plot(lim, lim, color=faint, lw=0.5, zorder=0, gid="d|diagonal")
    for r in CH.itertuples():
        ex.plot([r.published], [r.recombined], marker="o", ms=2.0, mew=0, mfc=owner_c, ls="none", zorder=3, gid=f"d|point|{r.screen}|{r.role}")
    ex.text(lim[1], lim[0] + 0.02, f"{len(CH)} of {len(CH)} role shares reproduced", ha="right", va="bottom", fontsize=fs.PT["tiny"], color=muted, gid="d|max")
    ex.set_xticks([0, 0.2, 0.4]); ex.set_yticks([0, 0.2, 0.4])
    ex.set_xlabel("Role share, Fig. 2A", labelpad=1.5); ex.set_ylabel("Recombined from T", labelpad=1.5)
    E.clean(ex)
    return fig


def source_data(R):
    ALL = R["ALL"].reset_index(drop=True)
    c = ALL.assign(molecule_number=np.arange(1, len(ALL) + 1))[["molecule_number", "molecule", "stratum", "T_O", "T_N", "D"]]
    c = pd.concat([c, pd.DataFrame([dict(molecule="mean over the 29", D=float(ALL.D.mean()), ci_lower=R["ci_all"][0], ci_upper=R["ci_all"][1],
                                         positive=int((ALL.D > 0).sum()))])], ignore_index=True).astype({"molecule_number": "Int64", "positive": "Int64"})
    a = R["TB"].assign(array=lambda d: d.array.map(fs.ARRAY_LABEL))
    b = c
    c = R["LO"].assign(array_name=lambda x: x.array.map(fs.ARRAY_LABEL))
    d = R["CH"].assign(array=lambda x: x.screen.map(fs.ARRAY_LABEL))
    readme = pd.DataFrame([
        dict(sheet="ED3a", panel="a", rows="one per molecule-array context", units="T", transformation=TRANSFORM["T"] + " " + TRANSFORM["a"]),
        dict(sheet="ED3b", panel="b", rows="one per molecule (29), numbered as in the draws file, then the mean and interval", units="T; D",
             transformation=TRANSFORM["b"]),
        dict(sheet="ED3c", panel="c", rows="one per refit (55)", units="shares", transformation=TRANSFORM["c"]),
        dict(sheet="ED3d", panel="d", rows="one per array and role (30)", units="shares", transformation=TRANSFORM["d"]),
        dict(sheet="FigS3_all_context_draws.csv (separate file)", panel="b",
             rows="one per resample (10,000): the 29 molecule numbers drawn (as in ED3b, ';' between them) and the resample's mean D",
             units="molecule numbers; D", transformation="The frozen analysis's own draws. The interval of b = the 2.5th and 97.5th "
                                                          "percentiles of mean_D (linear interpolation)."),
        dict(sheet="(all)", panel="-", rows="-", units="-",
             transformation="Measurements: DATASET_analysis.csv.gz (A1); cohorts: DESIGN_COUNTS.csv (DC1). Input hashes: "
                            "FigS3_SourceData_provenance.json.")])
    return dict(README=readme, ED3a=a, ED3b=b, ED3c=c, ED3d=d)


def main(out=None):
    A1, _ = E.public_units()
    DC = pd.read_csv(COHORTS)
    R = compute(A1, DC)
    fig = draw(R)
    prov = dict(figure=f"Fig. S{N} | {TITLE}", generated_by=E.rel(__file__),
                inputs=dict(A1=dict(file=E.rel(E.RAW), sha256=E.sha_file(E.RAW), rows=int(len(A1))),
                            DC1=dict(file=E.rel(COHORTS), sha256=E.sha_file(COHORTS)),
                            owner_roles=dict(file=E.rel(E.ANATOMY_ARRAYS), sha256=E.sha_file(E.ANATOMY_ARRAYS)),
                            frozen={n: dict(file=E.rel(p), sha256=E.sha_file(p)) for n, p in APPROVED.items()}),
                randomness="b only: default_rng(20261004) after the four earlier Phase 1 intervals (23 x 10,000, 19 x 10,000, "
                           "19 x 10,000 and 4 x 10,000, 8 x 10,000 integers); its 29 x 10,000 draws are released",
                transformation=TRANSFORM, all_context=dict(mean=float(R["ALL"].D.mean()), ci=R["ci_all"]))
    p = E.write(N, fig, H, source_data(R), prov, out, extra=(("draws", "FigS3_all_context_draws.csv"),), tables=(("draws", R["draws"]),))
    print(f"Fig. S{N}: all-context D {R['ALL'].D.mean():.3f} ({R['ci_all'][0]:.3f}-{R['ci_all'][1]:.3f}); rho {R['rho']:.3f}; "
          f"{len(R['LO'])} refits; wrote {len(p)} files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    main(ap.parse_args().out)
