"""Fig. S10 | A structure-based recommender gains most where molecules have been measured.

a  The structure-based recommender under molecule-disjoint validation (regimes.py): per held-out pair and partition, best
   measured yield of the recommender's trio minus that of the complementary trio, by regime; pooled over the fifteen arrays
   with 10,000 whole-array draws (common.whole_array_interval, seed 20260920).
b  Structural similarity against the agreement of cross-fitted preference profiles (structure.run_association, profile
   p_oof_centred): Spearman rho per array, permutation p (4,999 label permutations, seed 20260920), Benjamini-Hochberg q
   over the 14 qualifying arrays.
The fits take tens of minutes, so this script aggregates the frozen per-case and per-array outputs (fig3_regime_cases.csv,
structure_association.csv) as structure_in_place_of_record_15.py does and checks them against its result; the fits are
repeated from the public measurements and the files released here by 06_CODE/Checks/verify_ed10_refit.py, whose record
(08_REVIEW_AND_QC/extended_data/ED10_REFIT_VERIFICATION.json) the Extended Data audit checks against these outputs.
The hidden state of the fits is released: the molecule groups of the five partitions (registry entries assignment|...), the
condition folds (conditions|m) and, for panel b, the halves of the pairs (halves|screen|20260920). The training rows of each regime
(registry entries regime|...) are checked here to equal the reference definition regimes._historical_training_rows, so they
follow from the groups.
Inputs with a measurement: A1. With P2_RELEASE_MODE=public (the rebuild of the public data package) the hidden state is read
from the three released files instead of the registry; each regime's training rows are then derived from the released groups
by the reference definition alone.
    python 06_CODE/Figures/ed10.py [--out DIR]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import ed_common as E
from ed_common import fs
from common import arrays, bh, whole_array_interval
import amplitude
import regimes as RG
import structure as S

N = 10
PUBLIC_MODE = os.environ.get("P2_RELEASE_MODE") == "public"
FOLDS_REG = E.DATA / "Analysis/06_METADATA/frozen_folds.json"
RELEASED = dict(groups=E.SDDIR / "FigS10_molecule_groups.csv", condition_folds=E.SDDIR / "FigS10_condition_folds.csv",
                halves=E.SDDIR / "FigS10_pair_halves.csv")                  # read in public mode only
APPROVED = dict(cases=E.DATA / "Results/primary/fig3_regime_cases.csv", association=E.DATA / "Results/primary/structure_association.csv",
                restated=E.DATA / "Results/directional/structure_in_place_of_record_15.json")
TITLE = "A structure-based recommender gains most where molecules have been measured"
TITLES = dict(a="A structure-based recommender, by which molecules are new", b="Within screens, structural similarity tracks condition preference")
NOTES = dict(a="Analysis-screen held-out pairs; molecule-disjoint validation over five partitions; 10,000 whole-screen draws",
             b="Fourteen qualifying analysis screens; cross-fitted preference profiles; 4,999 label permutations; Benjamini–Hochberg q")
REGIMES = (("warm/warm", "Both molecules measured in other pairs"), ("new/known", "Partner 1 new"), ("known/new", "Partner 2 new"),
           ("new/new", "Both new"))
TRANSFORM = dict(
    a="Molecule-disjoint validation (regimes.py): the molecules of each role are split into groups (five partitions, groups in "
      "FigS10_molecule_groups.csv); for each pair of a partner-1 group and a partner-2 group, the pairs combining "
      "them are held out and the training pairs are, by regime: all other pairs (both molecules measured in other pairs; a "
      "target is scored only if both occur in training); all pairs without a held partner-1 molecule (partner 1 new); the "
      "mirror image (partner 2 new); all pairs without a held molecule in either role (both new). The recommender, fitted on "
      "the training pairs only: the out-of-fold, cross-fitted preference (condition folds in FigS10_condition_folds.csv) "
      "predicted by an equal blend of a ten-neighbour ECFP4 Tanimoto model and a rank-8 bilinear ridge model on truncated-SVD "
      "fingerprints (random_state 0) and ten standardized descriptors; its trio = the three largest of training condition mean + "
      "prediction. Gain = best measured yield of that trio minus that of the complementary trio of the training pairs (greedy); "
      "scored when every condition of the panels is measured. Interval: 10,000 whole-array draws, default_rng(20260920), arrays "
      "in sorted identifier order; pooled mean = sum / count over the scored cases.",
    b="Arrays first require at least 40 pairs of two named molecules and 200 comparable pair-pairs (at least eight common "
      "conditions); pairs with constant raw yield profiles are then removed. Profile: the out-of-fold slope residual with a condition mean removed "
      "that is estimated on the other half of the pairs (halves in FigS10_pair_halves.csv; condition folds as in "
      "panel a); agreement = Pearson r of two pairs' profiles over common conditions; similarity = mean of the two partners' "
      "ECFP4 Tanimoto (radius 2, 2,048 bits). rho = Spearman correlation of similarity and agreement over all comparable "
      "pair-pairs; p = (hits + 1) / 5,000 over 4,999 permutations of whole structure labels (default_rng(20260920), a fresh "
      "generator per array), the eligibility held fixed; q = Benjamini-Hochberg over the 14 arrays.")
H, AX_X, AX_W, ROW_R, GROUP_X, VALUE_X, PITCH, PITCH_B, GAP_G, XLIM = 117.0, 66.0, 70.0, 64.0, 4.2, 139.0, 3.6, 3.3, 1.6, (-3.5, 4.5)
TOP, KEY_X = dict(a=18.5, b=60.0), 150.0
num = lambda v, d=2: f"{v:.{d}f}".replace("-", "−")


def fmt(m, lo, hi):
    return f"{'+' if m >= 0 else ''}{num(m)} ({num(lo)} to {num(hi)})" if lo < 0 else f"{'+' if m >= 0 else ''}{num(m)} ({num(lo)}–{num(hi)})"


def hidden_state(A, assoc_arrays):
    """The registry entries the fits read, as tables, and the check that each regime's training rows equal the reference
    definition (so the regime entries follow from the groups)."""
    groups, cfolds, halves, checked = [], [], [], 0
    for key, (pairs, conds, M) in A.items():
        released = [sp.split(" >> ") for sp in pairs]
        cp = [S.canonical(sp) for sp in pairs]
        ids_of = {}
        for k, seed in enumerate(S.PARTITION_SEEDS):
            for role, s_ in ((1, seed), (2, seed + 1)):               # regimes.py draws role 2 with seed + 1
                ids, mp = S.assignment([v[role - 1] for v in cp], s_)
                ids = np.asarray(ids, int)
                seen = {}
                for (rel_, can_), g in zip(((r[role - 1], c[role - 1]) for r, c in zip(released, cp)), ids):
                    assert seen.setdefault(can_, int(g)) == int(g) and mp[can_] == int(g)
                    if (k, role, rel_) not in ids_of:
                        ids_of[(k, role, rel_)] = int(g)
                        groups.append(dict(screen=key, partition=k + 1, role=role, molecule=rel_, group=int(g) + 1))
            id1 = np.array([ids_of[(k, 1, r[0])] for r in released]); id2 = np.array([ids_of[(k, 2, r[1])] for r in released])
            map1 = {c[0]: g for c, g in zip(cp, id1)}; map2 = {c[1]: g for c, g in zip(cp, id2)}
            for a in sorted(set(id1)):
                for b in sorted(set(id2)):
                    te = np.flatnonzero((id1 == a) & (id2 == b))
                    if not len(te):
                        continue
                    held1 = {v for v, z in map1.items() if z == a}; held2 = {v for v, z in map2.items() if z == b}
                    for regime, _ in REGIMES:
                        ref = RG._historical_training_rows(cp, held1, held2, set(te.tolist()), regime)
                        reg = RG.training_rows(cp, held1, held2, set(te.tolist()), regime)
                        assert np.array_equal(np.asarray(reg, int), ref), (key, seed, a, b, regime)
                        checked += 1
        folds = amplitude.condition_folds(len(conds))
        assert sorted(np.concatenate(folds).tolist()) == list(range(len(conds)))
        cfolds += [dict(screen=key, condition_id=conds[c], fold=j + 1) for j, f in enumerate(folds) for c in f]
        if key in assoc_arrays:
            (h0, h1), _ = amplitude.halves(len(pairs), screen=key)
            assert sorted(np.r_[h0, h1].tolist()) == list(range(len(pairs)))
            halves += [dict(screen=key, subpair=pairs[p], half=h + 1) for h, hh in enumerate((h0, h1)) for p in hh]
    G = pd.DataFrame(groups).sort_values(["screen", "partition", "role", "molecule"], kind="stable").reset_index(drop=True)
    CF = pd.DataFrame(cfolds).sort_values(["screen", "condition_id"], kind="stable").reset_index(drop=True)
    HV = pd.DataFrame(halves).sort_values(["screen", "subpair"], kind="stable").reset_index(drop=True)
    return G, CF, HV, checked


def released_state(A, assoc_arrays):
    """Public mode: the hidden state from the released files. Every molecule keeps one group per canonical structure, every
    condition and pair is covered, and each held-out cell's training rows follow from the groups by the reference definition
    (counted as in hidden_state)."""
    G, CF, HV = (pd.read_csv(RELEASED[k]) for k in ("groups", "condition_folds", "halves"))
    checked = 0
    for key, (pairs, conds, M) in A.items():
        released = [sp.split(" >> ") for sp in pairs]
        cp = [S.canonical(sp) for sp in pairs]
        g = {(int(r.partition), int(r.role), r.molecule): int(r.group) for r in G[G.screen == key].itertuples()}
        for k, _ in enumerate(S.PARTITION_SEEDS):
            for role in (1, 2):
                seen = {}
                for r, c in zip(released, cp):
                    assert seen.setdefault(c[role - 1], g[(k + 1, role, r[role - 1])]) == g[(k + 1, role, r[role - 1])]
            id1 = np.array([g[(k + 1, 1, r[0])] - 1 for r in released]); id2 = np.array([g[(k + 1, 2, r[1])] - 1 for r in released])
            map1 = {c[0]: z for c, z in zip(cp, id1)}; map2 = {c[1]: z for c, z in zip(cp, id2)}
            for a in sorted(set(id1)):
                for b in sorted(set(id2)):
                    te = np.flatnonzero((id1 == a) & (id2 == b))
                    if not len(te):
                        continue
                    held1 = {v for v, z in map1.items() if z == a}; held2 = {v for v, z in map2.items() if z == b}
                    for regime, _ in REGIMES:
                        RG._historical_training_rows(cp, held1, held2, set(te.tolist()), regime)
                        checked += 1
        assert sorted(CF[CF.screen == key].condition_id) == sorted(conds)
        if key in assoc_arrays:
            assert sorted(HV[HV.screen == key].subpair) == sorted(pairs) and set(HV[HV.screen == key].half) == {1, 2}
    assert set(HV.screen) == set(assoc_arrays) and set(CF.screen) == set(G.screen) == set(A)
    return G, CF, HV, checked


def compute_a(A):
    C = pd.read_csv(APPROVED["cases"])
    C = C[C.screen.isin(list(A)) & (C.target == "p_oof_centred")]
    assert C.screen.nunique() == 15
    pairs_of = {k: set(v[0]) for k, v in A.items()}
    assert all(sp in pairs_of[k] for k, sp in zip(C.screen, C.subpair))
    restated = json.loads(APPROVED["restated"].read_text(encoding="utf-8"))["restated_15"]["regimes"]["p_oof_centred"]
    rows, cases = [], []
    for regime, label in REGIMES:
        z = C[(C.regime == regime) & C.scorable]
        m, lo, hi = (100 * v for v in whole_array_interval(z, "gain_greedy"))
        n_pairs = int(z.groupby("screen").subpair.nunique().sum())
        r = restated[regime]
        assert (n_pairs, len(z)) == (r["n_unique_pairs"], r["n_records"])
        assert max(abs(m - r["greedy"][0]), abs(lo - r["greedy"][1]), abs(hi - r["greedy"][2])) < E.TOL
        rows.append(dict(regime=regime, label=label, pairs=n_pairs, cases=len(z), arrays=z.screen.nunique(), mean_pp=m, ci_lower_pp=lo,
                         ci_upper_pp=hi))
        cases.append(z.assign(label=label))
    return pd.DataFrame(rows), pd.concat(cases, ignore_index=True)


def compute_b(order):
    S0 = pd.read_csv(APPROVED["association"])
    S15 = S0[S0.screen.isin(order)].copy()
    assert len(S15) == 14 and "Hiyama|1" not in set(S15.screen)
    S15["q"] = bh(S15.p_preference_c.to_numpy())
    restated = json.loads(APPROVED["restated"].read_text(encoding="utf-8"))["restated_15"]["association"]["preference_c"]
    med, pos, sig = float(S15.rho_preference_c.median()), int((S15.rho_preference_c > 0).sum()), int((S15.q < 0.05).sum())
    assert abs(med - restated["median_rho"]) < E.TOL and (pos, sig) == (restated["n_positive"], restated["n_q05"]) == (14, 13)
    T = S15.set_index("screen").loc[[k for k in order if k in set(S15.screen)]].reset_index()
    return T[["screen", "n_profiles", "n_pairs_preference_c", "rho_preference_c", "p_preference_c", "q"]], med, pos, sig


def draw(ea, Tb, med, pos, sig, order, group):
    fig, cv = E.figure(H, "ed10")
    text_c, muted, faint = fs.C["text"], fs.C["muted"], fs.C["faint"]
    fs.title(fig, E.LEFT, 0.6, TITLE, H).set_gid("title|main")
    for L, y in (("a", 8.0), ("b", 45.0)):
        E.heading(fig, cv, 0, y, L, TITLES[L], H, NOTES[L])
    # a: four regimes
    hb = len(REGIMES) * PITCH + 0.4
    ax = fs.axes_mm(fig, AX_X, TOP["a"], AX_W, hb, H); ax.set_gid("a|axes")
    ax.set_xlim(*XLIM); ax.set_ylim(hb, 0)
    ax.plot([0, 0], [0, hb], color=faint, lw=0.5, zorder=0, gid="a|zero")
    for i, r in enumerate(ea.itertuples()):
        y = PITCH / 2 + i * PITCH
        ax.plot([r.ci_lower_pp, r.ci_upper_pp], [y, y], color=text_c, lw=0.8, solid_capstyle="butt", zorder=2, gid=f"a|ci|{r.regime}")
        ax.plot([r.mean_pp], [y], marker="o", ms=2.6, mew=0, mfc=text_c, ls="none", zorder=3, gid=f"a|point|{r.regime}")
        cv.text(GROUP_X, TOP["a"] + y, r.label, ha="left", va="center", fontsize=fs.PT["tiny"], gid=f"a|row|{r.regime}")
        cv.text(ROW_R, TOP["a"] + y, f"{r.pairs:,} pairs", ha="right", va="center", fontsize=fs.PT["tiny"], color=muted, gid=f"a|n|{r.regime}")
        cv.text(VALUE_X, TOP["a"] + y, fmt(r.mean_pp, r.ci_lower_pp, r.ci_upper_pp), ha="left", va="center", fontsize=fs.PT["tiny"],
                gid=f"a|value|{r.regime}")
    ax.set_xticks(range(-3, 5)); ax.set_yticks([])
    ax.set_xlabel("Best measured yield: structure-ranked trio minus complementary trio (percentage points)", labelpad=1.5)
    E.clean(ax, keep=("bottom",))
    # b: one row per qualifying array, in the order of Fig. 1F
    keys = list(Tb.screen)
    ys, y_ = {}, PITCH_B / 2
    for i, k in enumerate(keys):
        if i and group[k] != group[keys[i - 1]]:
            y_ += GAP_G
        ys[k] = y_; y_ += PITCH_B
    hb = y_ - PITCH_B / 2 + 0.4
    for j, (lab, colr) in enumerate((("q < 0.05", text_c), ("q ≥ 0.05", faint))):
        x = KEY_X + 16.0 * j
        cv.plot([x + 0.7], [45.0 + 5.8], marker="o", ms=2.6, mew=0, mfc=colr, ls="none", gid=f"b|key|{j}")
        cv.text(x + 2.0, 45.0 + 5.8, lab, ha="left", va="center", fontsize=fs.PT["tiny"], gid=f"b|key_label|{j}")
    cv.text(E.RIGHT, TOP["b"] - 3.2, f"{pos} of 14 > 0; {sig} with q < 0.05", ha="right", va="center", fontsize=fs.PT["tiny"], color=muted,
            gid="b|counts")
    bx = fs.axes_mm(fig, AX_X, TOP["b"], AX_W, hb, H); bx.set_gid("b|axes")
    bx.set_xlim(-0.02, 0.30); bx.set_ylim(hb, 0)
    bx.plot([0, 0], [0, hb], color=faint, lw=0.5, zorder=0, gid="b|zero")
    bx.plot([med, med], [0, hb], color=muted, lw=0.6, ls=(0, (2.0, 1.4)), zorder=1, gid="b|median")
    xm = AX_X + (med - bx.get_xlim()[0]) / (bx.get_xlim()[1] - bx.get_xlim()[0]) * AX_W
    cv.text(xm, TOP["b"] - 3.2, f"median {num(med, 3)}", ha="center", va="center", fontsize=fs.PT["tiny"], color=muted, gid="b|median_label")
    for r in Tb.itertuples():
        y = ys[r.screen]
        bx.plot([r.rho_preference_c], [y], marker="o", ms=2.6, mew=0, mfc=text_c if r.q < 0.05 else faint, ls="none", zorder=3,
                gid=f"b|point|{r.screen}")
        cv.text(ROW_R, TOP["b"] + y, fs.ARRAY_LABEL[r.screen], ha="right", va="center", fontsize=fs.PT["tiny"], gid=f"b|name|{r.screen}")
        cv.text(VALUE_X, TOP["b"] + y, num(r.rho_preference_c, 3), ha="left", va="center", fontsize=fs.PT["tiny"], gid=f"b|value|{r.screen}")
    bx.set_xticks([0, 0.1, 0.2, 0.3]); bx.set_yticks([])
    bx.set_xlabel("Spearman ρ: partner similarity against agreement of preference profiles", labelpad=1.5)
    E.clean(bx, keep=("bottom",))
    return fig


def source_data(ea, Ca, Tb, med, pos, sig):
    a = ea.rename(columns={"label": "molecules", "regime": "regime_code"})[["molecules", "regime_code", "pairs", "cases", "arrays", "mean_pp",
                                                                         "ci_lower_pp", "ci_upper_pp"]]
    part = {s: i + 1 for i, s in enumerate(S.PARTITION_SEEDS)}
    ca = Ca.assign(array=Ca.screen.map(fs.ARRAY_LABEL), partition=Ca.seed.map(part), gain_pp=100 * Ca.gain_greedy)[
        ["label", "screen", "array", "partition", "subpair", "gain_pp"]].rename(columns={"label": "molecules"})
    order_r = {lab: i for i, (_, lab) in enumerate(REGIMES)}
    ca = ca.sort_values(["molecules", "screen", "partition", "subpair"], key=lambda s: s.map(order_r) if s.name == "molecules" else s,
                        kind="stable").reset_index(drop=True)
    b = Tb.assign(array=Tb.screen.map(fs.ARRAY_LABEL)).rename(columns={"n_profiles": "pairs", "n_pairs_preference_c": "comparisons",
                                                                     "rho_preference_c": "spearman_rho", "p_preference_c": "permutation_p",
                                                                     "q": "bh_q"})[["screen", "array", "pairs", "comparisons", "spearman_rho",
                                                                                     "permutation_p", "bh_q"]]
    bs = pd.DataFrame([dict(statistic="median Spearman rho", value=med), dict(statistic="arrays with rho > 0", value=pos),
                       dict(statistic="arrays with q < 0.05", value=sig), dict(statistic="qualifying arrays", value=14)])
    readme = pd.DataFrame([
        dict(sheet="ED10a", panel="a", estimand="best measured yield of the recommender's trio minus that of the complementary trio",
             rows="one per regime", units="percentage points of yield; counts", transformation=TRANSFORM["a"]),
        dict(sheet="ED10a_cases", panel="a", estimand="as ED10a", rows="one per scored held-out pair, partition and regime",
             units="percentage points", transformation=TRANSFORM["a"]),
        dict(sheet="ED10b", panel="b", estimand="Spearman rho of partner similarity and preference-profile agreement",
             rows="one per qualifying array, in the order of Fig. 1F", units="correlation; probabilities", transformation=TRANSFORM["b"]),
        dict(sheet="ED10b_summary", panel="b", estimand="as ED10b", rows="the four counts quoted in the legend", units="correlation; counts",
             transformation="From ED10b."),
        dict(sheet="FigS10_molecule_groups.csv (separate file)", panel="a", estimand="-",
             rows="one per array, partition (1-5), partner (1, 2) and molecule (as written in the measurement file)",
             units="group numbers", transformation="The frozen registry entries assignment|<seed>|<hash> of the five partitions "
                                                   "(partner 2 drawn with the partition's seed + 1)."),
        dict(sheet="FigS10_condition_folds.csv (separate file)", panel="a, b", estimand="-",
             rows="one per array and condition", units="fold numbers", transformation="The frozen registry entries conditions|<m> "
                                                                                      "(m = conditions of the array)."),
        dict(sheet="FigS10_pair_halves.csv (separate file)", panel="b", estimand="-", rows="one per pair of the 14 arrays",
             units="half numbers", transformation="The frozen registry entries halves|<screen>|20260920, preserving the original objects after exclusions."),
        dict(sheet="(all)", panel="-", estimand="-", rows="-", units="-",
             transformation="Measurements: DATASET_analysis.csv.gz (A1). The fits from these files: 06_CODE/Checks/verify_ed10_refit.py. "
                            "Input hashes: FigS10_SourceData_provenance.json.")])
    return dict(README=readme, ED10a=a, ED10a_cases=ca, ED10b=b, ED10b_summary=bs)


def main(out=None):
    A1, _ = E.public_units()
    A = arrays(A1)
    order, group = E.order()
    ea, Ca = compute_a(A)
    Tb, med, pos, sig = compute_b(order)
    G, CF, HV, checked = (released_state if PUBLIC_MODE else hidden_state)(A, set(Tb.screen))
    fig = draw(ea, Tb, med, pos, sig, order, group)
    if PUBLIC_MODE:                                   # the released hidden state read in place of the registry
        inputs = dict(A1=dict(file=E.rel(E.RAW), sha256=E.sha_file(E.RAW), rows=int(len(A1))),
                      released={k: dict(file=E.rel(p), sha256=E.sha_file(p)) for k, p in RELEASED.items()},
                      frozen={k: dict(file=E.rel(p), sha256=E.sha_file(p)) for k, p in APPROVED.items()})
    else:
        inputs = dict(A1=dict(file=E.rel(E.RAW), sha256=E.sha_file(E.RAW), rows=int(len(A1))),
                      fold_registry=dict(file=E.rel(FOLDS_REG), sha256=E.sha_file(FOLDS_REG),
                                         entries_used="assignment|..., conditions|<m>, halves|<screen>|20260920, regime|..."),
                      registry_integrity=[dict(file=E.rel(p), sha256=E.sha_file(p)) for p in E.REGISTRY_INTEGRITY],
                      frozen={k: dict(file=E.rel(p), sha256=E.sha_file(p)) for k, p in APPROVED.items()})
    prov = dict(figure=f"Fig. S{N} | {TITLE}", generated_by=E.rel(__file__),
                inputs=inputs,
                randomness=dict(groups="frozen registry (released as FigS10_molecule_groups.csv)",
                                folds="frozen registry (released as FigS10_condition_folds.csv and FigS10_pair_halves.csv)",
                                regime_rows=f"equal to regimes._historical_training_rows in all {checked} held-out cells and regimes",
                                a="common.whole_array_interval: default_rng(20260920), 10,000 draws, a fresh generator per regime",
                                b="structure.permutation_p: default_rng(20260920), 4,999 permutations, a fresh generator per array",
                                fits="sklearn TruncatedSVD random_state 0"),
                transformation=TRANSFORM, refit_verification="08_REVIEW_AND_QC/extended_data/ED10_REFIT_VERIFICATION.json",
                estimates=dict(a=ea.to_dict("records"), b=dict(median=med, positive=pos, q05=sig)))
    p = E.write(N, fig, H, source_data(ea, Ca, Tb, med, pos, sig), prov, out,
                extra=(("groups", "FigS10_molecule_groups.csv"), ("condition_folds", "FigS10_condition_folds.csv"),
                       ("halves", "FigS10_pair_halves.csv")),
                tables=(("groups", G), ("condition_folds", CF), ("halves", HV)))
    msg = (f"Fig. S{N}: " + "; ".join(f"{r.regime} {fmt(r.mean_pp, r.ci_lower_pp, r.ci_upper_pp)} ({r.pairs})" for r in ea.itertuples())
           + f" | median rho {med:.3f}, {pos}/14 > 0, {sig} q<0.05; regime cells checked {checked}; wrote {len(p)} files")
    print(msg.replace("−", "-").replace("–", "-"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    main(ap.parse_args().out)
