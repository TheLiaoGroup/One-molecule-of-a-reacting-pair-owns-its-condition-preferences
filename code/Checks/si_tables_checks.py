"""The per-array tables and single values of the Supplementary Information that no figure generator prints, recomputed from the
released measurements and from released result files that the rebuild verifies (Bruce, 2026-10-13: every number of the text is to
be recomputed by the rebuild).

Inputs and who verifies them in the rebuild: history_arrays.csv and history_cv.csv (Fig. S8's generator recomputes
them from the measurements), fig3_regime_cases.csv and structure_association.csv (the Fig. S10 re-fit recomputes every
case and every array), phase1_D.csv (Figure 4 and Fig. S3), decision_targets.csv (verify_release check 6 and the
package's claims step), and the measurements themselves.
1 table S5: per-array partner agreement, held-out R2 and history panel minus the two fixed trios (per-pair means).
2 table S7: per-array structure-based gains and records by exposure regime, both-new pairs, and the association.
3 Supplementary Note 15: the median association and its median without shared molecules.
4 Supplementary Note 10: E3 per array, and its intervals resampling the first or the second building block within arrays
  (10,000 draws, seed 20260928, the molecule resampling of the matched control).
5 Supplementary Note 7: the two strata weighted equally, with the interval drawn as the frozen analysis drew it (seed 20261004, after
  the two earlier intervals of that stream).
6 Counts of Methods and Supplementary Note 3: measured zeros, grid cells outside their pair's design, wells of the source file
  (the cells without a yield are the released wells without a yield of the fifteen arrays, --mask).
7 Results: the share of a product-forming pair's yield variance across conditions that its condition pattern keeps, pooled
  within each array (sum of squared pattern over sum of squared deviations of each pair's yields from its own mean).
8 table S18: per array, the roles and their distinct molecules, pairs, plates, conditions, distinct reagents and
  solvents, temperature and time, measurements, mean yield, exactly-zero fraction and product-forming fraction, from the
  released A1 records (--a1) and condition table (--conditions), in the order of Fig. 3A.
Writes 05_DATA/Results/directional/si_tables_checks.json, or the file named by --out.
    python 06_CODE/Checks/si_tables_checks.py --a1 CSV --conditions CSV --mask CSV [--out JSON]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "06_CODE/Figures"), str(ROOT / "06_CODE/Core/00_common"), str(ROOT / "06_CODE/Core/00_common/ctb")]
import figure1_data as F1                                                   # noqa: E402
from anatomy import FORMS, pattern                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "05_DATA/Results/directional/si_tables_checks.json"))
ap.add_argument("--a1", required=True)
ap.add_argument("--conditions", required=True)
ap.add_argument("--mask", required=True,
                help="the released wells without a yield, with their status")
ARGS = ap.parse_args()
RES = ROOT / "05_DATA/Results"
REGIMES = ["warm/warm", "new/known", "known/new", "new/new"]      # both recur, partner 1 new, partner 2 new, both new
NB = 10000


def bb_interval(d, unit, seed=20260928):
    rng = np.random.default_rng(seed)
    sums, counts = np.zeros(NB), np.zeros(NB)
    for _, g in d.groupby("screen"):
        a = g.groupby(unit)["difference_pp"].agg(["sum", "count"])
        ix = rng.integers(len(a), size=(NB, len(a)))
        sums += a["sum"].to_numpy()[ix].sum(1)
        counts += a["count"].to_numpy()[ix].sum(1)
    return [float(v) for v in np.quantile(sums / counts, [.025, .975])]


def whole_array(d, seed=20260928):
    """validation_core.summarize_effects: whole arrays resampled, 10,000 draws; the equal-array mean and leave-one-array-out means."""
    g = d.groupby("screen")["difference_pp"].agg(["sum", "count", "mean"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(g), size=(NB, len(g)))
    boot = g["sum"].to_numpy()[idx].sum(axis=1) / g["count"].to_numpy()[idx].sum(axis=1)
    return dict(ci=[float(v) for v in np.quantile(boot, [.025, .975])], equal_array_mean_pp=float(g["mean"].mean()),
                leave_one_array_out={k: float(d[d.screen != k].difference_pp.mean()) for k in g.index})


def main():
    res = {}
    fifteen = set(pd.read_csv(RES / "display_inputs/Anatomy_arrays.csv").screen)   # the analysed arrays; process arrays are not released
    assert len(fifteen) == 15
    # 1 table S5
    H = pd.read_csv(RES / "primary/history_arrays.csv").set_index("screen")
    cv = pd.read_csv(RES / "primary/history_cv.csv")
    cv = cv[cv.screen.isin(fifteen)]
    t5 = []
    assert set(cv.screen) == fifteen and set(H.index) >= fifteen
    for scr, g in cv.groupby("screen"):
        per = g.groupby("pair")[["H", "B", "C"]].mean()
        h = H.loc[scr]
        t5.append(dict(array=scr, r_share1=float(h.r_share1), r_share2=float(h.r_share2), r_neither=float(h.r_unrelated),
                       r2_p1=float(h.r2_p1), r2_p2=float(h.r2_p2), r2_both=float(h.r2_both),
                       r2_p1_slope_removed=float(h.r2_p1_slope_removed), r2_p2_slope_removed=float(h.r2_p2_slope_removed),
                       minus_mean_ranked_pp=float(100 * (per.H - per.B).mean()), minus_complementary_pp=float(100 * (per.H - per.C).mean()),
                       targets=int(len(per))))
    res["supplementary_table_5"] = t5
    # 2 table S7 and 3 Note 15
    rc = pd.read_csv(RES / "primary/fig3_regime_cases.csv")
    sc = rc[rc.scorable.astype(bool) & rc.screen.isin(fifteen)]
    sa = pd.read_csv(RES / "primary/structure_association.csv")
    sa = sa[sa.screen.isin(fifteen)].set_index("screen")
    t7 = []
    for scr, g in sc.groupby("screen"):
        row = dict(array=scr, both_new_pairs=int(g[g.regime == "new/new"].subpair.nunique()))
        for r in REGIMES:
            x = g[g.regime == r]
            row[f"gain_{r}"] = float(100 * x.gain_greedy.mean()); row[f"records_{r}"] = int(len(x))
        if scr in sa.index:
            row.update(rho=float(sa.loc[scr, "rho_preference_c"]), rho_no_shared=float(sa.loc[scr, "rho_no_shared_preference_c"]),
                       p=float(sa.loc[scr, "p_preference_c"]))
        t7.append(row)
    res["supplementary_table_7"] = t7
    res["note_15"] = dict(median_association=float(sa.rho_preference_c.median()),
                          median_without_shared_molecules=float(sa.rho_no_shared_preference_c.median()),
                          positive=int((sa.rho_preference_c > 0).sum()), positive_without_shared=int((sa.rho_no_shared_preference_c > 0).sum()),
                          arrays=int(len(sa)))
    # 4 Supplementary Note 10, E3
    d = pd.read_csv(RES / "reserve_validation/decision_targets.csv")
    d = d[d.screen.isin(fifteen)]
    pf = d[d.product_forming.astype(str).str.lower() == "true"]
    res["note_10_e3"] = dict(
        per_array={s: dict(mean_pp=float(g.difference_pp.mean()), pairs=int(len(g))) for s, g in d.groupby("screen")},
        all_pairs=dict(mean_pp=float(d.difference_pp.mean()), pairs=int(len(d)), second_unit=bb_interval(d, "building_block_b"),
                       first_unit=bb_interval(d, "building_block_a"), whole_array=whole_array(d)),
        product_forming=dict(mean_pp=float(pf.difference_pp.mean()), pairs=int(len(pf)), second_unit=bb_interval(pf, "building_block_b"),
                             first_unit=bb_interval(pf, "building_block_a"), whole_array=whole_array(pf)))
    # 5 Supplementary Note 7, the two strata weighted equally
    D = pd.read_csv(RES / "directional/phase1_D.csv").sort_values("molecule")      # the order in which the frozen analysis drew them
    rng = np.random.default_rng(20261004)
    x = D.D.to_numpy()
    _ = x[rng.integers(0, len(x), (NB, len(x)))]                                 # the interval of the mean over 23 molecules
    am, ac = D[D.stratum == "amine"].D.to_numpy(), D[D.stratum == "acid"].D.to_numpy()
    _ = am[rng.integers(0, len(am), (NB, len(am)))]                              # the interval of the amines
    bal = 0.5 * am[rng.integers(0, len(am), (NB, len(am)))].mean(1) + 0.5 * ac[rng.integers(0, len(ac), (NB, len(ac)))].mean(1)
    res["note_7_equal_strata"] = dict(mean=float(0.5 * am.mean() + 0.5 * ac.mean()), ci=[float(v) for v in np.percentile(bal, [2.5, 97.5])])
    # 6 counts
    A1, A3 = F1.public_units()
    grid = int(sum(g.subpair.nunique() * g.condition_id.nunique() for _, g in A1.groupby("screen")))
    blank = int((pd.read_csv(ARGS.mask).unit == "A1").sum())                  # the wells without a yield in the fifteen arrays
    res["counts"] = dict(measurements=int(len(A1)), exact_zeros=int((A1["yield"] == 0).sum()), grid_cells=grid, without_yield=blank,
                         grid_cells_not_in_design=grid - int(len(A1)) - blank, hiyama_duplicate=int(len(A3)),
                         source_wells=int(len(A1) + len(A3) + blank))
    # 7 the condition-pattern share of each pair's variance across conditions
    kept = {}
    for scr, g in A1.groupby("screen"):
        M = F1.matrix(g)[0].to_numpy(float)
        M = M[np.nanmax(M, 1) >= FORMS]
        kept[scr] = float(np.nansum(pattern(M)[0] ** 2) / np.nansum((M - np.nanmean(M, 1, keepdims=True)) ** 2))
    v = list(kept.values())
    res["pattern_variance_kept"] = dict(per_array=kept, median=float(np.median(v)), range=[float(min(v)), float(max(v))])
    # 8 table S18
    import fig_style as fs
    import anatomy
    from history import ROLE
    R1 = pd.read_csv(ARGS.a1, dtype={"condition_id": str})
    K = pd.read_csv(ARGS.conditions, dtype=str)
    hal, other = fs.array_order({k: anatomy.electrophile_role(k) is not None for k in fs.ARRAY_LABEL})
    summ = []
    for scr in hal + other:
        g = R1[R1.screen == scr]
        c = K[(K.rxn_class == g.rxn_class.iloc[0]) & K.condition_id.isin(set(g.condition_id))]
        assert len(c) == g.condition_id.nunique(), scr
        reag = set().union(*(set(c[f"Reagent{i}_Standardized Name"].dropna()) for i in range(1, 7)))
        tc, th = pd.to_numeric(c.temp_id), pd.to_numeric(c.time_id)
        assert tc.nunique() == 1 and th.nunique() == 1, scr
        best = g.groupby("subpair")["yield"].max()
        summ.append(dict(array=fs.ARRAY_LABEL[scr], array_text=fs.ARRAY_LABEL[scr], screen=scr, role_1=ROLE[scr][0][0],
                         molecules_1=int(g.sub_1_smiles.nunique()), role_2=ROLE[scr][1][0], molecules_2=int(g.sub_2_smiles.nunique()),
                         pairs=int(g.subpair.nunique()), plates=int(g.plate.nunique()), conditions=int(g.condition_id.nunique()),
                         reagents=len(reag), solvents=int(c["Standardized Name"].nunique()), temperature_c=f"{tc.iloc[0]:.0f}",
                         time_h=f"{th.iloc[0]:.0f}", measurements=int(len(g)), mean_yield=float(g["yield"].mean()),
                         zero_fraction=float((g["yield"] == 0).mean()), product_forming_fraction=float((best >= FORMS).mean())))
    assert len(summ) == 15 and sum(a["measurements"] for a in summ) == len(R1) == len(A1)
    assert sum(a["pairs"] for a in summ) == len(R1.drop_duplicates(["screen", "subpair"])) == len(A1.drop_duplicates(["screen", "subpair"]))
    res["array_summary"] = summ
    Path(ARGS.out).write_bytes((json.dumps(res, indent=1, default=float) + "\n").encode("utf-8"))
    print(json.dumps({k: v for k, v in res.items() if k not in ("supplementary_table_5", "supplementary_table_7")}, indent=1, default=float))


if __name__ == "__main__":
    main()
