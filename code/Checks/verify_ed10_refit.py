"""Re-fit of Fig. S10 from the public measurements and the files released with it.

Re-runs, on A1, the structure-based recommender under molecule-disjoint validation (the loop of regimes.py; the amplitude
R2 that the loop also computes is not reported in Fig. S10 and is skipped) and the association of structural
similarity with preference-profile agreement (structure.run_association). The molecule groups, the condition folds and the
pair halves are read from FigS10_molecule_groups.csv, FigS10_condition_folds.csv and
FigS10_pair_halves.csv; the training rows of each regime come from the reference definition
regimes._historical_training_rows. The fold registry is blocked: any read of it raises. Every case is compared with the
frozen fig3_regime_cases.csv and every array's association with the frozen structure_association.csv; the reported
summaries are then recomputed from the re-fit and compared with the Source Data of Fig. S10.
Fits run single-threaded (threadpoolctl), as in the frozen run; arrays run in parallel processes.
Writes 08_REVIEW_AND_QC/extended_data/ED10_REFIT_VERIFICATION.json.
    python 06_CODE/Checks/verify_ed10_refit.py [--jobs N] [--only KEY ...] [--released DIR] [--record PATH]
(--released and --only are for trial runs; the record the audit reads comes from a full run on the delivered files.)
"""
import argparse
import datetime
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "06_CODE/Figures"), str(ROOT / "06_CODE/Core/00_common"), str(ROOT / "06_CODE/Core/00_common/ctb")]
import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402

SD = ROOT / "04_SOURCE_DATA/Supplementary_Figures"
RELEASED = dict(groups=SD / "FigS10_molecule_groups.csv", condition_folds=SD / "FigS10_condition_folds.csv",
                halves=SD / "FigS10_pair_halves.csv")
SOURCE_DATA = SD / "FigS10_SourceData.xlsx"
FROZEN = dict(cases=ROOT / "05_DATA/Results/primary/fig3_regime_cases.csv",
              association=ROOT / "05_DATA/Results/primary/structure_association.csv")
RECORD = ROOT / "08_REVIEW_AND_QC/extended_data/ED10_REFIT_VERIFICATION.json"
CODE = [ROOT / "06_CODE/Core/00_common/ctb" / f for f in ("regimes.py", "structure.py", "preference.py", "amplitude.py", "common.py")]
TOL_YIELD, TOL_STAT = 1e-12, 1e-9
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rel = lambda p: Path(p).resolve().relative_to(ROOT).as_posix() if Path(p).resolve().is_relative_to(ROOT) else Path(p).name
STATE = {}


def install():
    """In each process: block the registry, take the regime rows from the reference definition, and serve the condition folds
    and pair halves of the array being fitted from the released files."""
    import boundaries, common, amplitude, structure, preference, regimes

    def blocked(*a, **k):
        raise RuntimeError("the fold registry was read")
    for mod in (boundaries, common, amplitude, structure):
        for name in ("frozen_folds", "frozen_assignment"):
            if hasattr(mod, name):
                setattr(mod, name, blocked)

    def condition_folds(m):
        assert m == len(STATE["conds"])
        return [f.copy() for f in STATE["folds"]]

    def halves(n, seed=20260920, *, screen):
        assert n == len(STATE["pairs"]) and seed == 20260920, "pair halves are released for the 14 arrays of panel b only"
        h0, h1 = STATE["halves"]
        return [(h0, h1), (h1, h0)]
    preference.condition_folds, preference.halves = condition_folds, halves
    regimes.training_rows = regimes._historical_training_rows


def regime_cases(key, pairs, M, groups):
    import preference as PREF, regimes as RG, structure as S
    from common import top_k, greedy_trio
    F1, F2, D = S.featurise(pairs)
    cp = [S.canonical(s) for s in pairs]
    released = [sp.split(" >> ") for sp in pairs]
    out = []
    for k, seed in enumerate(S.PARTITION_SEEDS):
        id1 = np.array([groups[(k + 1, 1, r[0])] - 1 for r in released])
        id2 = np.array([groups[(k + 1, 2, r[1])] - 1 for r in released])
        map1 = dict(zip([v[0] for v in cp], id1)); map2 = dict(zip([v[1] for v in cp], id2))
        for a in sorted(set(id1)):
            for b in sorted(set(id2)):
                te = np.flatnonzero((id1 == a) & (id2 == b))
                if not len(te):
                    continue
                held1 = {v for v, z in map1.items() if z == a}
                held2 = {v for v, z in map2.items() if z == b}
                for regime in RG.REGIMES:
                    tr = RG.training_rows(cp, held1, held2, set(te.tolist()), regime)
                    if len(tr) < 10:
                        continue
                    seen1 = {cp[i][0] for i in tr}; seen2 = {cp[i][1] for i in tr}
                    ok_te = [i for i in te
                             if ((cp[i][0] in seen1 and cp[i][1] in seen2) if regime == "warm/warm" else
                                 (cp[i][0] not in seen1 and cp[i][1] in seen2) if regime == "new/known" else
                                 (cp[i][0] in seen1 and cp[i][1] not in seen2) if regime == "known/new" else
                                 (cp[i][0] not in seen1 and cp[i][1] not in seen2))]
                    if not ok_te:
                        continue
                    ok_te = np.array(ok_te, int)
                    sp = PREF.split_decomposition(M, tr, ok_te)
                    T = np.full_like(M, np.nan); T[tr] = sp["train_p"]; T[ok_te] = sp["test_p"]
                    colm, gm, pp = S.fit_predict(M, F1, F2, D, tr, ok_te, target=T)
                    gp = top_k(colm); cpan = greedy_trio(M[tr])
                    for j, i in enumerate(ok_te):
                        sc = S.score(M[i], top_k(colm + pp[j]), gp, cpan)
                        out.append(dict(screen=key, regime=regime, target="p_oof_centred", seed=seed, subpair=pairs[i], n_train=len(tr), **sc))
    return out


def work(task):
    from threadpoolctl import threadpool_limits
    import structure as S
    key, pairs, conds, M, groups, folds, halves = task
    t0 = time.time()
    STATE.update(conds=conds, pairs=pairs, folds=folds, halves=halves)
    with threadpool_limits(limits=1):
        cases = regime_cases(key, pairs, M, groups)
        t1 = time.time()
        assoc = None
        if halves is not None:
            d, _ = S.run_association({key: (pairs, conds, M)})
            assert len(d) == 1
            assoc = d.iloc[0].to_dict()
    return key, cases, assoc, dict(regimes_s=round(t1 - t0, 1), association_s=round(time.time() - t1, 1))


def main(jobs, only, record, released_dir=None):
    import ed_common as E
    import ed10
    from common import arrays, bh, whole_array_interval
    t_start = time.time()
    if released_dir:
        global SOURCE_DATA
        for k in RELEASED:
            RELEASED[k] = Path(released_dir) / RELEASED[k].name
        SOURCE_DATA = Path(released_dir) / SOURCE_DATA.name
    A1, _ = E.public_units()
    A = arrays(A1)
    G = pd.read_csv(RELEASED["groups"]); CF = pd.read_csv(RELEASED["condition_folds"]); HV = pd.read_csv(RELEASED["halves"])
    tasks = []
    for key, (pairs, conds, M) in A.items():
        if only and key not in only:
            continue
        g = {(int(r.partition), int(r.role), r.molecule): int(r.group) for r in G[G.screen == key].itertuples()}
        fold_of = CF[CF.screen == key].set_index("condition_id").fold.reindex(conds).to_numpy()
        assert np.isfinite(fold_of).all()
        folds = [np.flatnonzero(fold_of == j) for j in range(1, int(fold_of.max()) + 1)]
        h = HV[HV.screen == key]
        halves = None
        if len(h):
            half_of = h.set_index("subpair").half.reindex(pairs).to_numpy()
            assert np.isfinite(half_of).all()
            halves = (np.flatnonzero(half_of == 1), np.flatnonzero(half_of == 2))
        tasks.append((key, list(pairs), list(conds), M, g, folds, halves))
    with ProcessPoolExecutor(max_workers=jobs, initializer=install) as ex:
        results = list(ex.map(work, tasks))
    timings = {k: t for k, _, _, t in results}
    cases = pd.DataFrame([c for _, cs, _, _ in results for c in cs])
    assoc = pd.DataFrame([a for _, _, a, _ in results if a is not None])
    keys = [t[0] for t in tasks]

    # every case against the frozen run
    C0 = pd.read_csv(FROZEN["cases"])
    C0 = C0[C0.screen.isin(keys)]
    on = ["screen", "regime", "target", "seed", "subpair"]
    J = cases.merge(C0, on=on, how="outer", suffixes=("", "_f"), indicator=True)
    cases_equal = bool((J._merge == "both").all()) and len(J) == len(C0) == len(cases)
    flags = ["n_train", "scorable", "rescue_greedy", "loss_greedy", "rescue_mean", "loss_mean"]
    sc = J[J.scorable.fillna(False).astype(bool)]
    flag_mismatch = int(sum(((J[c].fillna(-1).astype(float) != J[c + "_f"].fillna(-1).astype(float))).sum() for c in flags))
    yield_cols = ["model_best", "mean_best", "greedy_best", "gain_mean", "gain_greedy"]
    dev_yield = max(float((sc[c] - sc[c + "_f"]).abs().max()) for c in yield_cols) if len(sc) else float("nan")

    # every array's association against the frozen run
    S0 = pd.read_csv(FROZEN["association"]).set_index("screen")
    cols = [c for c in assoc.columns if c != "screen" and not c.startswith("q")]
    Sa = assoc.set_index("screen")
    dev_assoc = max(float(np.abs(Sa[c].to_numpy(float) - S0.loc[Sa.index, c].to_numpy(float)).max()) for c in cols)
    p_equal = all(bool((Sa[c] == S0.loc[Sa.index, c]).all()) for c in ("p", "p_preference", "p_preference_c"))

    # the reported summaries from the re-fit against the Source Data
    full = not only
    sd = {}
    if full:
        X = pd.read_excel(SOURCE_DATA, sheet_name=None)
        a = X["ED10a"].set_index("regime_code")
        dev_a = 0.0
        for regime, label in ed10.REGIMES:
            z = cases[(cases.regime == regime) & cases.scorable.astype(bool)]
            m, lo, hi = (100 * v for v in whole_array_interval(z, "gain_greedy"))
            n_pairs = int(z.groupby("screen").subpair.nunique().sum())
            r = a.loc[regime]
            assert (n_pairs, len(z)) == (int(r.pairs), int(r.cases)) and r.molecules == label
            dev_a = max(dev_a, abs(m - r.mean_pp), abs(lo - r.ci_lower_pp), abs(hi - r.ci_upper_pp))
        lab = dict(ed10.REGIMES)
        mine = cases[cases.scorable.astype(bool)].assign(molecules=lambda d: d.regime.map(lab),
                                                         partition=lambda d: d.seed.map({s: i + 1 for i, s in enumerate(ed10.S.PARTITION_SEEDS)}))
        K = mine.merge(X["ED10a_cases"], on=["molecules", "screen", "partition", "subpair"], how="outer", indicator=True)
        cases_sd_equal = bool((K._merge == "both").all()) and len(K) == len(X["ED10a_cases"])
        dev_cases_sd = float((100 * K.gain_greedy - K.gain_pp).abs().max())
        Sa = Sa.assign(q=bh(Sa.p_preference_c.to_numpy()))
        b = X["ED10b"].set_index("screen")
        assert set(b.index) == set(Sa.index) and len(b) == 14
        dev_b = max(float(np.abs(Sa.loc[b.index, x].to_numpy(float) - b[y].to_numpy(float)).max())
                    for x, y in (("rho_preference_c", "spearman_rho"), ("p_preference_c", "permutation_p"), ("q", "bh_q"),
                                 ("n_pairs_preference_c", "comparisons"), ("n_profiles", "pairs")))
        s = X["ED10b_summary"].set_index("statistic").value
        dev_bs = max(abs(float(Sa.rho_preference_c.median()) - s["median Spearman rho"]),
                     abs(int((Sa.rho_preference_c > 0).sum()) - s["arrays with rho > 0"]), abs(int((Sa.q < 0.05).sum()) - s["arrays with q < 0.05"]))
        sd = dict(ED10a_max_abs_dev_pp=dev_a, ED10a_cases_rows_equal=cases_sd_equal, ED10a_cases_max_abs_dev_pp=dev_cases_sd,
                  ED10b_max_abs_dev=dev_b, ED10b_summary_max_abs_dev=float(dev_bs))
    ok = (cases_equal and flag_mismatch == 0 and dev_yield <= TOL_YIELD and dev_assoc <= TOL_STAT and p_equal
          and (not full or (sd["ED10a_max_abs_dev_pp"] < 1e-7 and sd["ED10a_cases_rows_equal"] and sd["ED10a_cases_max_abs_dev_pp"] < 1e-7
                            and sd["ED10b_max_abs_dev"] < 1e-7 and sd["ED10b_summary_max_abs_dev"] < 1e-7)))
    import scipy, sklearn, rdkit
    out = dict(
        script=rel(__file__), script_sha256=sha(__file__), completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        scope="all 15 arrays" if full else f"partial: {sorted(keys)}", runtime_s=round(time.time() - t_start, 1), per_array_s=timings,
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                         sklearn=sklearn.__version__, rdkit=rdkit.__version__, threads_per_fit=1, processes=jobs),
        inputs=dict(A1=dict(file=rel(E.RAW), sha256=sha(E.RAW), rows=int(len(A1))),
                    released={k: dict(file=rel(p), sha256=sha(p)) for k, p in RELEASED.items()},
                    source_data=dict(file=rel(SOURCE_DATA), sha256=sha(SOURCE_DATA)),
                    frozen={k: dict(file=rel(p), sha256=sha(p)) for k, p in FROZEN.items()},
                    code={rel(p): sha(p) for p in CODE}),
        fold_registry="blocked: any read raises",
        regimes=dict(cases=int(len(cases)), scored=int(cases.scorable.astype(bool).sum()), arrays=len(keys), rows_equal=cases_equal,
                     flag_mismatches=flag_mismatch, max_abs_dev_yield=dev_yield, tolerance=TOL_YIELD),
        association=dict(arrays=int(len(assoc)), max_abs_dev=dev_assoc, permutation_p_equal=p_equal, tolerance=TOL_STAT),
        source_data=sd, result="PASS" if ok else "FAIL")
    Path(record).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("scope", "runtime_s", "regimes", "association", "source_data", "result")}, indent=1, default=float))
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--record", default=str(RECORD))
    ap.add_argument("--released")
    ar = ap.parse_args()
    sys.exit(0 if main(ar.jobs, set(ar.only or ()), ar.record, ar.released) else 1)
