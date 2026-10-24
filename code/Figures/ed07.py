"""Fig. S7 | A record grows with its partners.

a  The full history-depth record set, both roles pooled: SST and SSE_k of the 1,247 target-role records recomputed from A1
   and the two frozen display files (Fig4a_folds.csv, Fig4a_history_draws.csv; reused, not copied), and the interval of
   R2(6) - R2(1). That interval resamples the 150 array-role-molecule groups within arrays with the generator of the frozen
   anatomy run (seed 20261003), which first drew the depth subsets and then 2,000 resamples for the share interval (with
   redraws); the run is replayed (anatomy.depth_records with the frozen folds, anatomy.molecule_weights) and its 2,000
   group resamples are written out (FigS7_depth_interval_draws.csv).
The equal-depth transport panel is Figure 4B (06_CODE/Figures/figure4.py).
Checked against anatomy_depth.csv, anatomy.json and Figure4a_history_depth.csv.
Inputs with a measurement: A1; the released Fig4a_folds.csv and Fig4a_history_draws.csv; for the replay of the frozen stream,
the fifteen history|<array>|cv|0 entries of the fold registry. With P2_RELEASE_MODE=public (the rebuild of the public data
package) there is no replay: the records come from the frozen display files (figure4_data.released_records), the 2,000 group
resamples from the released FigS7_depth_interval_draws.csv and the share interval's redraw count from anatomy.json.
    python 06_CODE/Figures/ed07.py [--out DIR]
"""
import argparse
import json

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import ed_common as E
from ed_common import fs
import anatomy
import history
from figure4_data import PUBLIC as PUBLIC_MODE, released_records

N = 7
FOLDS_REG = E.DATA / "Analysis/06_METADATA/frozen_folds.json"
DISPLAY = E.ROOT / "05_DATA/Analysis/figure_display"
PUBLIC = dict(folds=DISPLAY / "Fig4a_folds.csv", draws=DISPLAY / "Fig4a_history_draws.csv")
INTERVAL_DRAWS = E.SDDIR / "FigS7_depth_interval_draws.csv"       # read in public mode only
APPROVED = dict(depth=E.DATA / "Results/primary/anatomy_depth.csv", anatomy=E.DATA / "Results/primary/anatomy.json",
                curve=E.DATA / "Results/display_inputs/Figure4a_history_depth.csv")
KS, NBOOT, SHRINK = (1, 2, 3, 4, 6), 2000, 4.0
TITLE = "A record grows with its partners"
NOTE = "Full history-depth record set; common-target curves in Fig. 4C"
TRANSFORM = dict(
    a="The full set of 1,247 target-role records (both roles; common-target curves in Fig. 4C): additive residual of the held-out pair against the profile of k "
      "randomly drawn training pairings of its molecule, five draws averaged (Fig4a_folds.csv, Fig4a_history_draws.csv). Explained "
      "= 1 - sum SSE_k / sum SST. Interval of R2(6) - R2(1): 2,000 resamples of the 150 array-role-molecule groups within arrays "
      "(exact resamples in FigS7_depth_interval_draws.csv); value of a resample = sum over its groups of (SSE_1 - SSE_6) "
      "/ sum of SST; 2.5th and 97.5th percentiles (linear interpolation).",
)
H = 62.0


def records_from_public(A1):
    """SST and SSE_k of every record from A1 and the two frozen display files alone."""
    FO, DR = pd.read_csv(PUBLIC["folds"]), pd.read_csv(PUBLIC["draws"])
    fold = {(r.screen, r.subpair): int(r.fold) for r in FO.itertuples()}
    mats = {}
    for key, g in A1.groupby("screen"):
        P = g.pivot(index="subpair", columns="condition_id", values="yield")
        mats[key] = (P.to_numpy(float), {sp: i for i, sp in enumerate(P.index)}, list(P.index))
    cache, rec = {}, {}
    for r in DR.itertuples():
        Y, pos, pairs = mats[r.screen]
        f = fold[(r.screen, r.target_pair)]
        if (r.screen, f) not in cache:
            tr = np.array([fold[(r.screen, sp)] != f for sp in pairs])
            cache[(r.screen, f)] = Y - np.nanmean(Y, 1)[:, None] - np.nanmean(Y[tr], 0)[None, :] + np.nanmean(Y[tr])
        R = cache[(r.screen, f)]
        p = pos[r.target_pair]; ok = np.isfinite(R[p]); y = R[p, ok]
        mol = r.target_pair.split(" >> ")[r.role - 1]
        sse = 0.0
        for d in range(1, 6):
            rows = [pos[f"{mol} >> {o}" if r.role == 1 else f"{o} >> {mol}"] for o in getattr(r, f"draw_{d}").split(";")]
            Z = R[rows][:, ok]
            sse += np.sum((y - np.nansum(Z, 0) / (np.isfinite(Z).sum(0) + SHRINK)) ** 2) / 5
        rec.setdefault((r.screen, r.role, r.target_pair), dict(screen=r.screen, role=r.role, target_pair=r.target_pair, molecule=mol,
                                                                 sst=float(np.sum(y ** 2))))[f"sse_{r.k}"] = float(sse)
    return pd.DataFrame(rec.values())


def replay_interval(A1):
    """The frozen anatomy run's generator: depth subsets, then the share interval (2,000 resamples with redraws), then the
    2,000 group resamples of the depth interval, which are returned."""
    AR = history.partner_arrays(A1)
    reg = json.loads(FOLDS_REG.read_text(encoding="utf-8"))
    folds_of = {k: [np.asarray(x, int) for x in reg[f"history|{k}|cv|0"]] for k in AR}
    del reg
    rng = np.random.default_rng(anatomy.SEED)
    depth, share_objs = [], {}
    for key, A in AR.items():
        act = np.nanmax(A["M"], 1) >= anatomy.FORMS
        share_objs[key] = (anatomy.Shares(anatomy.pattern(A["M"][act])[0], A["a"][act], A["b"][act]), A["a"][act], A["b"][act])
        depth += anatomy.depth_records(A, folds_of[key], act, rng)
    keys, boot, redraws = list(share_objs), [], 0
    for _ in range(NBOOT):
        vals = []
        for key in rng.choice(keys, len(keys)):
            sh, a, b = share_objs[key]
            for _try in range(50):
                r = sh.compute(anatomy.molecule_weights(a, b, rng))
                if np.isfinite(r["1"]) and np.isfinite(r["2"]):
                    break
                redraws += 1
            vals.append(r["1"] + r["2"])
        boot.append(np.median(vals))
    D = pd.DataFrame(depth)
    gk = D.groupby(["screen", "role", "molecule"])
    grp = gk.ngroup().to_numpy(); ng = int(grp.max()) + 1
    by_screen = pd.Series(gk.screen.first().to_numpy())
    picks = [np.concatenate([rng.choice(np.flatnonzero(by_screen == s), (by_screen == s).sum()) for s in by_screen.unique()])
             for _ in range(NBOOT)]
    names = gk.size().reset_index()[["screen", "role", "molecule"]]
    names["molecule"] = [(AR[r.screen]["ua"] if r.role == "1" else AR[r.screen]["ub"])[r.molecule] for r in names.itertuples()]
    names["role"] = names.role.astype(int)
    an = json.loads(APPROVED["anatomy"].read_text(encoding="utf-8"))
    share_ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    assert redraws == an["share"]["redrawn"] and np.allclose(share_ci, an["share"]["ci"], atol=1e-12, rtol=0), (redraws, share_ci)
    return D, grp, ng, names, np.array(picks), redraws


def released_interval(A1):
    """Public mode: the frozen run's records from the frozen display files (the values of the replay), and its 2,000 group
    resamples from the released draws file; groups numbered as in the replay (array, role, molecule index)."""
    AR = history.partner_arrays(A1)
    recs, _, _ = released_records(AR, E.owner_roles())
    D = pd.DataFrame([dict(screen=r["screen"], role=str(r["role"]), pair=r["pair"], molecule=r["molecule_index"], sst=r["sst"],
                           **{f"sse_{k}": r[f"sse_{k}"] for k in KS}) for r in recs])
    gk = D.groupby(["screen", "role", "molecule"])
    grp = gk.ngroup().to_numpy(); ng = int(grp.max()) + 1
    names = gk.size().reset_index()[["screen", "role", "molecule"]]
    names["molecule"] = [(AR[r.screen]["ua"] if r.role == "1" else AR[r.screen]["ub"])[r.molecule] for r in names.itertuples()]
    names["role"] = names.role.astype(int)
    dr = pd.read_csv(INTERVAL_DRAWS)
    picks = np.array([[int(g) - 1 for g in s.split(";")] for s in dr.groups])
    assert picks.shape == (NBOOT, ng) and (dr.draw.to_numpy() == np.arange(1, NBOOT + 1)).all()
    by_screen = gk.screen.first().to_numpy()
    uniq = pd.unique(by_screen)
    blocks = np.repeat(uniq, [int((by_screen == s).sum()) for s in uniq])        # each resample draws within arrays, array by array
    assert all((by_screen[row] == blocks).all() for row in picks), "a resample leaves its array"
    an = json.loads(APPROVED["anatomy"].read_text(encoding="utf-8"))
    return D, grp, ng, names, picks, an["share"]["redrawn"]


def compute_a(A1):
    P = records_from_public(A1)
    D, grp, ng, names, picks, redraws = (released_interval if PUBLIC_MODE else replay_interval)(A1)
    ref = pd.read_csv(APPROVED["depth"])
    assert len(P) == len(D) == len(ref) == 1247
    key = lambda d: d.assign(role=d.role.astype(int)).set_index(["screen", "role", "pair"])
    Dk, Rk = key(D), key(ref)
    dev_replay = max(float((Dk[c] - Rk[c]).abs().max()) for c in ["sst"] + [f"sse_{k}" for k in KS])
    Pk = P.set_index(["screen", "role", "target_pair"])
    names_idx = {(r.screen, r.role, r.molecule): i for i, r in enumerate(names.itertuples())}
    AR = history.partner_arrays(A1)
    D["target_pair"] = [AR[r.screen]["pairs"][r.pair] for r in D.itertuples()]
    Q = Pk.loc[list(zip(D.screen, D.role.astype(int), D.target_pair))]
    dev_public = max(float(np.abs(Q[c].to_numpy() - D[c].to_numpy()).max()) for c in ["sst"] + [f"sse_{k}" for k in KS])
    assert dev_replay < E.TOL and dev_public < E.TOL, (dev_replay, dev_public)
    curve = {k: float(1 - P[f"sse_{k}"].sum() / P.sst.sum()) for k in KS}
    fr = pd.read_csv(APPROVED["curve"]).set_index("partners").pooled_r2_percent
    assert all(abs(100 * curve[k] - fr[k]) < 1e-7 for k in KS)
    P["group"] = [names_idx[(r.screen, r.role, r.molecule)] for r in P.itertuples()]
    G = P.assign(gain=P.sse_1 - P.sse_6).groupby("group")[["gain", "sst"]].sum().reindex(range(ng))
    G = pd.concat([names, G.reset_index(drop=True), P.groupby("group").size().reindex(range(ng)).rename("records").reset_index(drop=True)], axis=1)
    vals = G.gain.to_numpy()[picks].sum(1) / G.sst.to_numpy()[picks].sum(1)
    ci = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    diff = float((P.sse_1 - P.sse_6).sum() / P.sst.sum())
    an = json.loads(APPROVED["anatomy"].read_text(encoding="utf-8"))["depth"]
    assert abs(diff - an["difference"][0]) < E.TOL and np.allclose(ci, an["difference"][1:], atol=1e-12, rtol=0), (diff, ci)
    draws = pd.DataFrame(dict(draw=np.arange(1, NBOOT + 1), groups=[";".join(str(g + 1) for g in row) for row in picks], difference=vals))
    return dict(P=P, curve=curve, G=G, ci=ci, diff=diff, draws=draws, redraws=redraws, dev=max(dev_replay, dev_public))


def draw(a):
    fig, cv = E.figure(H, "ed7")
    muted, text_c = fs.C["muted"], fs.C["text"]
    E.heading(fig, cv, 0, 0.6, None, TITLE, H, NOTE)
    ax = fs.axes_mm(fig, 12.0, 15.0, 56.0, 36.0, H); ax.set_gid("a|axes")
    ax.set_xlim(0.6, 6.4); ax.set_ylim(0, 15.5)
    xs, ys = np.array(KS, float), np.array([100 * a["curve"][k] for k in KS])
    ax.plot(xs, ys, color=text_c, lw=1.0, zorder=2, gid="a|line")
    for k_, v in zip(xs, ys):
        ax.plot([k_], [v], marker="o", ms=2.6, mew=0, mfc=text_c, ls="none", zorder=3, gid=f"a|point|{int(k_)}")
        ax.text(k_, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=fs.PT["tiny"], gid=f"a|value|{int(k_)}")
    lo, hi = a["ci"]
    ax.text(6.4, 1.4, f"six partners against one: +{100 * a['diff']:.1f} ({100 * lo:.1f}–{100 * hi:.1f})", ha="right", va="center",
            fontsize=fs.PT["tiny"], color=muted, gid="a|difference")
    ax.set_xticks(list(KS)); ax.set_yticks([0, 5, 10, 15])
    ax.set_xlabel("Earlier partners in the history", labelpad=1.5)
    ax.set_ylabel("Held-out yield profile explained (%)", labelpad=1.5)
    E.clean(ax)
    return fig


def source_data(a):
    curve = pd.DataFrame([dict(k=k, additive_profile_explained_pct=100 * a["curve"][k], records=len(a["P"]), groups=len(a["G"])) for k in KS])
    curve = pd.concat([curve, pd.DataFrame([dict(k=None, statistic="R2(6) - R2(1)", value=a["diff"], ci_lower=a["ci"][0], ci_upper=a["ci"][1])])],
                      ignore_index=True).astype({"k": "Int64", "records": "Int64", "groups": "Int64"})
    G = a["G"].assign(group_number=np.arange(1, len(a["G"]) + 1), array=lambda d: d.screen.map(fs.ARRAY_LABEL))[
        ["group_number", "screen", "array", "role", "molecule", "records", "gain", "sst"]].rename(columns={"gain": "sum_sse1_minus_sse6"})
    readme = pd.DataFrame([
        dict(sheet="ED7a", panel="a", estimand="additive residual response profile, both roles pooled", rows="one per k, then R2(6) - R2(1)",
             units="percent; fraction", transformation=TRANSFORM["a"]),
        dict(sheet="ED7a_groups", panel="a", estimand="as ED7a", rows="one per array-role-molecule group (150), numbered as in the draws "
             "file", units="sums of squared errors", transformation="Sums over the group's records of SSE_1 - SSE_6 and of SST; the "
                                                                      "records themselves are in Figure 4's Source Data compatibility sheet Fig4a_records."),
        dict(sheet="FigS7_depth_interval_draws.csv (separate file)", panel="a",
             rows="one per resample (2,000): the 150 group numbers drawn (as in ED7a_groups, ';' between them) and the resample's value",
             units="group numbers; fraction", transformation="The frozen anatomy run's own draws (they follow its depth subsets and its "
                                                               "share interval in one stream)."),
        dict(sheet="(all)", panel="-", estimand="-", rows="-", units="-",
             transformation="Measurements: DATASET_analysis.csv.gz (A1). Panel a reuses 05_DATA/Analysis/figure_display/Fig4a_folds.csv and "
                            "05_DATA/Analysis/figure_display/Fig4a_history_draws.csv. Owner roles: Anatomy_arrays.csv. Input hashes: "
                            "FigS7_SourceData_provenance.json.")])
    return dict(README=readme, ED7a=curve, ED7a_groups=G)


def main(out=None):
    A1, _ = E.public_units()
    a = compute_a(A1)
    fig = draw(a)
    inputs = dict(A1=dict(file=E.rel(E.RAW), sha256=E.sha_file(E.RAW), rows=int(len(A1))),
                  released={k: dict(file=E.rel(p), sha256=E.sha_file(p)) for k, p in PUBLIC.items()},
                  fold_registry=None if PUBLIC_MODE else dict(file=E.rel(FOLDS_REG), sha256=E.sha_file(FOLDS_REG),
                                                              entries_used="history|<array>|cv|0"),
                  owner_roles=dict(file=E.rel(E.ANATOMY_ARRAYS), sha256=E.sha_file(E.ANATOMY_ARRAYS)),
                  frozen={k: dict(file=E.rel(p), sha256=E.sha_file(p)) for k, p in APPROVED.items()})
    if PUBLIC_MODE:                                   # the released group resamples read in place of the registry replay
        inputs["released"]["depth_interval_draws"] = dict(file=E.rel(INTERVAL_DRAWS), sha256=E.sha_file(INTERVAL_DRAWS))
        del inputs["fold_registry"]
    prov = dict(figure=f"Fig. S{N} | {TITLE}", generated_by=E.rel(__file__),
                inputs=inputs,
                randomness=dict(a="default_rng(20261003): depth subsets, then 2,000 share resamples with "
                                  f"{a['redraws']} redraws, then the 2,000 depth-interval resamples (released)"),
                transformation=TRANSFORM, curve_pct={k: 100 * v for k, v in a["curve"].items()},
                difference=dict(value=a["diff"], ci=a["ci"]), agreement_with_frozen=dict(depth_records=a["dev"]))
    p = E.write(N, fig, H, source_data(a), prov, out, extra=(("draws", "FigS7_depth_interval_draws.csv"),),
                tables=(("draws", a["draws"]),))
    print(f"Fig. S{N}: pooled {', '.join(f'{100 * a['curve'][k]:.1f}' for k in KS)}; 6 vs 1 {100 * a['diff']:.1f} "
          f"({100 * a['ci'][0]:.1f}-{100 * a['ci'][1]:.1f}); wrote {len(p)} files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    main(ap.parse_args().out)
