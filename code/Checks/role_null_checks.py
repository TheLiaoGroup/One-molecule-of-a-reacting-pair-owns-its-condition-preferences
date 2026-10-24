"""Per-array owner margins, an equal-role null and a class-proxy test, from the released measurements (Bruce, 2026-10-13, in answer
to the third external review).

1 Per-array owner-minus-partner margins (table S1): the molecules of each role are resampled within the array, 2,000
  draws from a stream seeded 20261012 for each array, fitted residuals held fixed; the share of draws on the owner.
2 Equal-role null: on each array's real network of product-forming pairs and its real missing cells, yields are simulated with the
  array's condition means, a random intercept and slope per pair, and molecule-by-condition effects of equal variance for the two
  roles (together the array's measured molecular share), plus independent error. The condition pattern and both role shares are
  recomputed exactly as for the data, and the owner index (larger share over the sum) is taken; its median over the 15 arrays is
  compared with the measured median. Molecule effects are independent across conditions, or built on 8 or 4 condition factors
  shared by all molecules (the measured patterns have a median effective rank of about 11). 500 simulations each, seed 20261013.
3 Class proxy in SNAr: on the real SNAr network, yields carry a nucleophile-class-by-condition effect and independent error, and no
  aryl-halide or nucleophile-identity effect. The aryl-halide share and how often it reaches the measured 0.456 are recorded for
  class effects from 10% to 95% of the pattern variance. 1,000 simulations per level, seed 20261013.
4 SNAr nesting (Supplementary Note 9): among the pairs of the two retained nucleophile classes, the aryl halides that meet both
  classes, the in-sample fit of aryl-halide identity, and the class fit after each aryl halide's mean pattern is subtracted.
Writes 05_DATA/Results/directional/role_null_checks.json, or the file named by --out.
    python 06_CODE/Checks/role_null_checks.py [--out JSON] [--sims N]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

threadpool_limits(1)        # one BLAS thread: the eigendecompositions sum in a fixed order, so a rerun gives the same bits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "06_CODE/Figures"), str(ROOT / "06_CODE/Core/00_common"), str(ROOT / "06_CODE/Core/00_common/ctb")]
import figure1_data as F1                                                   # noqa: E402
from anatomy import FORMS, Shares, molecule_weights, pattern               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "05_DATA/Results/directional/role_null_checks.json"))
ap.add_argument("--sims", type=int, default=500)
ARGS = ap.parse_args()
PUB = pd.read_csv(ROOT / "05_DATA/Results/display_inputs/Anatomy_arrays.csv").set_index("screen")
OWN = PUB.owner_role.astype(int).to_dict()
SNAR = "SNAr|1"
LEVELS = (0.10, 0.25, 0.50, 0.75, 0.95)


def shares(M, a, b):
    E = pattern(M)[0]
    return Shares(E, a, b), E


def owner_index(s):
    hi, tot = max(s["1"], s["2"]), s["1"] + s["2"]
    return hi / tot if tot > 0 else np.nan


def simulate(P, a, b, mol_var, rng, k=None, class_of=None):
    """Yields on the real network: condition means, a random intercept and slope per pair, role effects of equal variance
    mol_var / 2 each (or, with class_of, one class effect of variance mol_var), independent error; missing where the data are."""
    n, C = P.shape
    g = np.nanmean(P, 0)
    alpha, beta = rng.normal(0, 0.1, n), rng.normal(1, 0.2, n)

    def effects(keys, var):
        u = {x: i for i, x in enumerate(dict.fromkeys(keys))}
        if k is None:
            Z = rng.normal(0, 1, (len(u), C))
        else:
            F = rng.normal(0, 1, (k, C)) / np.sqrt(k)
            Z = rng.normal(0, 1, (len(u), k)) @ F
        Z *= np.sqrt(var) / Z.std()
        return Z[[u[x] for x in keys]]
    if class_of is None:
        mol = effects(a, mol_var / 2) + effects(b, mol_var / 2)
    else:
        mol = effects([class_of[x] for x in a], mol_var)
    sd = np.nanstd(P - np.nanmean(P, 1, keepdims=True))
    Y = alpha[:, None] + beta[:, None] * g[None, :] + sd * (mol + rng.normal(0, np.sqrt(max(1 - mol_var, 1e-6)), (n, C)))
    return np.where(np.isfinite(P), Y, np.nan)


def main():
    A1, _ = F1.public_units()
    AR = {}
    for key, g in A1.groupby("screen"):
        P, a, b = F1.matrix(g)
        M = P.to_numpy(float)
        act = np.nanmax(M, 1) >= FORMS
        AR[key] = dict(M=M[act], a=a[act], b=b[act])
    res = {}

    # ------------------------------------------------------------ 1 per-array margins
    rows = []
    for key in sorted(AR):
        X = AR[key]
        sh, _ = shares(X["M"], X["a"], X["b"])
        o, p = ("1", "2") if OWN[key] == 1 else ("2", "1")
        s0 = sh.compute(np.ones(len(X["a"])))
        rng = np.random.default_rng(20261012)
        m = []
        for _ in range(2000):
            for _t in range(50):
                r = sh.compute(molecule_weights(X["a"], X["b"], rng))
                if np.isfinite(r[o]) and np.isfinite(r[p]):
                    break
            m.append(r[o] - r[p])
        m = np.array(m)
        rows.append(dict(array=key, owner_role=int(o), margin=float(s0[o] - s0[p]), lower=float(np.percentile(m, 2.5)),
                         upper=float(np.percentile(m, 97.5)), draws_on_owner=float((m > 0).mean()), share1=float(s0["1"]),
                         share2=float(s0["2"])))
    res["margins"] = rows
    observed = {r["array"]: owner_index({"1": r["share1"], "2": r["share2"]}) for r in rows}
    obs_median = float(np.median(list(observed.values())))
    mol = {r["array"]: r["share1"] + r["share2"] for r in rows}

    # ------------------------------------------------------------ 2 equal-role null
    rng = np.random.default_rng(20261013)
    null = {}
    for name, k in (("independent conditions", None), ("8 shared condition factors", 8), ("4 shared condition factors", 4)):
        meds = []
        for _ in range(ARGS.sims):
            idx = []
            for key in sorted(AR):
                X = AR[key]
                Y = simulate(X["M"], X["a"], X["b"], max(mol[key], 0.02), rng, k=k)
                sh, _ = shares(Y, X["a"], X["b"])
                idx.append(owner_index(sh.compute(np.ones(len(X["a"])))))
            meds.append(float(np.nanmedian(idx)))
        meds = np.array(meds)
        null[name] = dict(simulations=int(len(meds)), median=float(np.median(meds)), p95=float(np.percentile(meds, 95)),
                          reaching_observed=int((meds >= obs_median).sum()))
    res["equal_role_null"] = dict(observed_median_owner_index=obs_median, models=null)

    # ------------------------------------------------------------ 2b equal-role null on each array's measured condition covariance
    # (added 2026-10-14 in answer to the fifth review: role effects and pair error are drawn with the covariance of the array's
    # own measured condition patterns, so the simulated patterns have the measured effective rank; a separate stream)
    def eff_rank(E):
        S = pd.DataFrame(E).cov(min_periods=3).fillna(0).to_numpy()
        lam = np.clip(np.linalg.eigvalsh((S + S.T) / 2), 0, None)
        p = lam / lam.sum()
        p = p[p > 0]
        return float(np.exp(-(p * np.log(p)).sum()))
    basis, obs_rank = {}, {}
    for key in sorted(AR):
        E = pattern(AR[key]["M"])[0]
        S = pd.DataFrame(E).cov(min_periods=3).fillna(0).to_numpy()
        lam, V = np.linalg.eigh((S + S.T) / 2)
        lam = np.clip(lam, 0, None)
        L = V * np.sqrt(lam / lam.mean())          # draws z @ L.T have the measured covariance, scaled to unit mean variance
        basis[key] = L
        obs_rank[key] = eff_rank(E)
    rng_c = np.random.default_rng(20261014)
    meds, ranks = [], []
    for _ in range(ARGS.sims):
        idx, rk = [], []
        for key in sorted(AR):
            X, L = AR[key], basis[key]
            P = X["M"]
            n, C = P.shape
            m = max(mol[key], 0.02)
            g = np.nanmean(P, 0)
            alpha, beta = rng_c.normal(0, 0.1, n), rng_c.normal(1, 0.2, n)

            def eff(keys, var):
                u = {x: i for i, x in enumerate(dict.fromkeys(keys))}
                Z = rng_c.normal(0, 1, (len(u), C)) @ L.T * np.sqrt(var)
                return Z[[u[x] for x in keys]]
            noise = rng_c.normal(0, 1, (n, C)) @ L.T * np.sqrt(max(1 - m, 1e-6))
            sd = np.nanstd(P - np.nanmean(P, 1, keepdims=True))
            Y = alpha[:, None] + beta[:, None] * g[None, :] + sd * (eff(X["a"], m / 2) + eff(X["b"], m / 2) + noise)
            Y = np.where(np.isfinite(P), Y, np.nan)
            sh, Ey = shares(Y, X["a"], X["b"])
            idx.append(owner_index(sh.compute(np.ones(len(X["a"])))))
            rk.append(eff_rank(Ey))
        meds.append(float(np.nanmedian(idx))); ranks.append(float(np.median(rk)))
    meds = np.array(meds)
    res["equal_role_null_measured_covariance"] = dict(
        simulations=int(len(meds)), median=float(np.median(meds)), p95=float(np.percentile(meds, 95)),
        reaching_observed=int((meds >= obs_median).sum()), observed_median_owner_index=obs_median,
        effective_rank_observed_median=float(np.median(list(obs_rank.values()))), effective_rank_simulated_median=float(np.median(ranks)))

    # ------------------------------------------------------------ 2c molecules weighted equally, and how comparisons concentrate
    eq_rows, conc = [], []
    for key in sorted(AR):
        X = AR[key]
        sh, _ = shares(X["M"], X["a"], X["b"])
        V = float(sh.var.mean())
        eq = {}
        for role, keys in (("1", X["a"]), ("2", X["b"])):
            CR, MR = sh.CR[role], sh.MR[role]
            T, cnt = [], []
            for mname in dict.fromkeys(keys):
                ix = np.flatnonzero(keys == mname)
                d = MR[np.ix_(ix, ix)].sum()
                cnt.append(d)
                if d > 0:
                    T.append(CR[np.ix_(ix, ix)].sum() / d / V)
            eq[role] = float(np.mean(T)) if T else np.nan
            c = np.sort(np.array(cnt))[::-1]
            conc.append(dict(array=key, role=int(role), molecules_compared=int((c > 0).sum()),
                             top3_share=float(c[:3].sum() / c.sum()) if c.sum() else np.nan,
                             top1_share=float(c[0] / c.sum()) if c.sum() else np.nan))
        o_eq = 1 if eq["1"] >= eq["2"] else 2
        eq_rows.append(dict(array=key, share1=eq["1"], share2=eq["2"], owner=o_eq, owner_published=OWN[key],
                            molecular=eq["1"] + eq["2"], owner_index=owner_index(eq)))
    EQ, CO = pd.DataFrame(eq_rows), pd.DataFrame(conc)
    res["molecules_weighted_equally"] = dict(
        rows=eq_rows, molecular_median=float(EQ.molecular.median()), owner_index_median=float(EQ.owner_index.median()),
        same_owner=int((EQ.owner == EQ.owner_published).sum()),
        different_owner=sorted(EQ.array[EQ.owner != EQ.owner_published]),
        comparison_concentration=dict(roles=int(len(CO)), top3_share_median=float(CO.top3_share.median()),
                                      top1_share_max=float(CO.top1_share.max()), top1_share_max_array=str(CO.array[CO.top1_share.idxmax()])))

    # ------------------------------------------------------------ 3 class proxy in SNAr
    import amplitude
    X = AR[SNAR]
    cls = {x: dict(amplitude.CLASS_ARRAYS)[SNAR](x) for x in dict.fromkeys(X["a"])}
    sh, _ = shares(X["M"], X["a"], X["b"])
    halide_obs = float(sh.compute(np.ones(len(X["a"])))["2"])
    raw = pd.Series([cls[x] for x in X["a"]])
    top = [c for c in raw.value_counts().index if c != "other" and raw.value_counts()[c] >= 3][:2]
    keep = raw.isin(top).to_numpy()

    def explained(Z, z):
        gm = np.nanmean(Z, 0)
        fit = np.zeros_like(Z)
        for c in np.unique(z):
            fit[z == c] = np.nanmean(Z[z == c], 0)
        return float(1 - np.nansum((Z - fit) ** 2) / np.nansum((Z - gm) ** 2))
    E = pattern(X["M"])[0]
    class_fit_obs = explained(E[keep], raw[keep].to_numpy())
    proxy = []
    for lev in LEVELS:
        hs, fits, owners = [], [], 0
        for _ in range(2 * ARGS.sims):
            Y = simulate(X["M"], X["a"], X["b"], lev, rng, class_of=cls)
            sh_y, E_y = shares(Y, X["a"], X["b"])
            s = sh_y.compute(np.ones(len(X["a"])))
            hs.append(s["2"]); owners += s["2"] > s["1"]
            fits.append(explained(E_y[keep], raw[keep].to_numpy()))    # the class fit the data would then show
        hs, fits = np.array(hs), np.array(fits)
        proxy.append(dict(class_variance=lev, simulations=int(len(hs)), halide_share_median=float(np.median(hs)),
                          halide_share_p95=float(np.percentile(hs, 95)), reaching_observed=int((hs >= halide_obs).sum()),
                          halide_named_owner=int(owners), class_fit_median=float(np.median(fits)),
                          class_fit_p05=float(np.percentile(fits, 5))))
    res["snar_class_proxy"] = dict(observed_halide_share=halide_obs, observed_class_fit=class_fit_obs, pairs=int(len(X["a"])),
                                   conditions=int(X["M"].shape[1]),
                                   classes={c: int(sum(v == c for v in cls.values())) for c in sorted(set(cls.values()))}, levels=proxy)

    # ------------------------------------------------------------ 4 SNAr nesting
    sub, hal, lab = E[keep], X["b"][keep], raw[keep].to_numpy()
    hal_mean = {h: np.nanmean(sub[hal == h], 0) for h in np.unique(hal)}
    resid = sub - np.array([hal_mean[h] for h in hal])
    both = [h for h in np.unique(hal) if len(set(lab[hal == h])) == 2]
    res["snar_nesting"] = dict(pairs=int(keep.sum()), nucleophiles=int(len(set(X["a"][keep]))), aryl_halides=int(len(np.unique(hal))),
                               aryl_halides_meeting_both_classes=int(len(both)), pairs_of_those=int(np.isin(hal, both).sum()),
                               class_fit=explained(sub, lab), aryl_halide_fit=explained(sub, hal),
                               class_fit_after_aryl_halide=explained(resid, lab), classes=top)
    Path(ARGS.out).write_bytes((json.dumps(res, indent=1, default=float) + "\n").encode("utf-8"))
    print(json.dumps({"null": res["equal_role_null"], "null_cov": res["equal_role_null_measured_covariance"], "equal": {k: v for k, v in res["molecules_weighted_equally"].items() if k != "rows"}}, indent=1, default=float))


if __name__ == "__main__":
    main()
