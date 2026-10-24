"""Section 1: what one table of conditions costs.

Per array: the variance decomposition (substrate pair, condition, their interaction), the single
best-on-average condition against each pair's own best, and two fitted references built from the
array's own main effects:
  additive + noise    rows + columns, no interaction, measurement error added
  bounded + noise     yield = cap * logistic(row + column), fitted by least squares, error added
Writes out/cost_arrays.csv, out/cost_references.csv and out/cost.json.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from common import load_analysis, arrays, labels, save_json, OUT, USABLE, DEAD

NDRAW = 200


def decomposition(M):
    fin = np.isfinite(M); gm = np.nanmean(M)
    rm = np.nanmean(M, axis=1, keepdims=True); cm = np.nanmean(M, axis=0)[None, :]
    tot = np.sum((M - gm)[fin] ** 2)
    ones = np.ones_like(M)
    return dict(substrate=100 * np.sum(((rm - gm) * ones)[fin] ** 2) / tot,
                condition=100 * np.sum(((cm - gm) * ones)[fin] ** 2) / tot,
                interaction=100 * np.sum((M - rm - cm + gm)[fin] ** 2) / tot)


def opportunity(M):
    """The best-on-average condition (highest column mean) against each pair's own best. Every quantity is
    taken over the pairs measured under that condition, so the columns of Table 1 share one denominator."""
    bc = int(np.argmax(np.nanmean(M, axis=0)))
    col, mx = M[:, bc], np.nanmax(M, axis=1)
    obs = np.isfinite(col)
    nmax = np.nansum(M[obs] == mx[obs][:, None], axis=1)
    col, mx = col[obs], mx[obs]
    return dict(best_condition=bc,
                sole_pct=100 * np.mean(np.isclose(col, mx) & (nmax == 1)),
                below5_pct=100 * np.mean(col < DEAD),
                never5_pct=100 * np.mean(mx < DEAD),
                rescuable_pct=100 * np.mean((col < DEAD) & (mx >= USABLE)),
                gap_pp=100 * np.mean(mx - col),
                n_global_measured=int(obs.sum()))


def null_interaction(M, ndraw=NDRAW):
    """Interaction share of an interaction-free matrix: main effects plus error max(7% of yield,
    1.2 points), clipped to [0, 1], on the observed cells only."""
    msk = np.isfinite(M); gm = np.nanmean(M)
    add = np.clip(np.nanmean(M, axis=1, keepdims=True) + np.nanmean(M, axis=0)[None, :] - gm, 0.0, 1.0)
    vals = []
    for d in range(ndraw):
        rs = np.random.RandomState(7000 + d)
        sim = np.where(msk, np.clip(add + rs.normal(0, 1, M.shape) * np.maximum(0.07 * add, 0.012), 0, 1), np.nan)
        vals.append(decomposition(sim)["interaction"])
    return float(np.mean(vals))


def bounded_fit(y):
    """yield = cap * logistic(a_i + b_j) by least squares on observed cells (L-BFGS-B, two starts for
    the cap); the fixed cap = 1 fit is kept if the free cap lands on a worse optimum."""
    def fit(free_upper):
        n, c = y.shape; i, j = np.where(np.isfinite(y)); obs = y[i, j]; size = len(obs)
        gm = np.nanmean(y); cm = np.nanmean(y, axis=0); rm = np.nanmean(y, axis=1)
        b = logit(np.clip(cm, 1e-4, 1 - 1e-4)) - logit(np.clip(gm, 1e-4, 1 - 1e-4))
        a = logit(np.clip(rm, 1e-4, 1 - 1e-4)) + b[0]; b = b - b[0]
        start = np.r_[a, b[1:], 1.] if free_upper else np.r_[a, b[1:]]

        def objective(v):
            upper = v[-1] if free_upper else 1.
            s = expit(v[:n][i] + np.r_[0., v[n:n + c - 1]][j]); err = upper * s - obs
            ge = 2 * err * upper * s * (1 - s) / size
            grad = np.r_[np.bincount(i, weights=ge, minlength=n), np.bincount(j, weights=ge, minlength=c)[1:]]
            if free_upper:
                grad = np.r_[grad, 2 * np.mean(err * s)]
            return np.mean(err * err), grad
        bounds = [(-25, 25)] * (n + c - 1) + ([(0.001, 1.)] if free_upper else [])
        starts = [start]
        if free_upper:
            alt = start.copy(); alt[-1] = .5; starts.append(alt)
        lo = [bb[0] for bb in bounds]; hi = [bb[1] for bb in bounds]
        fits = [minimize(objective, np.clip(x, lo, hi), jac=True, method="L-BFGS-B", bounds=bounds,
                         options=dict(maxiter=2000, ftol=1e-12, gtol=1e-9, maxls=40)) for x in starts]
        best = min(fits, key=lambda f: f.fun); v = best.x; upper = v[-1] if free_upper else 1.
        return upper * expit(v[:n, None] + np.r_[0., v[n:n + c - 1]][None, :]), bool(best.success)
    fixed, ok_fixed = fit(False)
    free, ok_free = fit(True)
    if np.nansum((y - free) ** 2) > np.nansum((y - fixed) ** 2) + 1e-7:
        free, ok_free = fixed, ok_fixed
    return free, ok_free


def reference_opportunity(M, expectation, seed):
    rng = np.random.default_rng(seed); obs = np.isfinite(M); vals = []
    for _ in range(NDRAW):
        eps = rng.normal(size=M.shape) * np.maximum(.07 * np.maximum(expectation, 0), .012)
        vals.append(opportunity(np.where(obs, np.clip(expectation + eps, 0, 1), np.nan)))
    return {k: float(np.mean([v[k] for v in vals])) for k in ("sole_pct", "gap_pp")}


def main():
    d = load_analysis()
    lab = labels(d)
    rows, refs = [], []
    for ai, (key, (pairs, conds, M)) in enumerate(arrays(d).items()):
        rec = dict(screen=key, label=lab[key], pairs=len(pairs), conditions=len(conds),
                   measured=int(np.isfinite(M).sum()), designed=int(M.size))
        rec.update(decomposition(M)); rec.update(opportunity(M))
        rec["best_condition"] = conds[rec["best_condition"]]
        rec["null_interaction"] = null_interaction(M)
        gm = np.nanmean(M)
        additive = np.nanmean(M, axis=1, keepdims=True) + np.nanmean(M, axis=0, keepdims=True) - gm
        bounded, ok = bounded_fit(M)
        tot = np.nansum((M - gm) ** 2)
        rec["bounded_residual"] = 100 * np.nansum((M - bounded) ** 2) / tot
        rec["bounded_converged"] = ok
        for name, expectation in (("additive", additive), ("bounded", bounded)):
            r = reference_opportunity(M, expectation, 20260920 + ai)
            refs.append(dict(screen=key, reference=name, **r))
        rows.append(rec)
        print(f"  {key:28s} interaction {rec['interaction']:5.1f}%  gap {rec['gap_pp']:5.1f}  "
              f"rescuable {rec['rescuable_pct']:4.1f}%  bounded residual {rec['bounded_residual']:5.1f}%", flush=True)
    T = pd.DataFrame(rows); R = pd.DataFrame(refs)
    T.to_csv(OUT / "cost_arrays.csv", index=False); R.to_csv(OUT / "cost_references.csv", index=False)
    mol = pd.unique(pd.concat([d.sub_1_smiles, d.sub_2_smiles]).dropna())
    out = dict(
        n_measurements=int(len(d)), n_arrays=int(len(T)), n_transformations=int(d.transformation.nunique()),
        n_pairs=int(T.pairs.sum()), n_molecule_strings=int(len(mol)),
        designed_positions=int(T.designed.sum()), coverage_pct=100 * T.measured.sum() / T.designed.sum(),
        median_interaction=float(T.interaction.median()), median_condition=float(T.condition.median()),
        median_substrate=float(T.substrate.median()),
        n_interaction_gt_condition=int((T.interaction > T.condition).sum()),
        interaction_min=float(T.interaction.min()), interaction_max=float(T.interaction.max()),
        median_null_interaction=float(T.null_interaction.median()),
        median_sole=float(T.sole_pct.median()), median_below5=float(T.below5_pct.median()),
        median_never5=float(T.never5_pct.median()), median_rescuable=float(T.rescuable_pct.median()),
        rescuable_min=float(T.rescuable_pct.min()), rescuable_max=float(T.rescuable_pct.max()),
        n_rescuable_positive=int((T.rescuable_pct > 0).sum()),
        median_gap=float(T.gap_pp.median()), gap_min=float(T.gap_pp.min()), gap_max=float(T.gap_pp.max()),
        median_bounded_residual=float(T.bounded_residual.median()),
        all_bounded_converged=bool(T.bounded_converged.all()),
        median_gap_additive_ref=float(R[R.reference == "additive"].gap_pp.median()),
        median_gap_bounded_ref=float(R[R.reference == "bounded"].gap_pp.median()),
        median_sole_additive_ref=float(R[R.reference == "additive"].sole_pct.median()),
        median_sole_bounded_ref=float(R[R.reference == "bounded"].sole_pct.median()))
    save_json("cost.json", out)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()})


if __name__ == "__main__":
    main()
