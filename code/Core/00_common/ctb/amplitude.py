"""Section 2: how strongly a substrate pair responds (amplitude) versus which conditions it prefers.

  shared mode      first singular component of the additive residual; its condition loading against
                   the conditions' mean yield
  slope baseline   y_ij = a_i + s_i c_j with the condition signature c learned on the other half of the
                   pairs and a_i, s_i on the pair's training conditions; error on held-out conditions
  reagent profiles what reagent-level profiles add on top of the slope baseline, same held-out cells
  class slopes     descriptive slope of every pair against the signature of the other half; median over
                   molecules of each chemical class; seven pair splits
  SNAr bases       class profiles over base families after removing each pair's slope
Writes out/amplitude_arrays.csv, out/amplitude_classes.csv, out/amplitude_snar.csv, out/amplitude.json.
"""
from boundaries import frozen_folds
import numpy as np
import pandas as pd
from rdkit import Chem
from common import load_analysis, load_conditions, arrays, labels, save_json, OUT
from roles import condition_variables

PAIR_SEED, FOLD_SEED, NFOLD, LIVE, SHRINK = 20260920, 20260909, 5, 0.02, 4.0
SPLIT_SEEDS = [20260920, 21, 22, 23, 24, 25, 26]


# ---------------------------------------------------------------- fits
def additive(Y):
    """Grand mean, pair and condition effects by 100 rounds of alternating means over observed cells."""
    mu = np.nanmean(Y); a = np.zeros(len(Y)); b = np.zeros(Y.shape[1])
    for _ in range(100):
        a = np.nanmean(Y - mu - b[None, :], axis=1)
        b = np.nanmean(Y - mu - a[:, None], axis=0)
    return mu, a, b


def slope_fit(Y, c, cols):
    """Per-pair intercept and slope of yield on the signature c by least squares over the observed cells
    in cols (a pair with fewer than two usable cells keeps slope 1). Negative slopes are kept."""
    a = np.zeros(len(Y)); s = np.zeros(len(Y)); fallback = 0
    for i, row in enumerate(Y):
        ix = cols[np.isfinite(row[cols])]; x = c[ix]; y = row[ix]
        if len(ix) < 2 or np.sum((x - x.mean()) ** 2) < 1e-18:
            s[i] = 1.0; a[i] = np.mean(y - x); fallback += 1
        else:
            s[i] = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
            a[i] = y.mean() - s[i] * x.mean()
    return a[:, None] + s[:, None] * c[None, :], a, s, fallback


def shared_mode(Y):
    mu, a, b = additive(Y)
    R = np.nan_to_num(Y - mu - a[:, None] - b[None, :])
    _, sv, vt = np.linalg.svd(R, full_matrices=False)
    return sv[0] ** 2 / np.sum(sv ** 2), abs(np.corrcoef(vt[0], np.nanmean(Y, axis=0))[0, 1])


def halves(n, seed=PAIR_SEED, *, screen):
    h = frozen_folds(f"halves|{screen}|{seed}")
    if sorted(np.concatenate(h).tolist()) != list(range(n)):
        raise ValueError("Pair-half population mismatch")
    return [(h[0], h[1]), (h[1], h[0])]


def condition_folds(m):
    return frozen_folds(f"conditions|{m}")


def backfit_profiles(R, V, train, iters=15):
    """Additive model of the residual over all condition variables' shrunken level means, fitted on the
    training conditions by backfitting. Returns {variable: {level: pair vector}}."""
    vars_ = list(V)
    Rt = R[:, train]; fitted = np.zeros_like(Rt); comp = {v: np.zeros_like(Rt) for v in vars_}; F = {v: {} for v in vars_}
    for _ in range(iters):
        for v in vars_:
            part = Rt - (fitted - comp[v]); lab = V[v].values[train]; new = np.zeros_like(Rt)
            for lev in np.unique(lab):
                cols = np.where(lab == lev)[0]; sub = part[:, cols]; cnt = np.sum(~np.isnan(sub), 1)
                est = np.where(cnt > 0, np.nansum(sub, 1) / (cnt + SHRINK), 0.0); F[v][lev] = est; new[:, cols] = est[:, None]
            fitted = fitted - comp[v] + new; comp[v] = new
    return F


def heldout(Y, V, *, screen):
    """Held-out-condition error of the additive baseline, the slope baseline, and the slope baseline plus
    reagent-level profiles. Signature from the other pair half; two halves exchanged; five condition folds."""
    m = Y.shape[1]; sse = dict(mean_only=0.0, additive=0.0, slope=0.0, slope_profiles=0.0)
    for er, cr in halves(len(Y), screen=screen):
        mu, _, b = additive(Y[cr]); c = mu + b; E = Y[er]
        for te in condition_folds(m):
            tr = np.setdiff1d(np.arange(m), te); target = E[:, te]; ok = np.isfinite(target)
            base_add = (np.nanmean(E[:, tr] - c[tr], axis=1))[:, None] + c[None, :]
            base_mean = np.nanmean(E[:, tr], axis=1)[:, None] * np.ones((1, m))   # no condition term
            base_slope = slope_fit(E, c, tr)[0]
            sse["additive"] += np.sum((target[ok] - base_add[:, te][ok]) ** 2)
            sse["mean_only"] += np.sum((target[ok] - base_mean[:, te][ok]) ** 2)
            sse["slope"] += np.sum((target[ok] - base_slope[:, te][ok]) ** 2)
            if V is not None and V.shape[1]:
                F = backfit_profiles(E - base_slope, V, tr)
                joint = base_slope[:, te].copy()
                for jj, j in enumerate(te):
                    for v in V:
                        joint[:, jj] += F[v].get(V[v].values[j], 0.0)
                sse["slope_profiles"] += np.sum((target[ok] - joint[ok]) ** 2)
    out = dict(slope_gain=1 - sse["slope"] / sse["additive"],
               slope_gain_vs_mean_only=1 - sse["slope"] / sse["mean_only"],
               condition_effect_gain=1 - sse["additive"] / sse["mean_only"])
    if V is not None and V.shape[1]:
        out["profiles_r2"] = 1 - sse["slope_profiles"] / sse["slope"]
    return out


# ---------------------------------------------------------------- chemical classes (SMARTS on sub_1)
sm = Chem.MolFromSmarts


def carbonyl(s):
    m = Chem.MolFromSmiles(s); ald = m.HasSubstructMatch(sm("[CX3H1](=O)[#6]")); arom = m.HasSubstructMatch(sm("[CX3](=O)[a]"))
    if not ald and not arom and m.HasSubstructMatch(sm("[CX3](=O)[CX3]=[CX3]")):
        return "enone"
    return ("aryl " if arom else "alkyl ") + ("aldehyde" if ald else "ketone")


def nucleophile(s):
    m = Chem.MolFromSmiles(s)
    if m.HasSubstructMatch(sm("[nH]")): return "azole/lactam N–H"
    if m.HasSubstructMatch(sm("[NX3;H1,H2;$(N-C=O)]")): return "amide N–H"
    if m.HasSubstructMatch(sm("[NX3;H2;$(N-c)]")): return "aniline"
    if m.HasSubstructMatch(sm("[NX3;H1,H2;!$(N-C=O);!$(N-c);!$(N-S)]")): return "aliphatic amine"
    return "other"


def amine(s):
    m = Chem.MolFromSmiles(s)
    if m.HasSubstructMatch(sm("[NX3;H2][c]")): return "primary (hetero)aryl amine"
    if m.HasSubstructMatch(sm("[NX3;H1,H2;$(N-C=O)]")): return "amide N–H"
    if m.HasSubstructMatch(sm("[NX3;H2][C]")): return "primary alkyl amine"
    if m.HasSubstructMatch(sm("[NX3;H1]([c])")): return "N-aryl secondary amine"
    if m.HasSubstructMatch(sm("[NX3;H1]([C])[C]")): return "secondary alkyl amine"
    return "other N–H"


BASE_FAMILY = {"MsOH": "acid", "PTSA": "acid", "TFA": "acid", "AcOH": "acid", "DIPEA": "amine base", "TEA": "amine base",
               "2,6-lutidine": "amine base", "Py": "amine base", "DBU": "amine base", "Cs2CO3": "carbonate/acetate",
               "K2CO3": "carbonate/acetate", "KHCO3": "carbonate/acetate", "KOAc": "carbonate/acetate",
               "NaOtAm": "alkoxide/hydroxide", "KOtBu": "alkoxide/hydroxide", "KOH": "alkoxide/hydroxide",
               "NaOMe": "alkoxide/hydroxide"}
CLASS_ARRAYS = [("SNAr|1", nucleophile), ("Acylation|1", amine), ("Buchwald-Hartwig|1", amine),
                ("Reductive-amination|1", carbonyl)]


def class_slopes(Y, mols, classify, seed, *, screen):
    """Median over molecules (pairs reaching 2% averaged within a molecule) of the slope against the other
    half's signature, all observed conditions of the pair; also slope / the pair's highest yield."""
    slope = np.full(len(Y), np.nan)
    for er, cr in halves(len(Y), seed, screen=screen):
        mu, _, b = additive(Y[cr]); _, _, s, fb = slope_fit(Y[er], mu + b, np.arange(Y.shape[1]))
        assert fb == 0
        slope[er] = s
    mx = np.nanmax(Y, axis=1); live = mx >= LIVE
    P = pd.DataFrame(dict(molecule=mols, slope=slope, normalized=np.where(live, slope / np.where(live, mx, 1), np.nan)))[live]
    Q = P.groupby("molecule")[["slope", "normalized"]].mean()
    Q["cls"] = [classify(x) for x in Q.index]
    return Q, slope


def main():
    d = load_analysis(); C = load_conditions(); lab = labels(d)
    rxn = d.drop_duplicates("screen").set_index("screen").rxn_class.to_dict()
    A = arrays(d); rows = []
    for key, (pairs, conds, Y) in A.items():
        share, corr = shared_mode(Y)
        V = condition_variables(key, C, rxn[key], conds)
        h = heldout(Y, V, screen=key)
        rows.append(dict(screen=key, label=lab[key], mode_share=share, mode_abs_r=corr,
                         n_variables=V.shape[1], variables="|".join(V.columns), **h))
        print(f"  {key:28s} mode {share:.3f} |r| {corr:.3f}  slope gain {h['slope_gain']:.3f}  "
              f"profiles over slope {h.get('profiles_r2', np.nan):.3f}", flush=True)
    T = pd.DataFrame(rows); T.to_csv(OUT / "amplitude_arrays.csv", index=False)

    # ---- class slopes: Table 2
    crow, tab = [], []
    for key, classify in CLASS_ARRAYS:
        pairs, conds, Y = A[key]
        mols = np.array([p.split(" >> ")[0] for p in pairs])
        allcls = pd.Series(sorted(set(mols))).map(classify).value_counts()
        per_seed = {}
        for seed in SPLIT_SEEDS:
            Q, _ = class_slopes(Y, mols, classify, seed, screen=key)
            g = Q.groupby("cls")
            per_seed[seed] = pd.DataFrame(dict(n=g.size(), median_slope=g.slope.median(), median_normalized=g.normalized.median()))
            for cls, r in per_seed[seed].iterrows():
                crow.append(dict(screen=key, seed=seed, cls=cls, active=int(r.n), all=int(allcls.get(cls, 0)),
                                 median_slope=r.median_slope, median_normalized=r.median_normalized))
        prim = per_seed[PAIR_SEED]; prim = prim[prim.n >= 3]
        ca, cb = prim.median_slope.idxmax(), prim.median_slope.idxmin()
        ratios = [per_seed[s].loc[ca, "median_slope"] / per_seed[s].loc[cb, "median_slope"] for s in SPLIT_SEEDS]
        assert min(ratios) > 0
        tab.append(dict(screen=key, label=lab[key], class_a=ca, class_b=cb,
                        active_a=int(prim.loc[ca, "n"]), all_a=int(allcls[ca]), active_b=int(prim.loc[cb, "n"]), all_b=int(allcls[cb]),
                        slope_a=prim.loc[ca, "median_slope"], slope_b=prim.loc[cb, "median_slope"],
                        ratio_primary=prim.loc[ca, "median_slope"] / prim.loc[cb, "median_slope"],
                        ratio_min=min(ratios), ratio_max=max(ratios),
                        normalized_a=prim.loc[ca, "median_normalized"], normalized_b=prim.loc[cb, "median_normalized"]))
    # What the amide array's low-slope class is made of, so the text can name it: every member
    # is a primary sulfonamide except N,O-dimethylhydroxylamine.
    am_mols = sorted({p.split(" >> ")[0] for p in A["Acylation|1"][0]})
    other = [m for m in am_mols if amine(m) == "other N–H"]
    sulf = Chem.MolFromSmarts("[SX4](=O)(=O)[NX3;H2]")
    hits = [m for m in other if Chem.MolFromSmiles(m).HasSubstructMatch(sulf)]
    rest = sorted(Chem.MolToSmiles(Chem.MolFromSmiles(m)) for m in other if m not in hits)
    assert rest == ["CNOC"], rest              # N,O-dimethylhydroxylamine, and nothing else
    amide_other = dict(n=len(other), sulfonamides=len(hits))
    pd.DataFrame(crow).to_csv(OUT / "amplitude_class_slopes.csv", index=False)
    T2 = pd.DataFrame(tab); T2.to_csv(OUT / "amplitude_classes.csv", index=False)

    # ---- SNAr: base-family class profiles after each pair's slope is removed; three slope protocols
    key = "SNAr|1"; pairs, conds, Y = A[key]
    mols = np.array([p.split(" >> ")[0] for p in pairs]); live = np.nanmax(Y, axis=1) >= LIVE
    V = condition_variables(key, C, rxn[key], conds)
    mu, _, b = additive(Y); Rs = Y - slope_fit(Y, mu + b, np.arange(Y.shape[1]))[0]
    prof = {}
    for level in pd.unique(V["base"]):
        sub = Rs[:, V["base"].to_numpy() == level]; cnt = np.isfinite(sub).sum(1)
        prof[level] = np.where(cnt > 0, np.nansum(sub, 1) / (cnt + SHRINK), np.nan)
    U = pd.DataFrame(prof); U["molecule"] = mols; U = U[live]
    Pm = U.groupby("molecule").mean()
    fam = sorted(set(BASE_FAMILY.values()))
    Pf = pd.DataFrame({f: Pm[[v for v in Pm if BASE_FAMILY.get(v) == f]].mean(axis=1) for f in fam})
    Pf["cls"] = [nucleophile(x) for x in Pf.index]
    ca, cb = T2.set_index("screen").loc[key, ["class_a", "class_b"]]
    contrast = {f: float(Pf[Pf.cls == ca][f].mean() - Pf[Pf.cls == cb][f].mean()) for f in fam}
    # slope protocols (A2'): pair-half calibrated (Table 2); whole-array signature; joint row-scaled fit
    Qh, _ = class_slopes(Y, mols, nucleophile, PAIR_SEED, screen=key)
    s_whole = slope_fit(Y, mu + b, np.arange(Y.shape[1]))[2]
    s_joint = row_scaled_slopes(Y)
    proto = {}
    for name, s in (("pair_half", None), ("whole_array", s_whole), ("joint_row_scaled", s_joint)):
        if s is None:
            Q = Qh
        else:
            Q = pd.DataFrame(dict(molecule=mols, slope=s))[live].groupby("molecule")[["slope"]].mean()
            Q["cls"] = [nucleophile(x) for x in Q.index]
        med = Q.groupby("cls").slope.median()
        proto[name] = dict(a=float(med[ca]), b=float(med[cb]), ratio=float(med[ca] / med[cb]))
    pd.DataFrame([dict(protocol=k, **v) for k, v in proto.items()]).to_csv(OUT / "amplitude_snar_protocols.csv", index=False)
    Pf.to_csv(OUT / "amplitude_snar.csv")

    out = dict(median_mode_share=float(T.mode_share.median()), median_mode_abs_r=float(T.mode_abs_r.median()),
               median_slope_gain=float(T.slope_gain.median()), n_slope_gain_positive=int((T.slope_gain > 0).sum()),
               median_slope_gain_vs_mean_only=float(T.slope_gain_vs_mean_only.median()),
               median_condition_effect_gain=float(T.condition_effect_gain.median()),
               slope_gain=T.set_index("screen").slope_gain.to_dict(),
               median_profiles_r2=float(T.profiles_r2.median()), n_profiles_r2_positive=int((T.profiles_r2 > 0).sum()),
               snar_classes=dict(a=ca, b=cb), snar_base_contrast=contrast, snar_protocols=proto,
               amide_other_class=amide_other,
               n_snar_molecules=dict(a=int((Pf.cls == ca).sum()), b=int((Pf.cls == cb).sum())))
    save_json("amplitude.json", out)
    print(T2.round(3).to_string(index=False))
    print({k: v for k, v in out.items() if k != "slope_gain"})


def row_scaled_slopes(M, iters=5000, tol=1e-10):
    """Slopes of the joint fit y_ij = mu + r_i + s_i c_j (alternating least squares from the additive
    start, negative slopes set to zero, s rescaled to mean one) — the protocol of the partner analysis."""
    mask = np.isfinite(M); Y = np.where(mask, M, 0.0)
    mu0 = Y[mask].mean(); r = np.zeros(M.shape[0]); c = np.zeros(M.shape[1])
    for _ in range(500):
        r = ((Y - mu0 - c[None, :]) * mask).sum(1) / np.maximum(mask.sum(1), 1)
        c = ((Y - mu0 - r[:, None]) * mask).sum(0) / np.maximum(mask.sum(0), 1)
        mu0 = ((Y - r[:, None] - c[None, :]) * mask).sum() / mask.sum()
    mu = np.nanmean(M); s = np.ones(M.shape[0])
    for _ in range(iters):
        c_old = c.copy()
        for i in range(M.shape[0]):
            ok = mask[i]; x = c[ok]; y = M[i, ok] - mu
            if ok.sum() < 3 or x.var() < 1e-12:
                r[i] = y.mean(); s[i] = 1.0; continue
            s[i] = max(np.cov(x, y, bias=True)[0, 1] / x.var(), 0.0); r[i] = y.mean() - s[i] * x.mean()
        for j in range(M.shape[1]):
            ok = mask[:, j]; x = s[ok]; y = M[ok, j] - mu - r[ok]
            c[j] = (x @ y) / max(x @ x, 1e-9)
        c = c - np.average(c, weights=mask.sum(0)); sc = s.mean(); s = s / sc; c = c * sc
        if np.max(np.abs(c - c_old)) < tol:
            break
    return s


if __name__ == "__main__":
    main()
