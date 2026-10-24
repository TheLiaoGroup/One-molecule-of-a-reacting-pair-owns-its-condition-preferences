"""Where a pair's condition preference lives: building blocks, the combination, or measurement noise.

Fifteen arrays in which building blocks recur (history task). A pair is product-forming when it reaches
FORMS yield under at least one condition; 0.05 and 0.20 are reported as sensitivity.
  agreement   additive-residual profile correlations for pairs sharing only partner 1, only partner 2 or
              neither, refitted on product-forming pairs (history.pair_agreement)
  shares      condition pattern = each pair's residual after an OLS intercept and slope on the condition
              means; share of a partner = mean product of pattern profiles over pairs sharing only that
              partner (averaged over shared conditions) / mean pattern variance; interval from 2,000 draws
              resampling arrays and, within each array, the molecules of each role (draws with no
              qualifying comparison in an array are redrawn and counted); noise share from the Methods
              noise model, scaled so that it reproduces the released Hiyama C43/C46 duplicate
  prediction  held-out R2 of jointly backfitted partner profiles on the history folds, over all targets
              and over product-forming targets (all-target values must equal history_arrays.csv)
  selection   contrasts of history_cv.csv over all and product-forming targets; share of the gap between
              the complementary trio and the pair's best measured condition that the history panel closes
  depth       single-role held-out R2 when the target molecule's training history is cut to k pairings,
              on targets whose molecule has at least K_MAX training pairings. Rule fixed before running:
              "accumulates" only if pooled R2 rises at every step from k = 1 to K_MAX and the molecule-
              resampled interval of R2(K_MAX) - R2(1) lies above zero.
The same share estimator, frozen earlier for 04_ANALYSES/09, is checked against this one in the tests.
Writes anatomy.json, anatomy_arrays.csv and anatomy_depth.csv.
"""
import numpy as np
import pandas as pd
from common import load_analysis, five_folds, stream, owner_interval, save_json, OUT, USABLE, ROOT
from history import partner_arrays, additive_residual, pair_agreement, train_residual, joint_profiles, ROLE, SHRINK

FORMS = 0.10
SENSITIVITY = (0.05, 0.20)
NBOOT, SEED = 2000, 20261003
K_LIST, K_MAX, K_DRAWS = (1, 2, 3, 4, 6), 6, 5


# ---------------------------------------------------------------- condition pattern and shares
def pattern(M):
    s = np.nanmean(M, 0); E = np.full_like(M, np.nan); H = np.full_like(M, np.nan); fit = np.full_like(M, np.nan)
    for i in range(len(M)):
        o = np.isfinite(M[i]) & np.isfinite(s)
        X = np.c_[np.ones(o.sum()), s[o]]; XtXi = np.linalg.pinv(X.T @ X)
        coef = XtXi @ X.T @ M[i, o]
        fit[i, o] = X @ coef; E[i, o] = M[i, o] - fit[i, o]; H[i, o] = np.einsum("ij,jk,ik->i", X, XtXi, X)
    return E, H, fit


class Shares:
    def __init__(self, E, a, b):
        same1, same2 = a[:, None] == a[None, :], b[:, None] == b[None, :]
        R = {"1": same1 & ~same2, "2": same2 & ~same1, "N": ~same1 & ~same2}
        Mk = np.isfinite(E).astype(float); E0 = np.nan_to_num(E); den = Mk @ Mk.T
        C = np.divide(E0 @ E0.T, den, out=np.zeros_like(den), where=den > 0)
        self.CR = {k: C * r for k, r in R.items()}; self.MR = {k: (den > 0) * r.astype(float) for k, r in R.items()}
        self.var = np.nansum(E ** 2, 1) / Mk.sum(1)

    def compute(self, w):
        V = np.sum(w * self.var) / w.sum()
        out = {}
        for k in self.CR:
            d = w @ self.MR[k] @ w
            out[k] = np.nan if d == 0 else (w @ self.CR[k] @ w) / d / V
        return out


def ml_components(E, a, b):
    """Gaussian ML variance components of the same patterns: an estimator that shares no algebra with
    the method of moments, so agreement between them is evidence and not arithmetic."""
    from scipy.optimize import minimize
    R = np.nan_to_num(E); n, m = R.shape; reps = max(m - 2, 1)
    Z1 = (a[:, None] == np.unique(a)[None, :]).astype(float)
    Z2 = (b[:, None] == np.unique(b)[None, :]).astype(float)
    K1, K2, S = Z1 @ Z1.T, Z2 @ Z2.T, R @ R.T

    def nll(t):
        v1, v2, ve = np.exp(t); Sig = v1 * K1 + v2 * K2 + ve * np.eye(n)
        try:
            L = np.linalg.cholesky(Sig)
        except np.linalg.LinAlgError:
            return 1e30
        Li = np.linalg.solve(L, np.eye(n))
        return .5 * (reps * 2 * np.log(np.diag(L)).sum() + np.sum((Li.T @ Li) * S))

    tot = np.trace(S) / (n * reps); best = None
    for f in ([.3, .3, .4], [.1, .1, .8], [.45, .45, .1], [.05, .5, .45], [.5, .05, .45]):
        r = minimize(nll, np.log(np.array(f) * tot + 1e-12), method="Nelder-Mead",
                     options={"maxiter": 6000, "fatol": 1e-7})
        if best is None or r.fun < best.fun:
            best = r
    v1, v2, ve = np.exp(best.x); d = v1 + v2 + ve
    return v1 / d, v2 / d


def molecule_weights(a, b, rng):
    ua, ia = np.unique(a, return_inverse=True); ub, ib = np.unique(b, return_inverse=True)
    ca = np.bincount(rng.integers(0, len(ua), len(ua)), minlength=len(ua))
    cb = np.bincount(rng.integers(0, len(ub), len(ub)), minlength=len(ub))
    return (ca[ia] * cb[ib]).astype(float)


def noise_variance(M, fit, H, k):
    sig2 = k * np.maximum(0.07 * np.clip(fit, 0, 1), 0.012) ** 2
    return float(np.nanmean(np.nansum(sig2 * (1 - H), 1) / np.isfinite(M).sum(1)))


def hiyama_noise_scale():
    """Scale on the noise-model variance that reproduces the released C43/C46 duplicate."""
    C = pd.read_csv(ROOT / "05_DATA/Analysis/01_CANONICAL/measurements.csv.gz",
                    usecols=["screen", "pair_id", "condition_id", "yield", "partition", "qc_missing_yield"])
    C = C[(C.screen == "Hiyama|1") & (C.partition == "analysis") & ~C.qc_missing_yield]
    P = C[C.condition_id != "C46"].pivot_table(index="pair_id", columns="condition_id", values="yield")
    d46 = C[C.condition_id == "C46"].set_index("pair_id")["yield"].reindex(P.index).to_numpy()
    M = P.to_numpy(float); E, H, fit = pattern(M); j = list(P.columns).index("C43")
    ok = np.isfinite(d46) & np.isfinite(M[:, j])
    if ok.sum() < 50:
        raise ValueError("Hiyama duplicate not found")
    e43 = E[ok, j]; e46 = d46[ok] - fit[ok, j]       # the duplicate scored against the same pair-specific line
    r_emp = np.mean(e43 * e46) / np.mean((e43 ** 2 + e46 ** 2) / 2)
    n_model = np.mean(np.maximum(0.07 * np.clip(fit[ok, j], 0, 1), 0.012) ** 2 * (1 - H[ok, j]))
    return float(r_emp), float((1 - r_emp) * np.mean((e43 ** 2 + e46 ** 2) / 2) / n_model)


# ---------------------------------------------------------------- held-out prediction
def heldout_r2(A, folds, keep, backfit=10):
    """history.heldout_r2 for the joint profiles, scored only on targets with keep[p]."""
    M, ia, ib, na, nb = A["M"], A["ia"], A["ib"], len(A["ua"]), len(A["ub"])
    num = den = 0.0
    for f in folds:
        te = set(f.tolist()); tr = np.array([p for p in range(len(M)) if p not in te])
        R = train_residual(M, tr)[0]
        U, W = joint_profiles(R, ia, ib, na, nb, tr, rounds=backfit)
        for p in te:
            if not keep[p]:
                continue
            ok = np.isfinite(R[p]); y = R[p, ok]
            num += np.sum((y - U[ia[p], ok] - W[ib[p], ok]) ** 2); den += np.sum(y ** 2)
    return 1 - num / den if den > 0 else np.nan


def depth_records(A, folds, keep, rng):
    """Single-role held-out loss with the target molecule's history cut to k training pairings."""
    M = A["M"]; rows = []
    for f in folds:
        te = set(f.tolist()); tr = np.array([p for p in range(len(M)) if p not in te])
        R = train_residual(M, tr)[0]
        for role, idx in (("1", A["ia"]), ("2", A["ib"])):
            for p in te:
                if not keep[p]:
                    continue
                hist = tr[idx[tr] == idx[p]]
                if len(hist) < K_MAX:
                    continue
                ok = np.isfinite(R[p]); y = R[p, ok]
                rec = dict(screen=A["key"], role=role, pair=int(p), molecule=int(idx[p]), sst=float(np.sum(y ** 2)))
                for k in K_LIST:
                    sse = 0.0
                    for _ in range(K_DRAWS):
                        sub = rng.choice(hist, k, replace=False); Z = R[sub][:, ok]
                        prof = np.nansum(Z, 0) / (np.isfinite(Z).sum(0) + SHRINK)
                        sse += np.sum((y - prof) ** 2) / K_DRAWS
                    rec[f"sse_{k}"] = float(sse)
                rows.append(rec)
    return rows


# ---------------------------------------------------------------- partner composition
PARTNER_CLASSES = [  # first match wins; the reacting group of carbon nucleophiles is matched first
    ("boronic acid", "B(O)O"), ("Grignard reagent", "[Mg]"), ("organosilane", "[Si]"), ("terminal alkyne", "[CH]#C"),
    ("alkene", "C=C"),
    ("amide, sulfonamide or carbamate", "[NX3;H1,H2][CX3,SX4](=O)"), ("N-H azole or heteroarene", "[nH]"),
    ("primary aniline", "[NX3;H2]c"), ("secondary aniline", "[NX3;H1](c)[#6]"),
    ("primary aliphatic amine", "[NX3;H2][CX4]"), ("secondary aliphatic amine", "[NX3;H1]([CX4])[CX4]"),
    ("alcohol or phenol", "[OX2H]")]
LEAVING = [("I", "cI"), ("Br", "cBr"), ("Cl", "cCl"), ("F", "cF"), ("sulfonate", "cOS(=O)(=O)")]


def composition(A, er):
    """Classes of the partner molecules and leaving groups of the aryl electrophiles in one array."""
    from rdkit import Chem
    pats = [(n, Chem.MolFromSmarts(s)) for n, s in PARTNER_CLASSES]; lg = [(n, Chem.MolFromSmarts(s)) for n, s in LEAVING]
    first = lambda m, ps: next((n for n, p in ps if m.HasSubstructMatch(p)), "other")
    elec, part = (A["ua"], A["ub"]) if er == "1" else (A["ub"], A["ua"])
    rows = [dict(screen=A["key"], side="partner", category=first(Chem.MolFromSmiles(s), pats)) for s in part]
    rows += [dict(screen=A["key"], side="electrophile", category=first(Chem.MolFromSmiles(s), lg)) for s in elec]
    return rows


# ---------------------------------------------------------------- main
def electrophile_role(key):
    r1, r2 = ROLE[key][0][0], ROLE[key][1][0]
    h1, h2 = "halide" in r1, "halide" in r2
    return "1" if h1 and not h2 else ("2" if h2 and not h1 else None)


def class_pattern(E, mols, labels, rng, draws=4999):
    """How much of a condition pattern a substrate class explains, against labels shuffled between molecules.

    The additive share asks what one molecule carries; this asks what its class carries, which is the quantity
    a practical guide organises by. The classifier is amplitude.py's, so there is only one definition of a class.
    """
    keep = np.array([l is not None for l in labels])
    if keep.sum() < 8:
        return None
    sub, m, lab = E[keep], np.asarray(mols)[keep], np.asarray([l for l in labels if l is not None])
    if len(set(lab)) < 2:
        return None

    def explained(z):
        gm = np.nanmean(sub, 0); fit = np.zeros_like(sub)
        for c in np.unique(z):
            fit[z == c] = np.nanmean(sub[z == c], 0)
        return float(1 - np.nansum((sub - fit) ** 2) / np.nansum((sub - gm) ** 2))

    obs = explained(lab)
    um = list(dict.fromkeys(m)); by = {x: lab[m == x][0] for x in um}
    null = np.empty(draws)
    for t in range(draws):
        perm = dict(zip(um, rng.permutation([by[x] for x in um])))
        null[t] = explained(np.array([perm[x] for x in m]))
    return dict(explained=obs, null_median=float(np.median(null)), null_p95=float(np.percentile(null, 95)),
                p=float((1 + (null >= obs).sum()) / (1 + draws)), pairs=int(keep.sum()),
                molecules={c: int(sum(v == c for v in by.values())) for c in sorted(set(lab))})


CLASSED = None   # filled from amplitude.CLASS_ARRAYS at run time, so there is one list of classed arrays


def main():
    global CLASSED
    import amplitude
    CLASSED = dict(amplitude.CLASS_ARRAYS)
    AR = partner_arrays(load_analysis())
    rng = np.random.default_rng(SEED)
    r_emp, k_noise = hiyama_noise_scale()
    rows, share_objs, depth, class_effects = [], {}, [], []
    for key, A in AR.items():
        M = A["M"]; mx = np.nanmax(M, 1); act = mx >= FORMS
        Ra = additive_residual(M[act])[0]
        g1, g2, g0 = pair_agreement(Ra, A["a"][act], A["b"][act])
        E, H, fit = pattern(M[act])
        sh = Shares(E, A["a"][act], A["b"][act]); s = sh.compute(np.ones(act.sum()))
        ml1, ml2 = ml_components(E, A["a"][act], A["b"][act])
        noise = noise_variance(M[act], fit, H, k_noise) / sh.var.mean()
        share_objs[key] = (sh, A["a"][act], A["b"][act])
        if key in CLASSED:
            import amplitude
            raw = [CLASSED[key](x) for x in A["a"][act]]
            counts = pd.Series(raw).value_counts()
            top = [c for c in counts.index if c != "other" and counts[c] >= 3][:2]
            lab = [x if x in top else None for x in raw]
            ce = class_pattern(E, A["a"][act], lab, np.random.default_rng(20261008))
            if ce:
                ce.update(screen=key, classified_role=1,
                          classified_is_owner=bool(s["1"] >= s["2"]),
                          additive_share=float(s["1"]), classes=top)
                class_effects.append(ce)
        folds = five_folds(len(M), stream(key, "cv"))
        er = electrophile_role(key)
        owner = "1" if s["1"] >= s["2"] else "2"
        ml_owner = "1" if ml1 >= ml2 else "2"
        row = dict(screen=key, pairs=len(M), product_forming=int(act.sum()),
                   owner_role=owner, owner=ROLE[key][int(owner) - 1][0],
                   owner_share=s[owner], partner_share=s["2" if owner == "1" else "1"],
                   owner_fraction=(s[owner] / (s["1"] + s["2"]) if s["1"] + s["2"] > 0 else np.nan),
                   ml_owner_role=ml_owner, ml_owner=ROLE[key][int(ml_owner) - 1][0],
                   ml_owner_share=max(ml1, ml2), ml_partner_share=min(ml1, ml2),
                   r_share1=g1, r_share2=g2, r_neither=g0,
                   share1=s["1"], share2=s["2"], share_neither=s["N"], share=s["1"] + s["2"], noise=noise,
                   combination=1 - s["1"] - s["2"] - noise,
                   ml_share1=ml1, ml_share2=ml2, ml_share=ml1 + ml2,
                   electrophile_role=er, electrophile_carries=(None if er is None else bool(s[er] > s["2" if er == "1" else "1"])),
                   electrophile_carries_ml=(None if er is None else bool((ml1 if er == "1" else ml2) > (ml2 if er == "1" else ml1))),
                   r2_all=heldout_r2(A, folds, np.ones(len(M), bool)), r2_product_forming=heldout_r2(A, folds, act))
        for t in SENSITIVITY:
            keep = mx >= t; Et, *_ = pattern(M[keep]); st = Shares(Et, A["a"][keep], A["b"][keep]).compute(np.ones(keep.sum()))
            row[f"share_at_{t:.2f}"] = st["1"] + st["2"]
        rows.append(row)
        depth += depth_records(A, folds, act, rng)
        print(f"  {key:28s} share {row['share']:.2f} noise {noise:.3f} R2 {row['r2_all']:.3f}/{row['r2_product_forming']:.3f}", flush=True)
    T = pd.DataFrame(rows)
    ref = pd.read_csv(OUT / "history_arrays.csv").set_index("screen").r2_both
    if not np.allclose(T.set_index("screen").r2_all, ref.loc[T.screen], atol=1e-10):
        raise ValueError("all-target held-out R2 differs from history_arrays.csv")
    T.to_csv(OUT / "anatomy_arrays.csv", index=False)

    # hierarchical bootstrap of the median building-block share
    keys = list(share_objs); boot = []; redraws = 0
    for _ in range(NBOOT):
        vals = []
        for key in rng.choice(keys, len(keys)):
            sh, a, b = share_objs[key]
            for _try in range(50):
                r = sh.compute(molecule_weights(a, b, rng))
                if np.isfinite(r["1"]) and np.isfinite(r["2"]):
                    break
                redraws += 1
            vals.append(r["1"] + r["2"])
        boot.append(np.median(vals))

    # selection over all and product-forming targets
    cv = pd.read_csv(OUT / "history_cv.csv")
    best = {(k, p): (np.nanmax(A["M"][p])) for k, A in AR.items() for p in range(len(A["M"]))}
    cv["best"] = [best[(k, p)] for k, p in zip(cv.screen, cv.pair)]
    cv = cv.assign(H_C=cv.H - cv.C, N_C=cv.N - cv.C, gap=cv.best - cv.C,
                   rescue=(cv.C < USABLE) & (cv.H >= USABLE), loss=(cv.H < USABLE) & (cv.C >= USABLE))
    per = cv.groupby(["screen", "pair", "owner"], as_index=False)[["H_C", "N_C", "gap", "rescue", "loss", "best"]].mean()
    sel = {}
    for name, z in (("all", per), ("product_forming", per[per.best >= FORMS])):
        sel[name] = dict(n=int(len(z)), fitted=[100 * v for v in owner_interval(z, "H_C", "owner")],
                         raw=[100 * v for v in owner_interval(z, "N_C", "owner")],
                         rescue_pct=100 * float(z.rescue.mean()), loss_pct=100 * float(z.loss.mean()),
                         gap_mean=100 * float(z.gap.mean()), gap_closed=float(z.H_C.sum() / z.gap.sum()))
    sel["records"] = dict(rescues=int(cv.rescue.sum()), losses=int(cv.loss.sum()), records=int(len(cv)))

    # history depth
    D = pd.DataFrame(depth); D.to_csv(OUT / "anatomy_depth.csv", index=False)
    pooled = {k: float(1 - D[f"sse_{k}"].sum() / D.sst.sum()) for k in K_LIST}
    gain = (D[f"sse_1"] - D[f"sse_{K_MAX}"]).to_numpy(); sst = D.sst.to_numpy()
    grp = D.groupby(["screen", "role", "molecule"]).ngroup().to_numpy(); ng = grp.max() + 1
    gs, ss = np.bincount(grp, gain, ng), np.bincount(grp, sst, ng)
    by_screen = pd.Series(D.groupby(["screen", "role", "molecule"]).screen.first().to_numpy())
    db = []
    for _ in range(NBOOT):
        pick = np.concatenate([rng.choice(np.flatnonzero(by_screen == s), (by_screen == s).sum()) for s in by_screen.unique()])
        db.append(gs[pick].sum() / ss[pick].sum())
    rises = all(pooled[K_LIST[i + 1]] > pooled[K_LIST[i]] for i in range(len(K_LIST) - 1))
    depth_out = dict(targets=int(len(D)), molecules=int(ng), pooled_r2=pooled,
                     difference=[float(gain.sum() / sst.sum()), float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))],
                     rises_every_step=bool(rises), accumulates=bool(rises and np.percentile(db, 2.5) > 0))

    comp = pd.DataFrame([r for key, A in AR.items() if electrophile_role(key) for r in composition(A, electrophile_role(key))])
    comp = comp.groupby(["screen", "side", "category"]).size().rename("molecules").reset_index()
    comp.to_csv(OUT / "anatomy_composition.csv", index=False)
    top = comp[comp.side == "partner"].sort_values("molecules", ascending=False).groupby("screen").head(1).set_index("screen")
    tot = comp[comp.side == "partner"].groupby("screen").molecules.sum()
    ec = T.dropna(subset=["electrophile_role"])
    out = dict(forms=FORMS, n_arrays=int(len(T)), product_forming_pairs=int(T.product_forming.sum()), pairs=int(T.pairs.sum()),
               agreement=dict(share1=float(T.r_share1.median()), share2=float(T.r_share2.median()), neither=float(T.r_neither.median()),
                              arrays_either_above_neither=int(((T.r_share1 > T.r_neither) | (T.r_share2 > T.r_neither)).sum())),
               share=dict(median=float(T.share.median()), ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                          redrawn=int(redraws), range=[float(T.share.min()), float(T.share.max())],
                          neither_median=float(T.share_neither.median()),
                          noise_median=float(T.noise.median()), noise_range=[float(T.noise.min()), float(T.noise.max())],
                          combination_median=float(T.combination.median()),
                          sensitivity={f"{t:.2f}": float(T[f"share_at_{t:.2f}"].median()) for t in SENSITIVITY},
                          hiyama_duplicate_reliability=r_emp, noise_scale=k_noise),
               substrate_class=class_effects,
               ownership=dict(arrays=int(len(T)), same_owner_both_estimators=int((T.owner_role == T.ml_owner_role).sum()),
                              owner_at_least_twice_partner=int((T.owner_share >= 2 * T.partner_share.clip(lower=0)).sum()),
                              median_owner_fraction=float(T.owner_fraction.median()),
                              owners={r.screen: r.owner for r in T.itertuples()}),
               electrophile=dict(arrays=int(len(ec)), carries=int(ec.electrophile_carries.sum()),
                                 carries_ml=int(ec.electrophile_carries_ml.sum()),
                                 ml_median_share=float(T.ml_share.median()),
                                 dominant_partner_class={k: dict(category=top.loc[k, "category"], molecules=int(top.loc[k, "molecules"]),
                                                                 of=int(tot.loc[k])) for k in top.index}),
               prediction=dict(median_all=float(T.r2_all.median()), median_product_forming=float(T.r2_product_forming.median())),
               selection=sel, depth=depth_out)
    save_json("anatomy.json", out)
    print({k: v for k, v in out.items() if k not in ("selection", "depth")})
    print("selection", sel); print("depth", depth_out)


if __name__ == "__main__":
    main()
