"""Section 3: does molecular structure carry the condition preference?

Nature rebuild provenance: legacy `sealed` means development-seen but excluded from reported final
fits; `month`/`months`/`run_months` means archive-group labels. Neither establishes blind or temporal
validation. Original numerical output identifiers are retained for scientific equivalence.


  association   Spearman rank correlation, over pairs of substrate pairs, between structural similarity
                (ECFP4 Tanimoto averaged over the two partners) and the correlation of their condition-
                centred yield profiles; whole structure labels permuted 4,999 times; again without any
                two substrate pairs that share a molecule
  recommender   an equal blend of a 10-neighbour Tanimoto model and a rank-8 bilinear model predicts the
                interaction residual; its top three conditions against the three best training means
                (mean-ranked trio) and the greedy complementary trio, under three training designs:
                random pairs (molecules recur), size-matched random training, both partners withheld
  sealed        the same model trained on the analysis pairs and scored on the sealed structure-dissimilar
                pairs of eight arrays
Ties in every ranking go to the earlier training pair or condition (stable sort).
Writes out/structure_*.csv and out/structure.json.
"""
import sys
from boundaries import frozen_assignment, frozen_folds, frozen_fold_keys, validate_indices
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors, Crippen
from threadpoolctl import threadpool_limits
from common import (load_analysis, load_sealed, arrays, labels, matrix, top_k, greedy_trio,
                    whole_array_interval, bh, save_json, OUT, USABLE)

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
PARTITION_SEEDS = [20260920, 20261020, 20261120, 20261220, 20261320]
NPERM = 4999


def canonical(sp):
    return tuple(Chem.MolToSmiles(Chem.MolFromSmiles(v), isomericSmiles=True) if v else "" for v in sp.split(" >> "))


# ---------------------------------------------------------------- association
def corr_matrix(y, min_common=8):
    """Pearson r between every two rows over their commonly observed columns (at least min_common)."""
    o = np.isfinite(y).astype(float); x = np.nan_to_num(y); n = o @ o.T
    sx = x @ o.T; ss = (x * x) @ o.T; cov = x @ x.T - sx * sx.T / np.maximum(n, 1)
    var = np.maximum(ss - sx * sx / np.maximum(n, 1), 0); den = np.sqrt(var * var.T)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = cov / den
    out[(n < min_common) | (den < 1e-12)] = np.nan
    return np.clip(out, -1, 1)


def association(sim, cor, mask):
    i, j = np.where(np.triu(mask, 1))
    return float(stats.spearmanr(sim[i, j], cor[i, j]).statistic), i, j


def permutation_p(sim, prof, mask, rho, rng):
    """Whole structure labels permuted among substrate pairs, the eligibility mask held fixed."""
    i, j = np.where(np.triu(mask, 1))
    ranks = stats.rankdata(prof[i, j]); ranks = (ranks - ranks.mean()) / ranks.std()
    unique, ident = np.unique(sim, return_inverse=True); ident = ident.reshape(sim.shape)
    n, hits = len(sim), 0
    for _ in range(NPERM):
        perm = rng.permutation(n); ids = ident[perm[i], perm[j]]
        count = np.bincount(ids, minlength=len(unique)); mid = np.cumsum(count) - (count - 1) / 2
        r = mid[ids]; r = (r - r.mean()) / r.std()
        hits += abs(ranks @ r / len(r)) >= abs(rho) - 1e-12
    return (hits + 1) / (NPERM + 1)


def run_association(A):
    """The association between structural similarity and profile agreement, on the published centred
    response and on the preference left when the pair's amplitude is removed (preference.py). One
    eligibility mask and one similarity matrix serve all three, so the variants differ only in the
    profile they correlate."""
    import preference as PREF
    rows, bins = [], []
    for key, (pairs, conds, full) in A.items():
        cm = np.nanmean(full, axis=0); mols = [canonical(s) for s in pairs]
        keep = [i for i, m in enumerate(mols) if len(m) == 2 and all(m)]
        if len(keep) < 40:
            continue
        P, half, *_ = PREF.decompose(full, screen=key)
        y = full[keep]; mol = [mols[i] for i in keep]
        raw = corr_matrix(y); ctr = corr_matrix(y - cm)
        mask = np.isfinite(raw) & np.isfinite(ctr)
        if np.triu(mask, 1).sum() < 200:
            continue
        zero = np.nanmax(y, axis=1) - np.nanmin(y, axis=1) < 1e-12        # constant raw profile: no preference
        valid = np.flatnonzero(~zero)
        sel = np.asarray(keep)[valid]
        y = y[valid]; mol = [mol[i] for i in valid]
        ctr = ctr[np.ix_(valid, valid)]; mask = mask[np.ix_(valid, valid)]
        prof = {"centred": ctr,
                "preference": corr_matrix(P["p_oof"][sel]),
                "preference_c": corr_matrix(P["p_oof_centred"][sel])}
        n = len(y); sim = np.zeros((n, n))
        for role in (0, 1):
            fps = [GEN.GetFingerprint(Chem.MolFromSmiles(m[role])) for m in mol]
            for i_, fp in enumerate(fps):
                sim[i_] += .5 * np.asarray(DataStructs.BulkTanimotoSimilarity(fp, fps))
        no_shared = np.array([[not (set(a) & set(b)) for b in mol] for a in mol])
        rec = dict(screen=key, n_profiles=n)
        for name, C in prof.items():
            m_ = mask & np.isfinite(C)                       # the same eligibility, minus cells this
            r_, i, j = association(sim, C, m_)                # profile cannot define
            r_ns, ni, nj = association(sim, C, m_ & no_shared)
            p_ = permutation_p(sim, C, m_, r_, np.random.default_rng(20260920))
            suffix = "" if name == "centred" else "_" + name
            rec.update({f"rho{suffix}": r_, f"rho_no_shared{suffix}": r_ns, f"p{suffix}": p_,
                        f"n_pairs{suffix}": len(i), f"n_nonsharing{suffix}": len(ni)})
            if name == "preference_c":
                # the centring is estimated on the other half, so two pairs of one half have the same
                # quantity removed and two pairs of different halves do not
                same = half[sel][:, None] == half[sel][None, :]
                for cls, m2 in (("same_half", m_ & same), ("cross_half", m_ & ~same)):
                    r2_, i2, _ = association(sim, C, m2)
                    rec.update({f"rho_{cls}": r2_, f"n_pairs_{cls}": len(i2)})
            if name == "centred":
                for pop, a, b in (("all", i, j), ("no_shared", ni, nj)):
                    x = sim[a, b]; z = C[a, b]; q = pd.qcut(x, 5, labels=False, duplicates="drop")
                    for k in sorted(set(q)):
                        s = q == k
                        bins.append(dict(screen=key, population=pop, bin=int(k) + 1, n=int(s.sum()),
                                         similarity=float(x[s].mean()), profile_r=float(z[s].mean())))
        rows.append(rec)
        print(f"  association {key:28s} centred {rec['rho']:+.3f}  preference {rec['rho_preference']:+.3f}  "
              f"preference_c {rec['rho_preference_c']:+.3f}", flush=True)
    d = pd.DataFrame(rows)
    for suffix in ("", "_preference", "_preference_c"):
        d["q" + suffix] = bh(d["p" + suffix])
    return d, pd.DataFrame(bins)


# ---------------------------------------------------------------- recommender
def descriptors(m):
    if m is None:
        return [np.nan] * 10
    return [Descriptors.MolWt(m), Crippen.MolLogP(m), Descriptors.TPSA(m), Descriptors.NumHDonors(m),
            Descriptors.NumHAcceptors(m), Descriptors.NumRotatableBonds(m), Descriptors.NumAromaticRings(m),
            Descriptors.FractionCSP3(m), Descriptors.NumHeteroatoms(m), Descriptors.RingCount(m)]


def featurise(pairs):
    f1, f2, desc = [], [], []
    for sp in pairs:
        parts = [x for x in sp.split(" >> ") if x and x != "nan"][:2]
        mols = [Chem.MolFromSmiles(x) for x in parts] + [None] * (2 - len(parts))
        for store, m in zip((f1, f2), mols):
            bits = np.zeros(2048, dtype=np.int8)
            if m is not None:
                DataStructs.ConvertToNumpyArray(GEN.GetFingerprint(m), bits)
            store.append(bits)
        desc.append(descriptors(mols[0]) + descriptors(mols[1]))
    return np.array(f1, np.float32), np.array(f2, np.float32), np.nan_to_num(np.array(desc, np.float32))


def tanimoto(P, Q):
    s = P @ Q.T
    return s / np.clip(P.sum(1)[:, None] + Q.sum(1)[None, :] - s, 1e-9, None)


def feature_blocks(F1, F2, D, tr, te):
    """Training-fitted features of both partners: fingerprints compressed by truncated SVD and the
    standardised descriptors. One copy, used by the recommender and by the legibility comparison."""
    blocks = []
    for F in ([F1, F2] if F2.sum() > 0 else [F1]):
        k = min(48, len(tr) - 1, int((F[tr].sum(axis=0) > 0).sum()) - 1)
        if k < 2:
            continue
        svd = TruncatedSVD(n_components=k, random_state=0).fit(F[tr])
        X, Y = svd.transform(F[tr]), svd.transform(F[te])
        keep = X.std(axis=0) > 1e-3 * max(X.std(axis=0).max(), 1e-9)
        blocks.append((X[:, keep], Y[:, keep]))
    mu, sd = D[tr].mean(axis=0), D[tr].std(axis=0); keep = sd > 1e-8
    blocks.append(((D[tr][:, keep] - mu[keep]) / sd[keep], np.clip((D[te][:, keep] - mu[keep]) / sd[keep], -8, 8)))
    return np.hstack([b[0] for b in blocks]), np.hstack([b[1] for b in blocks])


def fit_predict(M, F1, F2, D, tr, te, target=None):
    """Compatibility adapter: materialize fitting data and outcome-free targets separately."""
    tr, te = validate_indices(tr, te)
    return fit_predict_separate(
        np.array(M[tr], copy=True),
        tuple(np.array(F[tr], copy=True) for F in (F1,F2,D)),
        tuple(np.array(F[te], copy=True) for F in (F1,F2,D)),
        None if target is None else np.array(target[tr], copy=True),
        has_second_role=F2.sum()>0)


def fit_predict_separate(train_y, train_features, target_features, train_response=None, has_second_role=True):
    """Fit/predict without accepting target outcomes. Scoring is performed by the caller."""
    F1,F2,D = (np.vstack([a,b]) for a,b in zip(train_features,target_features))
    tr = np.arange(len(train_y)); te = np.arange(len(train_y),len(F1))
    # Shape-only placeholder; target outcomes cannot reach any model-time statistic.
    M = np.vstack([train_y,np.full((len(te),train_y.shape[1]),np.nan)])
    target = None if train_response is None else np.vstack([train_response,np.full((len(te),train_response.shape[1]),np.nan)])
    return _fit_predict_training_only(M,F1,F2,D,tr,te,target,has_second_role)


def _fit_predict_training_only(M, F1, F2, D, tr, te, target=None, has_second_role=True):
    """Training rows tr only: condition means, grand mean and the predicted residual of te. The residual
    predicted is the interaction residual unless a target matrix is supplied, in which case its training
    rows are the quantity learned; the condition means always come from M, since they rank the panel."""
    tr, te = validate_indices(tr, te)
    has2 = has_second_role
    Mtr = np.array(M[tr], copy=True); colm = np.nanmean(Mtr, axis=0); gm = np.nanmean(Mtr)
    R = (np.nan_to_num(Mtr - np.nanmean(Mtr, axis=1, keepdims=True) - colm[None, :] + gm)
         if target is None else np.nan_to_num(target[tr]))
    Xtr, Xte = feature_blocks(F1, F2, D, tr, te)
    U, S, Vt = np.linalg.svd(R, full_matrices=False); r = min(8, len(S))
    bilinear = Ridge(alpha=100.0).fit(Xtr, U[:, :r] * S[:r]).predict(Xte) @ Vt[:r]
    sim = tanimoto(F1[te], F1[tr])
    if has2:
        sim = 0.5 * (sim + tanimoto(F2[te], F2[tr]))
    knn = np.zeros((len(te), M.shape[1]))
    for j in range(len(te)):
        nb = np.argsort(-sim[j], kind="stable")[:10]; w = np.clip(sim[j][nb], 0, None)
        knn[j] = (w @ R[nb]) / w.sum() if w.sum() > 0 else 0.0
    return colm, gm, np.clip(0.5 * (knn + bilinear), R.min(), R.max())


def score(y, mp, gp, cp):
    """Scorable only if every condition of the three panels was measured for this target."""
    union = np.unique(np.r_[mp, gp, cp])
    if not np.isfinite(y[union]).all():
        return dict(scorable=False)
    m, g, c = y[mp].max(), y[gp].max(), y[cp].max()
    return dict(scorable=True, model_best=m, mean_best=g, greedy_best=c, gain_mean=m - g, gain_greedy=m - c,
                rescue_greedy=bool(c < USABLE <= m), loss_greedy=bool(m < USABLE <= c),
                rescue_mean=bool(g < USABLE <= m), loss_mean=bool(m < USABLE <= g))


def assignment(values, seed):
    return frozen_assignment(values, seed)


def registered_kfold(n, *, screen):
    keys = sorted(frozen_fold_keys(f"kfold|{screen}|"), key=lambda key: int(key.rsplit('|',1)[1]))
    targets = []
    for key in keys:
        tr, te = frozen_folds(key)
        validate_indices(tr, te)
        if sorted(np.concatenate([tr,te]).tolist()) != list(range(n)):
            raise ValueError("Random-pair fold population mismatch")
        targets.extend(te.tolist())
        yield tr, te
    if sorted(targets) != list(range(n)):
        raise ValueError("Random-pair target population mismatch")


def run_designs(A, feats):
    rows, r2 = [], []
    for ai, (key, (pairs, conds, M)) in enumerate(A.items()):
        F1, F2, D = feats[key]; n = len(pairs); acc = dict(random=[0., 0.], grouped=[0., 0.], matched=[0., 0.])

        def record(design, seed, fold, train, te):
            if design != 'random':
                train, te = frozen_folds(f'design|{key}|{seed}|{fold}|{design}')
            colm, gm, pred = fit_predict(M, F1, F2, D, train, te)
            near = 0.5 * (tanimoto(F1[te], F1[train]) + tanimoto(F2[te], F2[train])) if F2.sum() > 0 \
                else tanimoto(F1[te], F1[train])
            rte = M[te] - np.nanmean(M[te], axis=1, keepdims=True) - colm[None, :] + gm; ok = np.isfinite(rte)
            acc[design][0] += float(np.sum((rte - pred)[ok] ** 2)); acc[design][1] += float(np.sum(rte[ok] ** 2))
            gp = top_k(colm); cp = greedy_trio(M[train])
            for j, i in enumerate(te):
                mp = top_k(colm + pred[j])
                rows.append(dict(design=design, seed=seed, fold=fold, screen=key, subpair=pairs[i], n_train=len(train),
                                 nearest=float(near[j].max()),
                                 model_panel="|".join(conds[c] for c in mp), mean_panel="|".join(conds[c] for c in gp),
                                 greedy_panel="|".join(conds[c] for c in cp), **score(M[i], mp, gp, cp)))
        for fold, (tr, te) in enumerate(registered_kfold(n, screen=key)):
            record("random", 0, str(fold), tr, te)
        r2.append(dict(design="random", screen=key, sse=acc["random"][0], sst=acc["random"][1]))
        cp_ = [canonical(s) for s in pairs]
        for seed in PARTITION_SEEDS:
            acc["grouped"] = [0., 0.]; acc["matched"] = [0., 0.]
            id1, map1 = assignment([v[0] for v in cp_], seed); id2, map2 = assignment([v[1] for v in cp_], seed + 1)
            for a in sorted(set(id1)):
                for b in sorted(set(id2)):
                    te = np.flatnonzero((id1 == a) & (id2 == b))
                    if not len(te):
                        continue
                    held = {v for v, z in map1.items() if z == a} | {v for v, z in map2.items() if z == b}
                    tr = np.array([i for i, pr in enumerate(cp_) if not (set(pr) & held)], int)
                    assert not {v for i in tr for v in cp_[i]} & {v for i in te for v in cp_[i]}
                    rng = np.random.default_rng(seed + 100000 + ai * 100 + a * 10 + b)
                    matched = np.sort(rng.choice(np.setdiff1d(np.arange(n), te), len(tr), replace=False))
                    record("grouped", seed, f"{a}{b}", tr, te)
                    record("matched", seed, f"{a}{b}", matched, te)
            for design in ("grouped", "matched"):
                r2.append(dict(design=design, seed=seed, screen=key, sse=acc[design][0], sst=acc[design][1]))
        print(f"  designs {key:28s} done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(r2)


def run_sealed(A, S):
    """Model trained on every analysis pair of the array, scored on its sealed pairs."""
    rows = []
    for key, g in S.groupby("screen", sort=True):
        pairs_a, conds, Ma = A[key]
        pairs_s, conds_s, Ms = matrix(g)
        Ms = pd.DataFrame(Ms, index=pairs_s, columns=conds_s).reindex(columns=conds).to_numpy(float)
        M = np.vstack([Ma, Ms]); pairs = list(pairs_a) + list(pairs_s)
        F1, F2, D = featurise(pairs)
        tr = np.arange(len(pairs_a)); te = np.arange(len(pairs_a), len(pairs))
        colm, gm, pred = fit_predict(M, F1, F2, D, tr, te)
        gp = top_k(colm); cp = greedy_trio(Ma)
        for j, i in enumerate(te):
            mp = top_k(colm + pred[j])
            rows.append(dict(design="sealed", screen=key, subpair=pairs[i], model_panel="|".join(conds[c] for c in mp),
                             **score(M[i], mp, gp, cp)))
    return pd.DataFrame(rows)


def summarise(z):
    z = z[z.scorable]; out = dict(n=int(len(z)), n_arrays=int(z.screen.nunique()))
    for comp in ("mean", "greedy"):
        m, lo, hi = whole_array_interval(z, "gain_" + comp)
        out[comp] = [100 * m, 100 * lo, 100 * hi]
        out["rescue_" + comp] = 100 * float(z["rescue_" + comp].mean()); out["loss_" + comp] = 100 * float(z["loss_" + comp].mean())
        out["rescue_" + comp + "_n"] = int(z["rescue_" + comp].sum()); out["loss_" + comp + "_n"] = int(z["loss_" + comp].sum())
    return out


def main():
    d = load_analysis(); A = arrays(d)
    part = [a for a in sys.argv[1:] if a in ("association", "designs", "sealed")] or ["association", "designs", "sealed"]
    res = {}
    if "association" in part:
        assoc, bins = run_association(A)
        assoc.to_csv(OUT / "structure_association.csv", index=False); bins.to_csv(OUT / "structure_bins.csv", index=False)
        res["association"] = dict(n_arrays=int(len(assoc)), median_rho=float(assoc.rho.median()),
                                  n_positive=int((assoc.rho > 0).sum()), n_q05=int((assoc.q < .05).sum()),
                                  median_rho_no_shared=float(assoc.rho_no_shared.median()),
                                  n_positive_no_shared=int((assoc.rho_no_shared > 0).sum()))
        for v in ("preference", "preference_c"):
            res["association"][v] = dict(
                median_rho=float(assoc["rho_" + v].median()), n_positive=int((assoc["rho_" + v] > 0).sum()),
                n_q05=int((assoc["q_" + v] < .05).sum()),
                median_rho_no_shared=float(assoc["rho_no_shared_" + v].median()),
                n_positive_no_shared=int((assoc["rho_no_shared_" + v] > 0).sum()))
    if "designs" in part:
        feats = {k: featurise(v[0]) for k, v in A.items()}
        cases, r2 = run_designs(A, feats)
        cases.to_csv(OUT / "structure_cases.csv", index=False); r2.to_csv(OUT / "structure_r2.csv", index=False)
        pr = r2.groupby(["design", "screen"])[["sse", "sst"]].sum(); pr["r2"] = 1 - pr.sse / pr.sst
        res["designs"] = {des: dict(summarise(cases[cases.design == des]), n_targets=int((cases.design == des).sum()),
                                    median_r2=float(pr.loc[des].r2.median()))
                          for des in ("random", "matched", "grouped")}
        gr = cases[(cases.design == "grouped") & cases.scorable].copy()
        gr["sextile"] = pd.qcut(gr.nearest, 6, labels=False, duplicates="drop")
        binned = gr.groupby("sextile").agg(n=("nearest", "size"), nearest=("nearest", "mean"),
                                           gain=("gain_greedy", "mean"))
        ci = {k: whole_array_interval(g_, "gain_greedy") for k, g_ in gr.groupby("sextile")}
        res["designs"]["neighbour"] = dict(median=float(gr.nearest.median()), low=float(gr.nearest.min()),
                                           high=float(gr.nearest.max()), n_bins=int(len(binned)),
                                           max_abs_bin_gain=float(100 * binned.gain.abs().max()),
                                           bins=[dict(nearest=float(r.nearest), gain=float(100 * r.gain), n=int(r.n),
                                                      lo=float(100 * ci[r.Index][1]), hi=float(100 * ci[r.Index][2]))
                                                 for r in binned.itertuples()])
        g, m = pr.loc["grouped"].r2, pr.loc["matched"].r2
        res["designs"]["n_grouped_r2_below_matched"] = int((g < m.reindex(g.index)).sum())
    if "sealed" in part:
        sealed = run_sealed(A, load_sealed())
        sealed.to_csv(OUT / "structure_sealed.csv", index=False)
        res["sealed"] = dict(summarise(sealed), n_targets=int(len(sealed)))
    old = {}
    try:
        from common import load_json
        old = load_json("structure.json")
    except FileNotFoundError:
        pass
    old.update(res); save_json("structure.json", old)
    print(res)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
