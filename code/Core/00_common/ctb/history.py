"""Section 4: does a building block's own screening history carry its condition preference?

Nature rebuild provenance: legacy `sealed` means development-seen but excluded from reported final
fits; `month`/`months`/`run_months` means archive-group labels. Neither establishes blind or temporal
validation. Original numerical output identifiers are retained for scientific equivalence.


Fifteen arrays in which building blocks recur with different partners.
  agreement     r of interaction residuals between two substrate pairs sharing partner 1, partner 2, neither
  held-out R2   profile of a partner = shrunken mean interaction residual of its training pairs under each
                condition; five folds over pairs; partner 1, partner 2, both (backfitted); and the same on
                the residual left after each pair's response slope is removed (amplitude removed)
  transfer      for a held-out pair, rank conditions by the training mean plus BOTH partners'
                shrunken profiles and take the top three (history panel); a partner with no
                record contributes zero; compare with the three best training means (mean-ranked trio),
                the greedy complementary trio, and the three conditions under which the owner itself gave
                the highest mean raw yield (model-free); three draws of five folds
  run months    the same with every archive label (run month) held out in turn
  sealed        the sealed pairs of six arrays, profiles from all analysis pairs
A target is scored only when every condition of the compared panels was measured for it.
Writes out/history_*.csv and out/history.json.
"""
import numpy as np
import pandas as pd
from common import (load_analysis, load_sealed, load_archive, top_k, greedy_trio, stream, five_folds,
                    owner_interval, save_json, OUT, USABLE, UNIQUE_PAIR_ARRAYS)

SHRINK = 4.0
NREP = 3
PROSPECTIVE_SEALED = ("Buchwald-Hartwig|2", "Kumada|1")   # sealed pairs set aside for a bench test; not scored
# chemical role of partner 1 (sub_1) and partner 2 (sub_2), with a SMARTS every member should carry
ROLE = {
    "Acylation|1": (("N–H partner", "[#7;!H0]"), ("carboxylic acid", "C(=O)[OX2H1]")),
    "Buchwald-Hartwig|1": (("N–H partner", "[#7;!H0]"), ("aryl (pseudo)halide", "[c,n][Cl,Br,I,$(OS(=O)(=O)C(F)(F)F)]")),
    "Buchwald-Hartwig|2": (("N–H partner", "[#7;!H0]"), ("aryl halide", "c[Cl,Br,I]")),
    "C-O competitive reactions|1": (("phenol or alcohol", "[OX2H1]"), ("aryl halide", "c[Cl,Br,I]")),
    "Chan-Lam|1": (("N–H partner", "[#7;!H0]"), ("boronic acid", "B(O)O")),
    "Esterification|1": (("alcohol", "[OX2H1]"), ("carboxylic acid", "C(=O)[OX2H1]")),
    "Grignard|1": (("Grignard reagent", "[Mg]"), ("carbonyl compound", "[CX3]=O")),
    "Heck|1": (("alkene", "C=C"), ("aryl halide", "c[Cl,Br,I]")),
    "Hiyama|1": (("organosilane", "[Si]"), ("aryl halide", "c[Cl,Br,I]")),
    "Kumada|1": (("aryl or acyl halide", "[$(c[Cl,Br,I]),$([CX3](=O)[Cl,Br,I])]"), ("Grignard reagent", "[Mg]")),
    "Reductive-amination|1": (("carbonyl compound", "[CX3]=O"), ("amine", "[NX3;H1,H2]")),
    "SNAr|1": (("N–H nucleophile", "[#7;!H0]"), ("aryl halide", "[c,n][F,Cl,Br,I]")),
    "Sonogashira|1": (("alkyne", "C#C"), ("aryl halide", "c[Cl,Br,I]")),
    "Suzuki|1": (("aryl halide", "c[Cl,Br,I]"), ("boronic acid", "B(O)O")),
    "Ullmann|1": (("phenol", "c[OX2H1]"), ("aryl halide", "c[Cl,Br,I]")),
}


def check_roles(AR):
    """Every partner of every array must carry the SMARTS of its role; writes out/history_roles.csv."""
    from rdkit import Chem
    rows = []
    for key, A in AR.items():
        rec = dict(screen=key)
        for side, names in ((1, A["ua"]), (2, A["ub"])):
            role, smarts = ROLE[key][side - 1]; pat = Chem.MolFromSmarts(smarts)
            miss = [x for x in names if not Chem.MolFromSmiles(x).HasSubstructMatch(pat)]
            assert not miss, (key, side, role, miss[:5])
            rec[f"role_{side}"] = role; rec[f"n_{side}"] = len(names)
        rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT / "history_roles.csv", index=False)


def partner_arrays(d):
    out = {}
    for key, g in d.groupby("screen"):
        if key in UNIQUE_PAIR_ARRAYS:
            continue
        P = g.pivot(index="subpair", columns="condition_id", values="yield")
        pairs = list(P.index)
        a = np.array([p.split(" >> ")[0] for p in pairs]); b = np.array([p.split(" >> ")[1] for p in pairs])
        ua, ub = sorted(set(a)), sorted(set(b))
        out[key] = dict(key=key, M=P.to_numpy(float), pairs=pairs, conds=list(P.columns), a=a, b=b, ua=ua, ub=ub,
                        ia=np.array([ua.index(x) for x in a]), ib=np.array([ub.index(x) for x in b]))
    return out


def additive_residual(M, iters=60):
    mask = np.isfinite(M); Y = np.where(mask, M, 0.0)
    mu = Y[mask].mean(); r = np.zeros(M.shape[0]); c = np.zeros(M.shape[1])
    for _ in range(iters):
        r = ((Y - mu - c[None, :]) * mask).sum(1) / np.maximum(mask.sum(1), 1)
        c = ((Y - mu - r[:, None]) * mask).sum(0) / np.maximum(mask.sum(0), 1)
        mu = ((Y - r[:, None] - c[None, :]) * mask).sum() / mask.sum()
    return np.where(mask, M - mu - r[:, None] - c[None, :], np.nan), r, c, mu


def row_scaled_residual(M, iters=5000, tol=1e-10):
    """Residual of y_ij = mu + r_i + s_i c_j (alternating least squares from the additive start; a
    negative slope is set to zero; s rescaled to mean one)."""
    mask = np.isfinite(M)
    _, r, c, _ = additive_residual(M, iters=500)
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
    return np.where(mask, M - mu - r[:, None] - s[:, None] * c[None, :], np.nan)


def train_residual(M, tr):
    colm = np.nanmean(M[tr], axis=0); gm = np.nanmean(M[tr])
    return M - np.nanmean(M, axis=1, keepdims=True) - colm[None, :] + gm, colm


def profile(R, idx, n, rows, k=SHRINK):
    # Materialize only training rows before any fitted profile statistic.
    R, idx = np.array(R[rows], copy=True), np.asarray(idx)[rows].copy()
    S = np.zeros((n, R.shape[1])); N = np.zeros_like(S)
    for p in range(len(R)):
        ok = np.isfinite(R[p]); S[idx[p], ok] += R[p, ok]; N[idx[p], ok] += 1
    return S / (N + k)


def heldout_r2(A, folds, R_fixed=None, backfit=10):
    M, ia, ib, na, nb = A["M"], A["ia"], A["ib"], len(A["ua"]), len(A["ub"])
    num = dict(p1=0.0, p2=0.0, both=0.0); den = 0.0
    for f in folds:
        te = set(f.tolist()); tr = [p for p in range(len(M)) if p not in te]
        R = R_fixed if R_fixed is not None else train_residual(M, np.array(tr))[0]
        U = profile(R, ia, na, tr); W = profile(R, ib, nb, tr)
        Uj = np.zeros_like(U); Wj = np.zeros_like(W)
        for _ in range(backfit):
            Uj = profile(R - Wj[ib], ia, na, tr); Wj = profile(R - Uj[ia], ib, nb, tr)
        for p in te:
            ok = np.isfinite(R[p]); y = R[p, ok]
            num["p1"] += np.sum((y - U[ia[p], ok]) ** 2); num["p2"] += np.sum((y - W[ib[p], ok]) ** 2)
            num["both"] += np.sum((y - Uj[ia[p], ok] - Wj[ib[p], ok]) ** 2); den += np.sum(y ** 2)
    return {k: 1 - v / den for k, v in num.items()}


def pair_agreement(R, a, b):
    valid = np.nanstd(R, axis=1) > 1e-6
    C = pd.DataFrame(R.T).corr().to_numpy(); C[~valid] = np.nan; C[:, ~valid] = np.nan
    iu = np.triu_indices(len(R), 1); cc = C[iu]
    s1 = a[iu[0]] == a[iu[1]]; s2 = b[iu[0]] == b[iu[1]]
    return np.nanmean(cc[s1 & ~s2]), np.nanmean(cc[s2 & ~s1]), np.nanmean(cc[~s1 & ~s2])


def joint_profiles(R, ia, ib, na, nb, tr, rounds=10):
    """Both partners' shrunken residual profiles, each refitted on what the other leaves."""
    R, ia, ib = np.array(R[tr], copy=True), np.asarray(ia)[tr].copy(), np.asarray(ib)[tr].copy()
    tr = np.arange(len(R))
    U = np.zeros((na, R.shape[1])); Wp = np.zeros((nb, R.shape[1]))
    for _ in range(rounds):
        U = profile(R - Wp[ib], ia, na, tr)
        Wp = profile(R - U[ia], ib, nb, tr)
    return U, Wp


def panels(M, colm, u, src_rows):
    """H ranks conditions by the training mean plus the records of both partners; a partner without a
    record contributes zero. N is the model-free comparator: the conditions under which the target's own
    building blocks gave their highest average raw yield."""
    raw = np.nanmean(M[src_rows], axis=0) if len(src_rows) else colm
    return dict(H=top_k(colm + u), B=top_k(colm), N=top_k(raw))


def scored(t, pn, C, meta):
    union = np.unique(np.r_[pn["H"], pn["B"], pn["N"], C])
    if not np.isfinite(t[union]).all():
        return None
    return dict(meta, H=t[pn["H"]].max(), B=t[pn["B"]].max(), N=t[pn["N"]].max(), C=t[C].max())


def owner_side(A, owner):
    io = A["ia"] if owner == "p1" else A["ib"]; no = len(A["ua"]) if owner == "p1" else len(A["ub"])
    return io, no


def cv_records(A, owner=None):
    """owner is kept only to name the resampling unit in the outputs; it plays no part in the panels."""
    M = A["M"]; ia, ib = A["ia"], A["ib"]; na, nb = len(A["ua"]), len(A["ub"])
    io, _ = owner_side(A, owner) if owner else (ia, na)
    rows = []
    for rep in range(NREP):
        for f in five_folds(len(M), stream(A["key"], "transfer", rep)):
            te = set(f.tolist()); tr = np.array([p for p in range(len(M)) if p not in te])
            R, colm = train_residual(M, tr)
            U, Wp = joint_profiles(R, ia, ib, na, nb, tr)
            seen_a, seen_b = set(ia[tr]), set(ib[tr]); C = greedy_trio(M[tr])
            for p in f:
                if np.all(np.isnan(M[p])) or not (ia[p] in seen_a or ib[p] in seen_b):
                    continue
                u = (U[ia[p]] if ia[p] in seen_a else 0.0) + (Wp[ib[p]] if ib[p] in seen_b else 0.0)
                src = tr[(ia[tr] == ia[p]) | (ib[tr] == ib[p])]
                r = scored(M[p], panels(M, colm, u, src), C,
                           dict(screen=A["key"], rep=rep, pair=int(p), owner=int(io[p]),
                                known_a=int(ia[p] in seen_a), known_b=int(ib[p] in seen_b)))
                if r:
                    rows.append(r)
    return rows


def random_profile_records(A, owner, ndraw=100):
    """Own history panel against panels built from randomly chosen other building blocks' histories."""
    M = A["M"]; io, no = owner_side(A, owner); rows = []
    for rep in range(NREP):
        rng = stream(A["key"], "random_profile", rep)
        for f in five_folds(len(M), stream(A["key"], "transfer", rep)):
            tr = np.setdiff1d(np.arange(len(M)), f); R, colm = train_residual(M, tr); U = profile(R, io, no, tr)
            seen = np.unique(io[tr]); base = top_k(colm)
            for q in f:
                if io[q] not in seen:
                    continue
                other = seen[seen != io[q]]
                if not len(other):
                    continue
                y = M[q]; own = top_k(colm + U[io[q]])
                if not np.isfinite(y[np.r_[own, base]]).all():
                    continue
                bv = y[base].max(); vals = []
                for o in rng.choice(other, ndraw, replace=True):
                    ch = top_k(colm + U[o])
                    if np.isfinite(y[ch]).all():
                        vals.append(y[ch].max() - bv)
                if vals:
                    rows.append(dict(screen=A["key"], pair=int(q), owner=int(io[q]), own=y[own].max() - bv, random=float(np.mean(vals))))
    return rows


def archive_records(A, owner, lab):
    M = A["M"]; ia, ib = A["ia"], A["ib"]; io, no = owner_side(A, owner)
    groups = np.array([lab.get(p, "") for p in A["pairs"]]); eligible = np.flatnonzero(groups != "")
    if len(set(groups[eligible])) < 2:
        return []
    rows = []
    for g in sorted(set(groups[eligible])):
        from boundaries import frozen_folds
        tr, te = frozen_folds(f'archive|{A["key"]}|{g}')
        R, colm = train_residual(M, tr); C = greedy_trio(M[tr])
        U, Wp = joint_profiles(R, ia, ib, len(A["ua"]), len(A["ub"]), tr)
        na = np.bincount(ia[tr], minlength=len(A["ua"])); nb = np.bincount(ib[tr], minlength=len(A["ub"]))
        for q in te:
            if not (na[ia[q]] > 0 and nb[ib[q]] > 0):      # both partners recur outside the held-out month
                continue
            u = U[ia[q]] + Wp[ib[q]]
            src = tr[(ia[tr] == ia[q]) | (ib[tr] == ib[q])]
            r = scored(M[q], panels(M, colm, u, src), C,
                       dict(screen=A["key"], month=g, pair=int(q), owner=int(io[q])))
            if r:
                rows.append(r)
    return rows


def sealed_records(A, owner, S):
    """Profiles and condition means from every analysis pair; a sealed pair is scored if at least one of
    its building blocks occurs in at least two of them."""
    M = A["M"]; ia, ib = A["ia"], A["ib"]; n = len(M)
    io, _ = owner_side(A, owner)
    ua, ub = A["ua"], A["ub"]
    R, colm = train_residual(M, np.arange(n))
    U, Wp = joint_profiles(R, ia, ib, len(ua), len(ub), np.arange(n))
    cnt_a = np.bincount(ia, minlength=len(ua)); cnt_b = np.bincount(ib, minlength=len(ub))
    C = greedy_trio(M); rows = []
    P = S.pivot(index="subpair", columns="condition_id", values="yield").reindex(columns=A["conds"])
    for sp, t in zip(P.index, P.to_numpy(float)):
        m1, m2 = sp.split(" >> ")
        ka = m1 in ua and cnt_a[ua.index(m1)] >= 2
        kb = m2 in ub and cnt_b[ub.index(m2)] >= 2
        if not (ka or kb):                      # at least one partner must carry a record
            continue
        u = (U[ua.index(m1)] if ka else 0.0) + (Wp[ub.index(m2)] if kb else 0.0)
        src = np.flatnonzero((ia == ua.index(m1)) if ka else (ib == ub.index(m2)))
        o = io[src[0]] if len(src) else 0
        r = scored(t, panels(M, colm, u, src), C,
                   dict(screen=A["key"], subpair=sp, owner=int(o), known_a=int(ka), known_b=int(kb)))
        if r:
            rows.append(r)
    return rows


def design_controls(AR, V):
    """Is the partner that owns the conditions just the side with more molecules, the structurally more
    spread side, or the side carrying more of the yield? Each control names a side per array; the count is
    how often it names the side the held-out R2 picks."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def spread(names):
        fps = [gen.GetFingerprint(m) for m in (Chem.MolFromSmiles(x) for x in names) if m is not None]
        if len(fps) < 2:
            return np.nan
        sims = [s for i, fp in enumerate(fps) for s in DataStructs.BulkTanimotoSimilarity(fp, fps[i + 1:])]
        return 1 - float(np.mean(sims))

    rows = []
    for key, A in AR.items():
        _, r, _, _ = additive_residual(A["M"])
        v1 = float(pd.Series(r).groupby(A["ia"]).mean().var()); v2 = float(pd.Series(r).groupby(A["ib"]).mean().var())
        n1, n2 = len(A["ua"]), len(A["ub"]); s1, s2 = spread(A["ua"]), spread(A["ub"])
        owner = V.loc[key, "owner"]
        rows.append(dict(screen=key, owner=owner, n_1=n1, n_2=n2, spread_1=s1, spread_2=s2,
                         variance_1=v1, variance_2=v2,
                         by_count="p1" if n1 > n2 else "p2", by_spread="p1" if s1 > s2 else "p2",
                         by_variance="p1" if v1 > v2 else "p2"))
    return pd.DataFrame(rows)


def contrasts(z):
    return z.assign(H_B=z.H - z.B, H_C=z.H - z.C, N_B=z.N - z.B, N_C=z.N - z.C, H_N=z.H - z.N, C_B=z.C - z.B,
                    rescue_B=(z.B < USABLE) & (z.H >= USABLE), loss_B=(z.H < USABLE) & (z.B >= USABLE),
                    rescue_C=(z.C < USABLE) & (z.H >= USABLE), loss_C=(z.H < USABLE) & (z.C >= USABLE))


def summary(z, per_target):
    z = contrasts(z)
    cols = ["H_B", "H_C", "N_B", "N_C", "H_N", "C_B", "rescue_B", "loss_B", "rescue_C", "loss_C"]
    if per_target:
        z = z.groupby(["screen", "pair", "owner"], as_index=False)[cols].mean()
    out = dict(n=int(len(z)), n_arrays=int(z.screen.nunique()), n_owners=int(z.groupby(["screen", "owner"]).ngroups))
    for c in ("H_B", "H_C", "N_B", "N_C", "H_N", "C_B"):
        m, lo, hi = owner_interval(z, c, "owner"); out[c] = [100 * m, 100 * lo, 100 * hi]
    for c in ("rescue_B", "loss_B", "rescue_C", "loss_C"):
        out[c] = 100 * float(z[c].mean())
    return out


def main():
    d = load_analysis(); AR = partner_arrays(d)
    check_roles(AR)
    rows = []
    for key, A in AR.items():
        folds = five_folds(len(A["M"]), stream(key, "cv"))
        cv = heldout_r2(A, folds)
        seq = heldout_r2(A, folds, backfit=1)
        sc = heldout_r2(A, folds, R_fixed=row_scaled_residual(A["M"]))
        R, *_ = additive_residual(A["M"])
        r1, r2, r0 = pair_agreement(R, A["a"], A["b"])
        rows.append(dict(screen=key, pairs=len(A["M"]), partners_1=len(A["ua"]), partners_2=len(A["ub"]),
                         r2_p1=cv["p1"], r2_p2=cv["p2"], r2_both=cv["both"], r2_both_sequential=seq["both"],
                         r2_p1_slope_removed=sc["p1"], r2_p2_slope_removed=sc["p2"],
                         r_share1=r1, r_share2=r2, r_unrelated=r0))
        print(f"  {key:28s} R2 p1 {cv['p1']:.3f} p2 {cv['p2']:.3f} both {cv['both']:.3f}", flush=True)
    V = pd.DataFrame(rows)
    V["owner"] = np.where(V.r2_p1 >= V.r2_p2, "p1", "p2")
    V["stronger"] = V[["r2_p1", "r2_p2"]].max(axis=1)
    V["stronger_slope_removed"] = V[["r2_p1_slope_removed", "r2_p2_slope_removed"]].max(axis=1)
    V.to_csv(OUT / "history_arrays.csv", index=False)
    own = V.set_index("screen").owner.to_dict()

    cv = pd.DataFrame([r for key, A in AR.items() for r in cv_records(A, own[key])])
    cv.to_csv(OUT / "history_cv.csv", index=False)
    rp = pd.DataFrame([r for key, A in AR.items() for r in random_profile_records(A, own[key])])
    rp = rp.groupby(["screen", "pair", "owner"], as_index=False)[["own", "random"]].mean()
    rp["difference"] = rp.own - rp.random
    rp.to_csv(OUT / "history_random_profile.csv", index=False)

    meta = load_archive(); meta["subpair"] = meta.sub_1_smiles + " >> " + meta.sub_2_smiles
    lab = {}
    for (key, sp), x in meta.groupby(["screen", "subpair"]):
        s = sorted(set(x.archive.dropna().astype(str)))
        if len(s) == 1:                               # a pair recorded under two months belongs to neither
            lab.setdefault(key, {})[sp] = s[0]
    ar = pd.DataFrame([r for key, A in AR.items() for r in archive_records(A, own[key], lab.get(key, {}))])
    ar.to_csv(OUT / "history_months.csv", index=False)

    ctrl = design_controls(AR, V.set_index("screen"))
    ctrl.to_csv(OUT / "history_owner_controls.csv", index=False)

    S = load_sealed()
    se = pd.DataFrame([r for key, A in AR.items() if key in set(S.screen) and key not in PROSPECTIVE_SEALED
                       for r in sealed_records(A, own[key], S[S.screen == key])])
    se.to_csv(OUT / "history_sealed.csv", index=False)

    out = dict(n_arrays=int(len(V)), n_pairs=int(V.pairs.sum()),
               median_r_share1=float(V.r_share1.median()), median_r_share2=float(V.r_share2.median()),
               median_r_unrelated=float(V.r_unrelated.median()),
               median_r2_both=float(V.r2_both.median()), median_r2_both_sequential=float(V.r2_both_sequential.median()),
               median_stronger=float(V.stronger.median()), median_stronger_slope_removed=float(V.stronger_slope_removed.median()),
               owners=V.set_index("screen")[["owner", "r2_p1", "r2_p2"]].to_dict("index"),
               cv=summary(cv, per_target=True), cv_records=int(len(cv)),
               random_profile={c: [100 * v for v in owner_interval(rp, c, "owner")] for c in ("own", "random", "difference")},
               random_profile_n=int(len(rp)),
               months=summary(ar.rename(columns={"month": "rep"}), per_target=False),
               months_per_array=ar.groupby("screen").size().to_dict(),
               sealed=summary(se.rename(columns={"subpair": "pair"}), per_target=False),
               sealed_per_array=se.groupby("screen").size().to_dict(),
               owner_controls={c: int((ctrl[c] == ctrl.owner).sum()) for c in ("by_count", "by_spread", "by_variance")},
               owner_controls_n=int(len(ctrl)),
               owner_is_partner_2=int((ctrl.owner == "p2").sum()))
    save_json("history.json", out)
    for k in ("cv", "months", "sealed"):
        print(k, {a: (b if not isinstance(b, list) else [round(x, 2) for x in b]) for a, b in out[k].items()})
    print("random profile", {k: [round(x, 2) for x in v] for k, v in out["random_profile"].items()}, out["random_profile_n"])
    print({k: v for k, v in out.items() if k.startswith("median")})


if __name__ == "__main__":
    main()
