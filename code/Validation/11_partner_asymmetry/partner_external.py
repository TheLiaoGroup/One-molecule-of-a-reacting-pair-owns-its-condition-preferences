"""Does the aryl electrophile's history predict a new pair's condition pattern better than its partner's?

Pre-registered in 04_ANALYSES/10_partner_asymmetry/PREREGISTRATION.md.
  power     emulate the external design on held-out analysis pairs (reads no external outcome)
  external  score the additional archived combinations (reads external outcomes)
"""
import os
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("P2_RESULTS_ROOT", ROOT / "05_DATA/Results")) / "partner_asymmetry"
import sys
sys.path.insert(0, str(ROOT / "06_CODE/Validation/06_validation"))
from validation_core import canonical
SEED, SHRINK, MIN_CELLS = 20261002, 4.0, 8


def load_analysis():
    A = pd.read_csv(ROOT / "05_DATA/Analysis/02_ANALYSIS_TRAINING/analysis_measurements.csv.gz",
                    usecols=["screen", "pair_id", "sub_1_smiles", "sub_2_smiles", "condition_id", "yield",
                             "allowed_for_model_fit", "qc_duplicate_condition", "qc_missing_yield"])
    return A[A.allowed_for_model_fit & ~A.qc_duplicate_condition & ~A.qc_missing_yield]


def electrophile_roles():
    """Role index (1 or 2) of the aryl (pseudo)halide in every array that has one, read from the role table."""
    R = pd.read_csv(Path(os.environ.get("P2_RESULTS_ROOT", ROOT / "05_DATA/Results")) / "primary/history_roles.csv")
    roles = {}
    for r in R.itertuples():
        h1, h2 = "halide" in r.role_1, "halide" in r.role_2
        if h1 != h2:
            roles[r.screen] = 1 if h1 else 2
    if not roles:
        raise SystemExit("no electrophile arrays found")
    return roles


def array_matrix(A, screen):
    """Molecules are identified by RDKit canonical SMILES, as in the original external evaluation (amendment 1)."""
    G = A[A.screen == screen]
    P = G.pivot_table(index="pair_id", columns="condition_id", values="yield")
    meta = G.drop_duplicates("pair_id").set_index("pair_id").loc[P.index]
    can = {s: canonical(s) for s in set(meta.sub_1_smiles) | set(meta.sub_2_smiles)}
    return (P.to_numpy(float), meta.sub_1_smiles.map(can).to_numpy(), meta.sub_2_smiles.map(can).to_numpy(),
            list(P.columns))


def slope_residuals(Y, s):
    E = np.full_like(Y, np.nan)
    for i in range(len(Y)):
        o = ~np.isnan(Y[i]) & ~np.isnan(s)
        X = np.c_[np.ones(o.sum()), s[o]]
        E[i, o] = Y[i, o] - X @ np.linalg.lstsq(X, Y[i, o], rcond=None)[0]
    return E


def train(Y, s1, s2):
    """Single-role shrunken profiles of slope-adjusted residuals; usable = at least two distinct partners."""
    s = np.nanmean(Y, 0)
    E = slope_residuals(Y, s); M = ~np.isnan(E)
    prof = {}
    for role, keys, other in ((1, s1, s2), (2, s2, s1)):
        prof[role] = {u: np.nansum(E[keys == u], 0) / (M[keys == u].sum(0) + SHRINK)
                      for u in np.unique(keys) if len(set(other[keys == u])) >= 2}
    return s, prof


def score(y, s, prof, a, b, erole):
    """Squared errors of electrophile-only and partner-only predictions for one target."""
    o = ~np.isnan(y) & ~np.isnan(s)
    if o.sum() < MIN_CELLS:
        return None
    X = np.c_[np.ones(o.sum()), s[o]]
    proj = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    e = proj(y[o])
    key = {1: a, 2: b}
    uE = proj(prof[erole][key[erole]][o]); uN = proj(prof[3 - erole][key[3 - erole]][o])
    return dict(sse_e=float(np.sum((e - uE) ** 2)), sse_n=float(np.sum((e - uN) ** 2)), sst=float(np.sum(e ** 2)))


def summarise(L, n_boot, rng):
    """Pooled difference D = R2_electrophile - R2_partner with two resampling schemes."""
    # A draw with zero pooled denominator is undefined; it is counted, not silently dropped.
    D = lambda d: (d.sse_n.sum() - d.sse_e.sum()) / d.sst.sum() if d.sst.sum() > 0 else np.nan
    L = L.assign(gain=L.sse_n - L.sse_e)
    A = L.groupby("screen")[["gain", "sst"]].sum()                     # whole-array units
    ia = rng.integers(0, len(A), (n_boot, len(A)))
    num, den = A.gain.to_numpy()[ia].sum(1), A.sst.to_numpy()[ia].sum(1)
    whole = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    num = np.zeros(n_boot); den = np.zeros(n_boot)                     # electrophile clusters within arrays
    for _, g in L.groupby("screen"):
        c = g.groupby("electrophile")[["gain", "sst"]].sum()
        ic = rng.integers(0, len(c), (n_boot, len(c)))
        num += c.gain.to_numpy()[ic].sum(1); den += c.sst.to_numpy()[ic].sum(1)
    clus = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    arrays = A.index
    per_array = L.groupby("screen").apply(lambda g: D(g), include_groups=False)
    return dict(targets=int(len(L)), arrays=int(len(arrays)), D=float(D(L)),
                r2_electrophile=float(1 - L.sse_e.sum() / L.sst.sum()), r2_partner=float(1 - L.sse_n.sum() / L.sst.sum()),
                ci_whole_array=[float(np.nanpercentile(whole, 2.5)), float(np.nanpercentile(whole, 97.5))],
                ci_electrophile_cluster=[float(np.nanpercentile(clus, 2.5)), float(np.nanpercentile(clus, 97.5))],
                undefined_draws={"whole_array": int(np.isnan(whole).sum()), "electrophile_cluster": int(np.isnan(clus).sum())},
                arrays_positive=int((per_array > 0).sum()), per_array={k: float(v) for k, v in per_array.items()})


def external_targets():
    T = pd.read_csv(ROOT / "05_DATA/Archived_Evaluation/07_RESTRICTED_LOCAL_ONLY/raw/retrospective_target_manifest.csv")
    T = T[T.residual_eligible & T.known_a & T.known_b].copy()
    T["canonical_a"] = T.canonical_a.map(canonical); T["canonical_b"] = T.canonical_b.map(canonical)
    return T


def power(reps, n_boot):
    A = load_analysis(); roles = electrophile_roles(); rng = np.random.default_rng(SEED)
    counts = external_targets().groupby("screen").size()
    counts = counts[counts.index.isin(roles.keys())]
    mats = {k: array_matrix(A, k) for k in counts.index}
    rows = []
    for rep in range(reps):
        L = []
        for k, n in counts.items():
            Y, s1, s2, _ = mats[k]; hold = rng.random(len(Y)) < min(0.5, max(0.15, 2.5 * n / len(Y))); tr = ~hold
            s, prof = train(Y[tr], s1[tr], s2[tr])
            cand = [i for i in np.flatnonzero(hold) if s1[i] in prof[1] and s2[i] in prof[2]]
            for i in rng.permutation(cand)[:n]:
                r = score(Y[i], s, prof, s1[i], s2[i], roles[k])
                if r:
                    L.append(dict(screen=k, electrophile=(s1 if roles[k] == 1 else s2)[i], **r))
        res = summarise(pd.DataFrame(L), n_boot, rng)
        rows.append(dict(rep=rep, D=res["D"], lo_whole=res["ci_whole_array"][0], lo_cluster=res["ci_electrophile_cluster"][0],
                         arrays_positive=res["arrays_positive"], arrays=res["arrays"], targets=res["targets"]))
    return pd.DataFrame(rows)


def external(n_boot):
    A = load_analysis(); roles = electrophile_roles(); rng = np.random.default_rng(SEED)
    T = external_targets(); T = T[T.screen.isin(roles.keys())]
    models = {}
    for k in T.screen.unique():
        Y, s1, s2, conds = array_matrix(A, k); s, prof = train(Y, s1, s2); models[k] = (s, prof, conds)
        sub = T[T.screen == k]
        if not (sub.canonical_a.isin(prof[1].keys()).all() and sub.canonical_b.isin(prof[2].keys()).all()):
            raise SystemExit(f"usable-block rule disagrees with the manifest in {k}")
    # Outcomes are read only after every training-side profile has been formed.
    Z = pd.read_csv(ROOT / "05_DATA/Archived_Evaluation/07_RESTRICTED_LOCAL_ONLY/derived/additional_retrospective_measurements.csv.gz")
    Z = Z[Z.read_for_scoring & Z.finite_numeric & Z.candidate_id.isin(T.candidate_id)]
    Z = Z[~Z.duplicated(["candidate_id", "condition_id"], keep=False)]
    L = []
    for c in T.itertuples():
        s, prof, conds = models[c.screen]
        z = Z[Z.candidate_id == c.candidate_id].set_index("condition_id")["yield"]
        y = np.array([z.get(cid, np.nan) for cid in conds], float)
        r = score(y, s, prof, c.canonical_a, c.canonical_b, roles[c.screen])
        if r:
            L.append(dict(screen=c.screen, candidate_id=c.candidate_id,
                          electrophile=c.canonical_a if roles[c.screen] == 1 else c.canonical_b, **r))
    L = pd.DataFrame(L)
    return L, summarise(L, n_boot, rng)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["power", "external"])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == "power":
        P = power(a.reps, a.boot); P.to_csv(OUT / "power_emulation.csv", index=False)
        print(P.describe().round(3).to_string())
        print("power, whole-array lower bound > 0:", round((P.lo_whole > 0).mean(), 3))
        print("power, electrophile-cluster lower bound > 0:", round((P.lo_cluster > 0).mean(), 3))
    else:
        L, res = external(a.boot)
        L.to_csv(OUT / "external_target_losses.csv", index=False)
        (OUT / "external_summary.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
