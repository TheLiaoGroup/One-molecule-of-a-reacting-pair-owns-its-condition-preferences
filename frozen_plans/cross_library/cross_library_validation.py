"""Does the owner's record beat its partner's on pairs that lie outside this paper's rectangular arrays?

Pre-registered in 04_ANALYSES/13_cross_library/PREREGISTRATION.md.
  power   emulate both endpoints on analysis pairs only (reads no sealed outcome)
  freeze  build the outcome-free target manifest and the training-only profiles
  score   read the sealed outcomes and score

The estimators are imported, not reimplemented: the single-role profiles, the loss and the pooled
statistic come from 03_CODE/11_partner_asymmetry/partner_external.py, which is frozen under its own
pre-registration and is left byte-identical by this module.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "06_RESULTS" / "cross_library"
SEALED = ROOT / "02_DATA/04_SEALED/never_opened_cross_library.csv.gz"
sys.path[:0] = [str(ROOT / "03_CODE/00_common"), str(ROOT / "03_CODE/00_common/ctb"),
                str(ROOT / "03_CODE/06_validation"), str(ROOT / "03_CODE/11_partner_asymmetry")]
import history
from boundaries import read_table, validate_training
from validation_core import canonical
import partner_external as pe

SEED, NBOOT = 20261007, 2000


def owner_roles():
    T = pd.read_csv(ROOT / "06_RESULTS/primary/anatomy_arrays.csv")
    roles = {r.screen: int(r.owner_role) for r in T.itertuples()}
    if not roles:
        raise SystemExit("no owners found in anatomy_arrays.csv")
    return roles


def training():
    return validate_training(read_table(ROOT / "02_DATA/02_ANALYSIS_TRAINING/analysis_measurements.csv.gz"))


def sealed():
    S = read_table(SEALED)
    if S.allowed_for_model_fit.any():
        raise SystemExit("sealed rows are marked fittable")
    return S


def targets(S):
    """One row per sealed pair: identities and the conditions it covers. No outcome."""
    P = S.drop_duplicates("subpair").copy()
    P["canonical_a"] = [canonical(x) for x in P.sub_1_smiles]
    P["canonical_b"] = [canonical(x) for x in P.sub_2_smiles]
    P["n_conditions"] = P.subpair.map(S.groupby("subpair").condition_id.nunique())
    return P[["screen", "subpair", "sub_1_smiles", "sub_2_smiles", "canonical_a", "canonical_b", "n_conditions"]]


def models(train, screens):
    return {k: (*pe.train(*pe.array_matrix(train, k)[:3]), pe.array_matrix(train, k)[3]) for k in screens}


def losses(T, M, roles, outcomes):
    """Owner-only against partner-only squared error for every scorable sealed pair."""
    by_pair = {k: g.set_index("condition_id")["yield"] for k, g in outcomes.groupby(["screen", "subpair"])}
    L = []
    for c in T.itertuples():
        s, prof, conds = M[c.screen]
        if c.canonical_a not in prof[1] or c.canonical_b not in prof[2]:
            continue
        z = by_pair[(c.screen, c.subpair)]
        z = z[~z.index.duplicated(keep=False)]
        y = np.array([z.get(cid, np.nan) for cid in conds], float)
        r = pe.score(y, s, prof, c.canonical_a, c.canonical_b, roles[c.screen])
        if r:
            own = c.canonical_a if roles[c.screen] == 1 else c.canonical_b
            L.append(dict(screen=c.screen, subpair=c.subpair, electrophile=own,
                          sse_e=r["sse_e"], sse_n=r["sse_n"], sst=r["sst"]))
    if not L:
        raise SystemExit("zero sealed pairs were scorable")
    return pd.DataFrame(L)


def sign_test(per_array):
    """Secondary endpoint: in how many arrays does the owner's record win, against a coin."""
    from math import comb
    vals = [v for v in per_array.values() if v == v]
    k, n = sum(v > 0 for v in vals), len(vals)
    p = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return dict(arrays=n, favouring_owner=k, one_sided_p=p)


def power(reps):
    """Both endpoints emulated on held-out analysis pairs at the sealed target counts."""
    train, roles, rng = training(), owner_roles(), np.random.default_rng(SEED)
    counts = targets(sealed()).groupby("screen").size()
    mats = {k: pe.array_matrix(train, k) for k in counts.index}
    rows = []
    for rep in range(reps):
        L = []
        for k, n in counts.items():
            Y, s1, s2, _ = mats[k]
            hold = np.zeros(len(Y), bool)
            hold[rng.permutation(len(Y))[:min(n, int(.4 * len(Y)))]] = True
            s, prof = pe.train(Y[~hold], s1[~hold], s2[~hold])
            cand = [i for i in np.flatnonzero(hold) if s1[i] in prof[1] and s2[i] in prof[2]]
            for i in cand:
                r = pe.score(Y[i], s, prof, s1[i], s2[i], roles[k])
                if r:
                    L.append(dict(screen=k, electrophile=(s1 if roles[k] == 1 else s2)[i], **r))
        if not L:
            continue
        res = pe.summarise(pd.DataFrame(L), 400, rng)
        st = sign_test(res["per_array"])
        rows.append(dict(rep=rep, D=res["D"], lo_cluster=res["ci_electrophile_cluster"][0],
                         lo_whole=res["ci_whole_array"][0], targets=res["targets"],
                         arrays=st["arrays"], favouring_owner=st["favouring_owner"], sign_p=st["one_sided_p"]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["power", "freeze", "score"])
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == "power":
        P = power(a.reps)
        P.to_csv(OUT / "power.csv", index=False)
        out = dict(reps=int(len(P)), median_D=float(P.D.median()), median_targets=float(P.targets.median()),
                   power_cluster=float((P.lo_cluster > 0).mean()), power_whole=float((P.lo_whole > 0).mean()),
                   median_arrays_favouring_owner=float(P.favouring_owner.median()),
                   median_arrays=float(P.arrays.median()),
                   power_sign_test=float((P.sign_p < 0.05).mean()))
        (OUT / "power.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        return
    train, roles, S = training(), owner_roles(), sealed()
    T = targets(S)
    M = models(train, sorted(T.screen.unique()))
    if a.mode == "freeze":
        T.to_csv(OUT / "target_manifest.csv", index=False)
        usable = [dict(screen=c.screen, subpair=c.subpair,
                       owner_has_record=c.canonical_a in M[c.screen][1][1] if roles[c.screen] == 1
                       else c.canonical_b in M[c.screen][1][2],
                       both_have_record=(c.canonical_a in M[c.screen][1][1]) and (c.canonical_b in M[c.screen][1][2]))
                  for c in T.itertuples()]
        pd.DataFrame(usable).to_csv(OUT / "training_only_usability.csv", index=False)
        print(f"frozen: {len(T)} sealed targets in {T.screen.nunique()} arrays, "
              f"{sum(u['both_have_record'] for u in usable)} with a record on both sides")
        return
    # Outcomes are read only here, after the manifest and the training-side profiles exist.
    L = losses(T, M, roles, S[["screen", "subpair", "condition_id", "yield"]])
    res = pe.summarise(L, NBOOT, np.random.default_rng(SEED))
    res["sign_test"] = sign_test(res["per_array"])
    L.to_csv(OUT / "target_losses.csv", index=False)
    (OUT / "cross_library.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "per_array"}, indent=2))
    print("\nper array:")
    for k, v in sorted(res["per_array"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:28s} {v:+.3f}")


if __name__ == "__main__":
    main()
