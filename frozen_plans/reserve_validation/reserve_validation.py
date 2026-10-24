"""Do the ownership rule and the record rule hold on pairs this paper has never analysed?

Pre-registered in 04_ANALYSES/11_reserve_validation/PREREGISTRATION.md.
  power   emulate the three endpoints on held-out analysis pairs (reads no reserve outcome)
  freeze  build the target manifest and the training-only predictions (reads no reserve outcome)
  score   read the sealed reserve outcomes and score the three endpoints

Every estimator is imported from the frozen modules that produced the main text; nothing is
reimplemented here, so a change to an estimator cannot leave this test using an older rule.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "06_RESULTS" / "reserve_validation"
SEALED = ROOT / "02_DATA/04_SEALED/never_analysed_reserve.csv.gz"
sys.path[:0] = [str(ROOT / "03_CODE/00_common"), str(ROOT / "03_CODE/00_common/ctb"),
                str(ROOT / "03_CODE/06_validation"), str(ROOT / "03_CODE/11_partner_asymmetry")]
import anatomy, history, validation_core as v
from boundaries import read_table, validate_training
from validation_core import canonical
import partner_external as pe

SEED, NBOOT, MIN_PAIRS_E1 = 20261004, 2000, 50


def owner_roles():
    """Role index of the owning building block in every analysed array, read from the frozen result table."""
    T = pd.read_csv(ROOT / "06_RESULTS/primary/anatomy_arrays.csv")
    roles = {r.screen: int(r.owner_role) for r in T.itertuples()}
    if not roles:
        raise SystemExit("no owners found in anatomy_arrays.csv")
    return roles


def training():
    return validate_training(read_table(ROOT / "02_DATA/02_ANALYSIS_TRAINING/analysis_measurements.csv.gz"))


def reserve():
    """The sealed never-analysed rows. The reader verifies the registered SHA-256 before returning them."""
    R = read_table(SEALED)
    if R.allowed_for_model_fit.any():
        raise SystemExit("reserve rows are marked fittable")
    return R


def reserve_targets(R):
    """One row per reserve pair: identities and condition count only, no outcome."""
    P = R.drop_duplicates("subpair").copy()
    P["canonical_a"] = [canonical(s) for s in P.sub_1_smiles]
    P["canonical_b"] = [canonical(s) for s in P.sub_2_smiles]
    n = R.groupby("subpair").condition_id.nunique()
    P["n_conditions"] = P.subpair.map(n)
    return P[["screen", "subpair", "sub_1_smiles", "sub_2_smiles", "canonical_a", "canonical_b", "n_conditions"]]


# ------------------------------------------------------------------ E1: ownership, re-estimated on reserve pairs
def ownership(R, roles, rng):
    """The paper's share estimator applied to reserve pairs alone. Does the same partner still own it?"""
    rows = []
    for key, A in history.partner_arrays(R).items():
        M = A["M"]
        keep = np.nanmax(M, 1) >= anatomy.FORMS            # the paper's product-forming gate, same constant
        if keep.sum() < MIN_PAIRS_E1:
            rows.append(dict(screen=key, reserve_product_forming=int(keep.sum()), eligible=False))
            continue
        E, _, _ = anatomy.pattern(M[keep])
        a, b = A["a"][keep], A["b"][keep]
        S = anatomy.Shares(E, a, b)
        s = S.compute(np.ones(keep.sum()))
        own = roles[key]
        draws = np.empty(NBOOT)                        # resample molecules, as the main text does
        for t in range(NBOOT):
            d = S.compute(anatomy.molecule_weights(a, b, rng))
            draws[t] = d[str(own)] - d[str(3 - own)]
        rows.append(dict(screen=key, reserve_product_forming=int(keep.sum()), eligible=True,
                         paper_owner_role=own, share_role1=s["1"], share_role2=s["2"], share_neither=s["N"],
                         reserve_owner_role=1 if s["1"] >= s["2"] else 2,
                         owner_margin=float(s[str(own)] - s[str(3 - own)]),
                         draws_favouring_paper_owner=float(np.nanmean(draws > 0)),
                         margin_ci=[float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ E2: whose record predicts a new combination
def record_transfer(train, T, roles, rng, outcomes=None):
    """Owner-only against partner-only prediction of a reserve pair's condition pattern.

    The estimator is partner_external's, with the owning building block in place of the aryl electrophile;
    its cluster column is named `electrophile`, and here that column holds the owner.
    """
    models, L = {}, []
    for key in sorted(T.screen.unique()):
        Y, s1, s2, conds = pe.array_matrix(train, key)
        models[key] = (*pe.train(Y, s1, s2), conds)
    if outcomes is None:
        return models, None, None
    by_pair = {k: g.set_index("condition_id")["yield"] for k, g in outcomes.groupby(["screen", "subpair"])}
    usable = []
    for c in T.itertuples():
        s, prof, conds = models[c.screen]
        if c.canonical_a not in prof[1] or c.canonical_b not in prof[2]:
            continue
        usable.append(c.subpair)
        z = by_pair[(c.screen, c.subpair)]
        y = np.array([z.get(cid, np.nan) for cid in conds], float)
        r = pe.score(y, s, prof, c.canonical_a, c.canonical_b, roles[c.screen])
        if r:
            own = c.canonical_a if roles[c.screen] == 1 else c.canonical_b
            L.append(dict(screen=c.screen, subpair=c.subpair, electrophile=own,
                          sse_e=r["sse_e"], sse_n=r["sse_n"], sst=r["sst"]))
    L = pd.DataFrame(L)
    if L.empty:
        raise SystemExit("E2 matched zero reserve targets")
    return models, L, pe.summarise(L, NBOOT, rng)


# ------------------------------------------------------------------ E3: the three-condition decision
def decision(train, T, outcomes=None):
    """Top three by training mean plus both fitted profiles, against the complementary fixed trio."""
    fits = {k: v.fit_array(A) for k, A in history.partner_arrays(train).items()}
    plan = []
    for c in T.itertuples():
        fit = fits.get(c.screen)
        if fit is None:
            continue
        try:
            rec = v.recommend(fit, c.canonical_a, c.canonical_b)
        except ValueError:
            continue                                   # neither block has two training partners, or pair is in training
        plan.append(dict(screen=c.screen, subpair=c.subpair, conditions=fit["conditions"],
                         history=[fit["conditions"][j] for j in rec["history"]],
                         complementary=[fit["conditions"][j] for j in rec["complementary"]],
                         depth_a=rec["depth_a"], depth_b=rec["depth_b"]))
    if not plan:
        raise SystemExit("E3 planned zero reserve targets")
    if outcomes is None:
        return plan, None, None
    by_pair = {k: g for k, g in outcomes.groupby(["screen", "subpair"])}
    rows = []
    for p in plan:
        z = by_pair[(p["screen"], p["subpair"])]
        z = z[~z.duplicated("condition_id", keep=False)].set_index("condition_id")["yield"]
        y = np.array([z.get(cid, np.nan) for cid in p["conditions"]], float)
        idx = {cid: j for j, cid in enumerate(p["conditions"])}
        hp = np.array([idx[c] for c in p["history"]]); cp = np.array([idx[c] for c in p["complementary"]])
        r = v.score_panel(y, hp, cp)
        if r is None:
            continue
        rows.append(dict(screen=p["screen"], subpair=p["subpair"], **r,
                         product_forming=bool(np.nanmax(y) >= anatomy.FORMS),
                         building_block_a=p["subpair"].split(" >> ")[0], building_block_b=p["subpair"].split(" >> ")[1]))
    D = pd.DataFrame(rows)
    if D.empty:
        raise SystemExit("E3 scored zero reserve targets")
    return plan, D, dict(all_pairs=v.summarize_effects(D),
                         product_forming=v.summarize_effects(D[D.product_forming]))


# ------------------------------------------------------------------ power, on analysis pairs only
def power(reps):
    """Emulate E2 and E3 on held-out analysis pairs at the reserve target counts. Reads no reserve outcome."""
    train = training(); roles = owner_roles(); rng = np.random.default_rng(SEED)
    T = reserve_targets(reserve())
    counts = T.groupby("screen").size()
    mats = {k: pe.array_matrix(train, k) for k in counts.index}
    arrays = {k: A for k, A in history.partner_arrays(train).items() if k in set(counts.index)}
    e2, e3 = [], []
    for rep in range(reps):
        L, G = [], []
        for k, n in counts.items():
            Y, s1, s2, _ = mats[k]
            # hold out only as many pairs as the reserve supplies, so training stays as close as the
            # analysis set allows to the real design (full array trains, reserve pairs are extra targets).
            # Training is still smaller than the real test's, so the resulting power is a lower bound.
            hold = np.zeros(len(Y), bool)
            hold[rng.permutation(len(Y))[:min(n, int(.4 * len(Y)))]] = True
            tr = ~hold
            s, prof = pe.train(Y[tr], s1[tr], s2[tr])
            cand = [i for i in np.flatnonzero(hold) if s1[i] in prof[1] and s2[i] in prof[2]]
            for i in rng.permutation(cand)[:n]:
                r = pe.score(Y[i], s, prof, s1[i], s2[i], roles[k])
                if r:
                    L.append(dict(screen=k, electrophile=(s1 if roles[k] == 1 else s2)[i], **r))
            # E3 is emulated through the frozen decision path itself, so that the comparator here is the
            # same greedy complementary trio the real test uses, not a top-of-the-mean stand-in.
            A = arrays[k]; sub = {**A, "M": A["M"][tr], "a": A["a"][tr], "b": A["b"][tr]}
            sub["ua"], sub["ub"] = sorted(set(sub["a"])), sorted(set(sub["b"]))
            sub["ia"] = np.array([sub["ua"].index(x) for x in sub["a"]])
            sub["ib"] = np.array([sub["ub"].index(x) for x in sub["b"]])
            fit = v.fit_array(sub)
            for i in rng.permutation(np.flatnonzero(hold))[:n]:
                try:
                    rec = v.recommend(fit, canonical(A["a"][i]), canonical(A["b"][i]))
                except ValueError:
                    continue
                q = v.score_panel(A["M"][i], rec["history"], rec["complementary"])
                if q:
                    G.append(dict(screen=k, building_block_a=A["a"][i], **q))
        if L:
            r2 = pe.summarise(pd.DataFrame(L), 400, rng)
            e2.append(dict(rep=rep, D=r2["D"], lo_cluster=r2["ci_electrophile_cluster"][0],
                           lo_whole=r2["ci_whole_array"][0], targets=r2["targets"]))
        if G:
            r3 = v.summarize_effects(pd.DataFrame(G), nboot=400)
            e3.append(dict(rep=rep, mean_pp=r3["mean_pp"], lo=r3["ci_low"], targets=r3["n_targets"]))
    return pd.DataFrame(e2), pd.DataFrame(e3)


def power_ownership(reps, draws=400):
    """Can a reserve-sized subsample of one array recover that array's owner? Analysis pairs only."""
    train = training(); roles = owner_roles(); rng = np.random.default_rng(SEED)
    T = reserve_targets(reserve())
    pf_counts = {}
    for key, A in history.partner_arrays(reserve()).items():
        pf_counts[key] = int((np.nanmax(A["M"], 1) >= anatomy.FORMS).sum())
    rows = []
    for key, A in history.partner_arrays(train).items():
        n = pf_counts.get(key, 0)
        if n < MIN_PAIRS_E1:
            continue
        M = A["M"]; keep = np.flatnonzero(np.nanmax(M, 1) >= anatomy.FORMS)
        own = roles[key]
        hits = []
        for _ in range(reps):
            sel = rng.permutation(keep)[:n]
            E, _, _ = anatomy.pattern(M[sel]); a, b = A["a"][sel], A["b"][sel]
            S = anatomy.Shares(E, a, b)
            d = np.empty(draws)
            for t in range(draws):
                q = S.compute(anatomy.molecule_weights(a, b, rng))
                d[t] = q[str(own)] - q[str(3 - own)]
            hits.append(np.nanmean(d > 0) >= .95)
        rows.append(dict(screen=key, reserve_product_forming=n, reps=reps, power=float(np.mean(hits))))
    if not rows:
        raise SystemExit("no array has enough reserve product-forming pairs for E1")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ entry point
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["power", "freeze", "score"])
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    if a.mode == "power":
        P2, P3 = power(a.reps)
        P1 = power_ownership(max(10, a.reps // 4))
        P2.to_csv(OUT / "power_record_transfer.csv", index=False)
        P3.to_csv(OUT / "power_decision.csv", index=False)
        P1.to_csv(OUT / "power_ownership.csv", index=False)
        out = dict(reps=int(a.reps), e1=json.loads(P1.to_json(orient="records")),
                   e2=dict(median_D=float(P2.D.median()), targets=float(P2.targets.median()),
                           power_cluster=float((P2.lo_cluster > 0).mean()), power_whole=float((P2.lo_whole > 0).mean())),
                   e3=dict(median_pp=float(P3.mean_pp.median()), targets=float(P3.targets.median()),
                           power=float((P3.lo > 0).mean())))
        (OUT / "power.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        return
    train = training(); roles = owner_roles(); R = reserve(); T = reserve_targets(R)
    if a.mode == "freeze":
        T.to_csv(OUT / "reserve_target_manifest.csv", index=False)
        plan, _, _ = decision(train, T)
        pd.DataFrame([{k: (json.dumps(x) if isinstance(x, list) else x) for k, x in p.items()} for p in plan]
                     ).to_csv(OUT / "training_only_predictions.csv", index=False)
        print(f"frozen: {len(T)} reserve targets, {len(plan)} with a recommendable trio")
        return
    # Outcomes are read only here, after the manifest and the predictions exist.
    O = R[["screen", "subpair", "condition_id", "yield"]]
    E1 = ownership(R, roles, rng)
    _, L2, S2 = record_transfer(train, T, roles, rng, outcomes=O)
    _, D3, S3 = decision(train, T, outcomes=O)
    E1.to_csv(OUT / "ownership_reserve.csv", index=False)
    L2.to_csv(OUT / "record_transfer_losses.csv", index=False)
    D3.to_csv(OUT / "decision_targets.csv", index=False)
    summary = dict(reserve_rows=int(len(R)), reserve_pairs=int(T.subpair.nunique()),
                   arrays=int(T.screen.nunique()), e1=json.loads(E1.to_json(orient="records")),
                   e2=S2, e3=S3)
    (OUT / "reserve_validation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(dict(e1=[r for r in summary["e1"] if r["eligible"]], e2=S2, e3=S3), indent=2)[:4000])


if __name__ == "__main__":
    main()
