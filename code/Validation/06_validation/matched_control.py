"""Additional retrospective control. Never writes to the baseline analysis or manuscript."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
sys.path[:0] = [str(WORK / "06_CODE/Core/00_common"), str(WORK / "06_CODE/Core/00_common/ctb")]
from common import data_file, five_folds, load_analysis, stream, top_k
from history import ROLE, partner_arrays, profile


def read_config():
    config_path=HERE / "analysis_config.json"
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != "44db94208d0b37247f357dee4763ff781245a3a9a63e5349c186d06c6e331170":
        raise ValueError("Frozen matched-control definition changed")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["folds"] == 5 and cfg["panel_size"] == 3
    assert cfg["roles"] == ["p1", "p2"] and cfg["target_complete_conditions"] is True
    assert cfg["history_count_difference"] == 0 and cfg["target_min_valid_repeats"] == 1
    assert cfg["controls"] == ["unmatched", "primary", "strict", "activity", "structure", "history2"]
    return cfg


def fit_role(M, idx, n_ids, tr, shrink=4.0):
    """All quantities below use training rows only; evaluation is a separate function."""
    train = np.asarray(M, float)[tr]
    ids = np.asarray(idx)[tr]
    finite = np.isfinite(train)
    colm = np.nanmean(train, axis=0)
    if not np.isfinite(colm).all():
        raise ValueError("A condition has no training measurements")
    gm = float(np.nanmean(train))
    rowmean = np.nanmean(train, axis=1)
    R = train - rowmean[:, None] - colm[None, :] + gm
    U = profile(R, ids, n_ids, np.arange(len(tr)), k=shrink)
    counts = np.bincount(ids, minlength=n_ids)
    observed = np.zeros((n_ids, train.shape[1]), dtype=int)
    np.add.at(observed, ids, finite.astype(int))
    activity = np.full(n_ids, np.nan)
    known = counts > 0
    activity[known] = np.bincount(ids, weights=rowmean, minlength=n_ids)[known] / counts[known]
    scores = U + colm[None, :]
    return dict(counts=counts, observed=observed, activity=activity, colm=colm,
                profiles=U, scores=scores,
                panels=np.stack([top_k(s) for s in scores]), baseline=top_k(colm))


def matching_diagnostics(target, fit):
    obs = fit["observed"]
    support = obs > 0
    intersection = np.sum(support & support[target], axis=1)
    union = np.sum(support | support[target], axis=1)
    jaccard = np.divide(intersection, union, out=np.zeros(len(obs), float), where=union > 0)
    delta = np.abs(obs - obs[target])
    depth = int(fit["counts"][target])
    return dict(jaccard=jaccard, max_count_delta=delta.max(axis=1),
                normalized_l1=delta.sum(axis=1) / max(1, depth * obs.shape[1]),
                support_count_delta=np.abs(support.sum(axis=1) - support[target].sum()),
                activity_delta=fit["activity"] - fit["activity"][target])


def primary_donors(target, fit, cfg):
    d = matching_diagnostics(target, fit)
    counts = fit["counts"]
    ok = ((counts > 0) & (counts == counts[target]) &
          (d["jaccard"] >= cfg["support_jaccard_min"]) &
          (d["support_count_delta"] <= cfg["covered_condition_count_difference_max"]) &
          (d["max_count_delta"] <= cfg["per_condition_count_difference_max"]) &
          (d["normalized_l1"] <= cfg["normalized_count_l1_max"] + 1e-12))
    ok[target] = False
    return np.flatnonzero(ok)


def control_donors(target, fit, similarity, cfg):
    counts = fit["counts"]
    if counts[target] == 0:
        return {k: np.array([], dtype=int) for k in cfg["controls"]}
    unmatched = np.flatnonzero((counts > 0) & (np.arange(len(counts)) != target))
    primary = primary_donors(target, fit, cfg)
    strict = unmatched[(counts[unmatched] == counts[target]) &
                       np.all(fit["observed"][unmatched] == fit["observed"][target], axis=1)]
    activity = primary[np.abs(fit["activity"][primary] - fit["activity"][target]) <=
                       cfg["activity_mean_yield_caliper"] + 1e-12]
    close = primary[similarity[target, primary] >= cfg["structure_tanimoto_min"]]
    structural = (close[np.isclose(similarity[target, close], similarity[target, close].max(),
                                   atol=1e-12, rtol=0)] if len(close) else close)
    return dict(unmatched=unmatched, primary=primary, strict=strict, activity=activity,
                structure=structural, history2=primary if counts[target] >= 2 else primary[:0])


def score_comparison(y, own_panel, donor_panels, baseline_panel):
    """Only complete target response profiles are eligible. Donors are already frozen."""
    y = np.asarray(y, float)
    if not np.isfinite(y).all():
        raise ValueError("Primary analysis requires a fully measured target")
    donor_panels = np.asarray(donor_panels, int)
    if not len(donor_panels):
        raise ValueError("Cannot score an empty donor set")
    own = float(y[own_panel].max())
    donor_values = y[donor_panels].max(axis=1)
    base = float(y[baseline_panel].max())
    return dict(own=own, substitute=float(donor_values.mean()), baseline=base,
                difference=float(own - donor_values.mean()),
                own_gain=own - base, substitute_gain=float(donor_values.mean() - base))


def structural_similarities(names, cfg):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=cfg["fingerprint_radius"], fpSize=cfg["fingerprint_bits"])
    molecules = [Chem.MolFromSmiles(s) for s in names]
    if any(m is None for m in molecules):
        raise ValueError("Invalid molecular structure")
    fps = [generator.GetFingerprint(m) for m in molecules]
    return np.asarray([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps])


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_or_record_lock():
    from boundaries import verify_file
    from common import TRAIN
    verify_file(TRAIN)
    return {"training_sha256": hash_file(TRAIN), "config_sha256": hash_file(HERE / "analysis_config.json")}


def summarize_targets(folds):
    keys = ["control", "screen", "role", "pair", "molecule"]
    metrics = ["own", "substitute", "baseline", "difference", "own_gain", "substitute_gain"]
    target = folds.groupby(keys, as_index=False)[metrics].mean()
    extra = folds.groupby(keys).agg(n_repeats=("rep", "nunique"),
                                   mean_donors=("n_donors", "mean"),
                                   min_history=("n_history", "min"),
                                   mean_history=("n_history", "mean")).reset_index()
    return target.merge(extra, on=keys, validate="one_to_one")


def array_bootstrap(z, cfg):
    a = z.groupby("screen")["difference"].agg(["sum", "count"])
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    ix = rng.integers(len(a), size=(cfg["bootstrap_arrays"], len(a)))
    vals = a["sum"].to_numpy()[ix].sum(1) / a["count"].to_numpy()[ix].sum(1)
    return np.quantile(vals, [.025, .975])


def molecule_bootstrap(z, cfg):
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    sums = np.zeros(cfg["bootstrap_molecules"])
    counts = np.zeros_like(sums)
    for _, g in z.groupby("screen"):
        a = g.groupby("molecule")["difference"].agg(["sum", "count"])
        ix = rng.integers(len(a), size=(len(sums), len(a)))
        sums += a["sum"].to_numpy()[ix].sum(1)
        counts += a["count"].to_numpy()[ix].sum(1)
    return np.quantile(sums / counts, [.025, .975])


def summarize_pool(z, cfg, label):
    lo, hi = array_bootstrap(z, cfg)
    ml, mh = molecule_bootstrap(z, cfg)
    return dict(control=label, role=z.role.iloc[0], n_targets=len(z),
                n_arrays=z.screen.nunique(), n_molecules=z.groupby(["screen", "molecule"]).ngroups,
                difference_pp=100*z.difference.mean(), array_ci_low_pp=100*lo, array_ci_high_pp=100*hi,
                molecule_ci_low_pp=100*ml, molecule_ci_high_pp=100*mh,
                own_gain_pp=100*z.own_gain.mean(), substitute_gain_pp=100*z.substitute_gain.mean(),
                equal_array_mean_pp=100*z.groupby("screen").difference.mean().mean(),
                targets_1_repeat=int((z.n_repeats == 1).sum()),
                targets_2_repeats=int((z.n_repeats == 2).sum()),
                targets_3_repeats=int((z.n_repeats == 3).sum()))


def make_summaries(out, cfg):
    folds = pd.read_csv(out / "fold_effects.csv")
    targets = summarize_targets(folds)
    targets.to_csv(out / "target_effects.csv", index=False)
    per = targets.groupby(["control", "screen", "role"]).agg(
        n_targets=("pair", "size"), n_molecules=("molecule", "nunique"),
        difference_pp=("difference", lambda x: 100*x.mean()),
        own_gain_pp=("own_gain", lambda x: 100*x.mean()),
        substitute_gain_pp=("substitute_gain", lambda x: 100*x.mean()),
        mean_history=("mean_history", "mean"), mean_donors=("mean_donors", "mean")).reset_index()
    # Include zero-match array/role/control combinations explicitly.
    population = pd.read_csv(out / "population.csv")
    grid = pd.MultiIndex.from_product([cfg["controls"], population.screen.unique(), cfg["roles"]],
                                      names=["control", "screen", "role"]).to_frame(index=False)
    per = grid.merge(per, how="left", on=["control", "screen", "role"])
    per[["n_targets", "n_molecules"]] = per[["n_targets", "n_molecules"]].fillna(0).astype(int)
    per.to_csv(out / "array_effects.csv", index=False)
    summaries = []
    for (control, role), z in targets.groupby(["control", "role"]):
        summaries.append(summarize_pool(z, cfg, control))
        all_reps = z[z.n_repeats == cfg["repeats"]]
        if len(all_reps):
            summaries.append(summarize_pool(all_reps, cfg, control + "_all3repeats"))
    pd.DataFrame(summaries).to_csv(out / "pooled_effects.csv", index=False)

    # Compare specifications only after aligning the exact target/repeat/fold records.
    common_summaries = []
    join = ["screen", "role", "pair", "molecule", "rep", "fold"]
    paired_details = []
    for other in ["unmatched", "strict", "activity", "structure", "history2"]:
        a = folds[folds.control == "primary"].drop(columns="control")
        b = folds[folds.control == other].drop(columns="control")
        common = a.merge(b, on=join, suffixes=("_primary", "_other"), validate="one_to_one")
        common["other_control"] = other
        paired_details.append(common)
        for specification in ["primary", "other"]:
            cols = join + [f"{x}_{specification}" for x in
                           ["own", "substitute", "baseline", "difference", "own_gain",
                            "substitute_gain", "n_donors", "n_history"]]
            f = common[cols].rename(columns={c: c.removesuffix("_"+specification) for c in cols})
            f["control"] = specification
            t = summarize_targets(f)
            for role, z in t.groupby("role"):
                record = summarize_pool(z, cfg, specification)
                record["intersection_with"] = other
                common_summaries.append(record)
    pd.concat(paired_details, ignore_index=True).to_csv(out / "common_fold_effects.csv.gz", index=False)
    pd.DataFrame(common_summaries).to_csv(out / "common_target_summaries.csv", index=False)
    return targets


def run(out, cfg):
    import datetime
    verify_or_record_lock()
    out.mkdir(parents=True, exist_ok=True)
    arrays = partner_arrays(load_analysis())
    status, effects, population, identities, pairs, fold_assignments = [], [], [], [], [], []
    donor_fields = ["screen", "role", "rep", "fold", "pair", "molecule", "donor",
                    "own_history", "donor_history", "support_jaccard", "max_count_delta",
                    "normalized_l1", "support_count_delta", "activity_delta", "tanimoto",
                    "own_yield", "donor_yield", "baseline_yield"] + ["in_"+k for k in cfg["controls"]]
    profile_fields = ["screen", "role", "rep", "fold", "molecule", "n_history", "activity",
                      "condition_ids", "observed_counts", "profile", "training_condition_mean", "panel"]
    with gzip.open(out / "donor_evaluations.csv.gz", "wt", encoding="utf-8", newline="") as fh, \
            gzip.open(out / "training_profiles.csv.gz", "wt", encoding="utf-8", newline="") as pf:
        dw, pw = csv.DictWriter(fh, donor_fields), csv.DictWriter(pf, profile_fields)
        dw.writeheader(); pw.writeheader()
        for key, A in arrays.items():
            M = A["M"]
            complete = np.isfinite(M).all(axis=1)
            similarities = {}
            for role, names in [("p1", A["ua"]), ("p2", A["ub"])]:
                similarities[role] = structural_similarities(names, cfg)
                identities.extend(dict(screen=key, role=role, molecule=i, smiles=s,
                                       chemical_role=ROLE[key][int(role[-1])-1][0])
                                  for i, s in enumerate(names))
                population.append(dict(screen=key, role=role, total_pairs=len(M),
                                       complete_pairs=int(complete.sum()),
                                       chemical_role=ROLE[key][int(role[-1])-1][0]))
            pairs.extend(dict(screen=key, pair=i, subpair=sp, complete=bool(complete[i]))
                         for i, sp in enumerate(A["pairs"]))
            for rep in range(cfg["repeats"]):
                for fold, te in enumerate(five_folds(len(M), stream(key, cfg["fold_stream"], rep))):
                    tr = np.setdiff1d(np.arange(len(M)), te)
                    fold_assignments.extend(dict(screen=key, rep=rep, fold=fold, pair=int(q)) for q in te)
                    for role, idx, names in [("p1", A["ia"], A["ua"]), ("p2", A["ib"], A["ub"])]:
                        fit = fit_role(M, idx, len(names), tr, cfg["profile_shrinkage"])
                        for molecule in np.flatnonzero(fit["counts"]):
                            pw.writerow(dict(screen=key, role=role, rep=rep, fold=fold, molecule=molecule,
                                             n_history=fit["counts"][molecule], activity=fit["activity"][molecule],
                                             condition_ids="|".join(A["conds"]),
                                             observed_counts="|".join(map(str, fit["observed"][molecule])),
                                             profile="|".join(format(v, ".17g") for v in fit["profiles"][molecule]),
                                             training_condition_mean="|".join(format(v, ".17g") for v in fit["colm"]),
                                             panel="|".join(A["conds"][c] for c in fit["panels"][molecule])))
                        for q in te:
                            target = int(idx[q])
                            depth = int(fit["counts"][target])
                            meta = dict(screen=key, role=role, rep=rep, fold=fold,
                                        pair=int(q), molecule=target)
                            sets = control_donors(target, fit, similarities[role], cfg)
                            for kind, donors in sets.items():
                                reason = ("target_incomplete" if not complete[q] else
                                          "identity_unseen" if depth == 0 else
                                          "no_eligible_donor" if not len(donors) else "scored")
                                status.append(dict(meta, control=kind, n_history=depth,
                                                   n_donors=len(donors), status=reason))
                                if reason != "scored":
                                    continue
                                result = score_comparison(M[q], fit["panels"][target],
                                                          fit["panels"][donors], fit["baseline"])
                                effects.append(dict(meta, control=kind, n_history=depth,
                                                    n_donors=len(donors), **result))
                            if not complete[q] or not depth:
                                continue
                            diag = matching_diagnostics(target, fit)
                            members = {k: set(v.tolist()) for k, v in sets.items()}
                            for donor in sets["unmatched"]:
                                rec = dict(meta, donor=int(donor), own_history=depth,
                                           donor_history=int(fit["counts"][donor]),
                                           support_jaccard=diag["jaccard"][donor],
                                           max_count_delta=int(diag["max_count_delta"][donor]),
                                           normalized_l1=diag["normalized_l1"][donor],
                                           support_count_delta=int(diag["support_count_delta"][donor]),
                                           activity_delta=diag["activity_delta"][donor],
                                           tanimoto=similarities[role][target, donor],
                                           own_yield=float(M[q, fit["panels"][target]].max()),
                                           donor_yield=float(M[q, fit["panels"][donor]].max()),
                                           baseline_yield=float(M[q, fit["baseline"]].max()))
                                rec.update({"in_"+k: int(donor in members[k]) for k in cfg["controls"]})
                                dw.writerow(rec)
            print(f"Prepared comparisons: {key}", flush=True)
    for name, rows in [("population", population), ("identities", identities), ("target_pairs", pairs),
                       ("fold_assignments", fold_assignments), ("fold_status", status), ("fold_effects", effects)]:
        pd.DataFrame(rows).to_csv(out / (name + ".csv"), index=False)
    make_summaries(out, cfg)
    metadata = dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    script_sha256=hash_file(Path(__file__)), config=cfg,
                    baseline_estimators_preserved=True, results_directory=str(out.resolve()))
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    verify_or_record_lock()
    print(pd.read_csv(out / "pooled_effects.csv").query("control == 'primary'").to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    run(args.out, read_config())
