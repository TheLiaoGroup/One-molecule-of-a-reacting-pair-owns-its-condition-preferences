"""Retained Set A interval draws, owned by the Figure 4 generator.

The legacy data functions retain their original AST and their own 2,000-draw
namespace. The current Figure 4 history calculations use separate settings.
"""
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "06_CODE/Core/00_common"), str(ROOT / "06_CODE/Core/00_common/ctb"),
                str(ROOT / "06_CODE/Validation/06_validation"), str(ROOT / "06_CODE/Validation/11_partner_asymmetry")]
from figure1_data import public_units
import anatomy
import history
import partner_external as pe

DATA = ROOT / "05_DATA"
SET_A, SET_B = DATA / "Validation_Set_A/never_analysed_reserve.csv.gz", DATA / "Validation_Set_B/never_opened_cross_library.csv.gz"
APPROVED = {"record_transfer_losses.csv": DATA / "Results/reserve_validation/record_transfer_losses.csv",
            "reserve_validation.json": DATA / "Results/reserve_validation/reserve_validation.json",
            "Anatomy_arrays.csv": DATA / "Results/display_inputs/Anatomy_arrays.csv"}
OUT = {"SetA_whole_array_bootstrap_draws.csv": ROOT / "04_SOURCE_DATA/Figure4/SetA_whole_array_bootstrap_draws.csv",
       "SetA_owner_block_bootstrap_draws.csv": ROOT / "04_SOURCE_DATA/Figure4/SetA_owner_block_bootstrap_draws.csv"}
N_SA1, N_SB1 = 29517, 30567
SEED_A, NBOOT, K2_MAX = 20261004, 2000, 45
TOL = 1e-9
sha_text = lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()


def public_sets():
    """SA1 and SB1 as the release defines them; SA1 by its rule (record-transfer targets and product-forming amide pairs)."""
    SA = pd.read_csv(SET_A, low_memory=False)
    key = SA.screen + "|" + SA.subpair
    tl = pd.read_csv(APPROVED["record_transfer_losses.csv"])
    best = pd.to_numeric(SA["yield"], errors="coerce").groupby(key).max()
    keep = set(tl.screen + "|" + tl.subpair) | {k for k, v in best.items() if k.startswith("Acylation|1|") and v >= anatomy.FORMS}
    SA = SA[key.isin(keep)].assign(**{"yield": lambda d: pd.to_numeric(d["yield"], errors="coerce")})
    SB = pd.read_csv(SET_B, low_memory=False)
    SB["yield"] = pd.to_numeric(SB["yield"], errors="coerce")
    assert len(SA) == N_SA1 and len(SB) == N_SB1, (len(SA), len(SB))
    return SA, SB


def models(A1, screens):
    """partner_external's single-role profiles on the analysis rows of each array (A1), with the patterns behind them."""
    out = {}
    for key in screens:
        G = A1[A1.screen == key]
        P = G.pivot(index="subpair", columns="condition_id", values="yield")
        meta = G.drop_duplicates("subpair").set_index("subpair").loc[P.index]
        can = {s: pe.canonical(s) for s in set(meta.sub_1_smiles) | set(meta.sub_2_smiles)}
        Y, s1, s2 = P.to_numpy(float), meta.sub_1_smiles.map(can).to_numpy(), meta.sub_2_smiles.map(can).to_numpy()
        s, prof = pe.train(Y, s1, s2)
        E = pe.slope_residuals(Y, s)
        out[key] = dict(s=s, prof=prof, conds=list(P.columns), s1=s1, s2=s2, E=E, Mk=~np.isnan(E), pairs=len(Y), rows=len(G))
    return out


def subsets_k2(ids):
    c = list(itertools.combinations(range(len(ids)), 2))
    return c if len(c) <= K2_MAX else sorted(c, key=lambda ij: sha_text("\t".join(sorted((ids[ij[0]], ids[ij[1]])))))[:K2_MAX]


def losses(S, M, owner, dedup, set_name):
    rows = []
    for (key, sp), g in S.groupby(["screen", "subpair"], sort=False):
        m = M[key]
        a, b = pe.canonical(g.sub_1_smiles.iloc[0]), pe.canonical(g.sub_2_smiles.iloc[0])
        if a not in m["prof"][1] or b not in m["prof"][2]:
            continue
        z = g.set_index("condition_id")["yield"]
        if dedup:
            z = z[~z.index.duplicated(keep=False)]
        y = np.array([z.get(c, np.nan) for c in m["conds"]], float)
        r = pe.score(y, m["s"], m["prof"], a, b, owner[key])
        if not r:
            continue
        o = ~np.isnan(y) & ~np.isnan(m["s"])
        X = np.c_[np.ones(o.sum()), m["s"][o]]
        proj = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
        e = proj(y[o])
        k2 = {}
        for side, role in (("e", owner[key]), ("n", 3 - owner[key])):
            keys, other = (m["s1"], m["s2"]) if role == 1 else (m["s2"], m["s1"])
            rws = np.flatnonzero(keys == (a if role == 1 else b))
            sse = [float(np.sum((e - proj((np.nansum(m["E"][rws[[i, j]]], 0) / (m["Mk"][rws[[i, j]]].sum(0) + pe.SHRINK))[o])) ** 2))
                   for i, j in subsets_k2(list(other[rws]))]
            k2[f"sse_{side}_k2"], k2[f"subsets_{side}"], k2[f"history_{side}"] = float(np.mean(sse)), len(sse), len(rws)
        rows.append(dict(set=set_name, screen=key, subpair=sp, electrophile=a if owner[key] == 1 else b, **r, **k2))
    return pd.DataFrame(rows)


def check_losses(L, approved, cols):
    J = L.merge(approved, on=["screen", "subpair"], how="outer", suffixes=("", "_approved"), indicator=True)
    assert (J._merge == "both").all() and len(J) == len(approved), J._merge.value_counts().to_dict()
    d = max(float((J[c] - J[c + "_approved"]).abs().max()) for c in cols)
    assert d < TOL and (J.electrophile == J.electrophile_approved).all(), d
    return d


def frozen_run_state(SA):
    """The generator of the frozen Set A run, after the amide ownership estimate's 2,000 molecule resamples."""
    rng = np.random.default_rng(SEED_A)
    for key, A in history.partner_arrays(SA).items():
        keep = np.nanmax(A["M"], 1) >= anatomy.FORMS
        if keep.sum() >= 50:                                                  # the estimate's eligibility rule
            for _ in range(NBOOT):
                anatomy.molecule_weights(A["a"][keep], A["b"][keep], rng)
    return rng


def frozen_draws(LA, S, rng):
    """The frozen Set A run's draws, taken from the state it is given in partner_external.summarise's order: first the whole
    arrays, then, array by array, the owning building blocks within each array. Two tables with one row per draw: the arrays
    drawn, or in each array the owner molecules drawn (';' between them), and the draw's D. The 2.5th and 97.5th percentiles
    of D must be the frozen intervals."""
    L = LA.assign(gain=LA.sse_n - LA.sse_e)
    G = L.groupby("screen")[["gain", "sst"]].sum()                                     # summarise's units, in its order
    ia = rng.integers(0, len(G), (NBOOT, len(G)))
    d = G.gain.to_numpy()[ia].sum(1) / G.sst.to_numpy()[ia].sum(1)
    assert np.array_equal(np.percentile(d, [2.5, 97.5]), S["ci_whole_array"]), (np.percentile(d, [2.5, 97.5]), S["ci_whole_array"])
    whole = pd.DataFrame(G.index.to_numpy()[ia], columns=[f"screen_{j + 1}" for j in range(len(G))])
    num, den, owners = np.zeros(NBOOT), np.zeros(NBOOT), {}
    for key, g in L.groupby("screen"):
        c = g.groupby("electrophile")[["gain", "sst"]].sum()
        assert not any(";" in m for m in c.index)
        ic = rng.integers(0, len(c), (NBOOT, len(c)))
        num += c.gain.to_numpy()[ic].sum(1); den += c.sst.to_numpy()[ic].sum(1)
        owners[key] = [";".join(r) for r in c.index.to_numpy()[ic]]
    e = num / den
    assert np.array_equal(np.percentile(e, [2.5, 97.5]), S["ci_electrophile_cluster"]), (np.percentile(e, [2.5, 97.5]),
                                                                                       S["ci_electrophile_cluster"])
    block = pd.DataFrame(owners)
    for t in (whole, block):
        t.insert(0, "draw", np.arange(1, NBOOT + 1))
    return whole.assign(D=d), block.assign(D=e)


def write_seta_bootstrap(out=None):
    """Recompute both retained CSVs; return paths for the Figure 5 provenance."""
    A1, _ = public_units()
    SA, SB = public_sets()
    owner = pd.read_csv(APPROVED["Anatomy_arrays.csv"]).set_index("screen").owner_role.astype(int).to_dict()
    M = models(A1, sorted(set(SA.screen) | set(SB.screen)))
    LA = losses(SA, M, owner, False, "Set A")
    check_losses(LA, pd.read_csv(APPROVED["record_transfer_losses.csv"]), ["sse_e", "sse_n", "sst"])
    summary = pe.summarise(LA.drop(columns="set"), NBOOT, frozen_run_state(SA))
    reference = json.loads(APPROVED["reserve_validation.json"].read_text(encoding="utf-8"))["e2"]
    for key in ("targets", "arrays", "arrays_positive"):
        assert summary[key] == reference[key], key
    for key in ("D", "r2_electrophile", "r2_partner"):
        assert abs(summary[key] - reference[key]) < TOL, key
    for key in ("ci_whole_array", "ci_electrophile_cluster"):
        assert np.allclose(summary[key], reference[key], atol=1e-12, rtol=0), key
    assert summary["per_array"].keys() == reference["per_array"].keys() and all(
        abs(summary["per_array"][key] - value) < TOL for key, value in reference["per_array"].items())
    whole, block = frozen_draws(LA, summary, frozen_run_state(SA))
    paths = {}
    for name, table in zip(OUT, (whole, block)):
        logical = OUT[name]
        actual = Path(out) / name if out is not None else logical
        actual.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(actual, index=False, lineterminator="\n")
        assert pd.read_csv(actual, float_precision="round_trip").equals(table), name
        paths[name] = {"logical": logical, "actual": actual}
    return paths
