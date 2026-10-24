"""Match this article's released measurements to the published Zhang et al. amide-coupling collection.

Run without --out to check the released record keys and flags. Supply --zhang to recompute the match flags;
with --out DIR, write the flags and the Zhang-only counts to that directory.
"""

N = dict(P2=244179, Zhang=47015)

import argparse

import ast

import gzip

import hashlib

import io

import itertools

import json

import platform

from collections import Counter

from pathlib import Path

import numpy as np

import pandas as pd

import rdkit

from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "05_DATA/Release/00_RAW_IMMUTABLE"

DATA = ROOT / "05_DATA"

OUT = ROOT / "07_METHODS_AND_PROVENANCE/provenance_public"

RULE = OUT / "PROV-1+A2_PUBLIC_RULE.md"

VERSION = "PROV-1+A2"

ZHANG = dict(repository="github.com/aichemeco/amide_coupling", commit="f6fdbe283fc24af53389aab8dcba5e46386ac9c0",
             path="data/all_HTE_with_condition.csv", git_blob="778db5176ed6a1b3b11cbb80da4eacbd40ea74c5", bytes=17657403,
             sha256="5085e1ae50ee26173bd170a4b10a99c4e3101a46b706d115e425aff17c091b41",
             citation="Zhang, C. et al. Chem. Sci. 16, 11809-11822 (2025). doi:10.1039/d5sc03364k")

UNITS = dict(A1=177039, A3=74, SA1=36499, SB1=30567)

NAMES = [f"Reagent{i}_Standardized Name" for i in range(1, 7)]

SEP = "\x1f"

RDLogger.DisableLog("rdApp.*")

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
_can, _mol = {}, {}

def canon(s):
    if not isinstance(s, str):
        return None
    if s not in _can:
        m = Chem.MolFromSmiles(s)
        _mol[s], _can[s] = m, (Chem.MolToSmiles(m, isomericSmiles=True) if m is not None else None)
    return _can[s]

def yield_key(t, two_stage=True):
    """The stored yield rounded to nine and then six decimals; two_stage=False rounds once, for the guard only."""
    try:
        v = float(t)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    return round(round(v, 9) if two_stage else v, 6) + 0.0

def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def key(lib, label, s1, s2, y):
    return None if any(x is None for x in (lib, label, s1, s2, y)) else SEP.join((lib, label, s1, s2, repr(y)))

def condition_table(path):
    C = pd.read_csv(path, dtype=str)
    assert C.groupby("rxn_class").reaction_type.nunique().max() == 1 and C.groupby(["reaction_type", "condition_id"]).size().max() == 1
    C["names"] = [tuple(x for x in r if isinstance(x, str) and x.strip()) for r in C[NAMES].itertuples(index=False)]
    return C.set_index(["reaction_type", "condition_id"]), dict(C.drop_duplicates("rxn_class")[["rxn_class", "reaction_type"]].to_numpy())

def role_table():
    src = (ROOT / "06_CODE/Core/00_common/ctb/history.py").read_text(encoding="utf-8")
    return next(ast.literal_eval(n.value) for n in ast.parse(src).body
                if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "ROLE" for t in n.targets))

def share(smiles, screens, ROLE, role):
    """Share of the distinct molecules matching the role's SMARTS under any of the screens."""
    pats = [Chem.MolFromSmarts(ROLE[s][role][1]) for s in screens]
    ms = [_mol[s] for s in set(smiles) if canon(s) is not None]
    return float(np.mean([any(m.HasSubstructMatch(p) for p in pats) for m in ms])) if ms else 0.0

def released(T, lib_of):
    """P2: the released measurements, unit by unit (A1, A3, SA1, SB1) in released order."""
    A = pd.read_csv(RAW / "DATASET_analysis.csv.gz", dtype=str)
    y = pd.to_numeric(A["yield"], errors="coerce")
    process = A.screen.str.contains("CMC", regex=False)                  # process-development arrays, not released
    dup = (A.screen == "Hiyama|1") & (A.condition_id == "C46")
    SA = pd.read_csv(DATA / "Validation_Set_A/never_analysed_reserve.csv.gz", dtype=str)
    sk = SA.screen + "|" + SA.subpair
    tl = pd.read_csv(DATA / "Results/reserve_validation/record_transfer_losses.csv", dtype=str)
    best = pd.to_numeric(SA["yield"], errors="coerce").groupby(sk).max()
    dt = pd.read_csv(DATA / "Results/reserve_validation/decision_targets.csv", dtype=str)
    keep = (set(tl.screen + "|" + tl.subpair) | set(dt.screen + "|" + dt.subpair)
            | {k for k, v in best.items() if k.startswith("Acylation|1|") and v >= 0.10})
    units = dict(A1=A[y.notna() & ~dup & ~process], A3=A[dup], SA1=SA[sk.isin(keep)],
                 SB1=pd.read_csv(DATA / "Validation_Set_B/never_opened_cross_library.csv.gz", dtype=str))
    for u, d in units.items():
        assert len(d) == UNITS[u], (u, len(d))
    P = pd.concat([d.assign(unit=u) for u, d in units.items()], ignore_index=True)
    assert len(P) == N["P2"] and pd.to_numeric(P["yield"], errors="coerce").notna().all()
    lib = P.rxn_class.map(lib_of)
    rows = T.reindex(list(zip(lib, P.condition_id)))
    ok = rows.names.notna().to_numpy()
    for a, b in [(f"reagent{i}_id", f"Reagent{i}_id") for i in range(1, 7)] + [("solvent_id", "solvent_id")]:
        ok &= P[a].fillna("").to_numpy() == rows[b].fillna("").to_numpy()
    for a in ("temp_id", "time_id"):
        u, v = pd.to_numeric(P[a], errors="coerce").to_numpy(), pd.to_numeric(rows[a], errors="coerce").to_numpy()
        ok &= (u == v) | (np.isnan(u) & np.isnan(v))
    P["lib"], P["label"], P["ok"] = lib.to_numpy(), P.condition_id, ok
    P["s1"], P["s2"] = P.sub_1_smiles.map(canon), P.sub_2_smiles.map(canon)
    return P

def zhang(path, T, lib_of, ROLE, screens_of):
    """Zhang: the commit's file; conditions resolved through the code bridge, partners in stored order."""
    raw = Path(path).read_bytes()
    raw = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    if hashlib.sha256(raw).hexdigest() != ZHANG["sha256"]:
        raise SystemExit("STOP: the Zhang file is not the file of the pinned commit")
    Z = pd.read_csv(io.BytesIO(raw), dtype=str)
    assert len(Z) == N["Zhang"]
    L = lib_of["Acylation"]
    Z["label"] = "C" + Z.condition_id
    labs = [(r.label, r) for r in Z.drop_duplicates("condition_id").itertuples() if (L, r.label) in T.index]
    roles = ("Activation_ID", "Additive_ID", "Base_ID")
    cand = {}
    for lab, r in labs:
        names, solvent = set(T.loc[(L, lab)].names), T.loc[(L, lab)]["Standardized Name"]
        for k, s in [((c, getattr(r, c)), names) for c in roles] + [(("solvent_id", r.solvent_id), {solvent})]:
            cand[k] = cand[k] & s if k in cand else set(s)
    assign = {}
    while True:                                    # elimination within each component role; no candidate left means "none"
        changed = False
        for k, s in cand.items():
            if k not in assign and len(s) <= 1:
                assign[k], changed = (next(iter(s)) if s else None), True
        for k in cand:
            if k not in assign:
                cand[k] = cand[k] - {v for kk, v in assign.items() if v is not None and kk[0] == k[0]}
        if not changed:
            break

    def agrees(lab, r):
        names = list(T.loc[(L, lab)].names)
        opts = [{assign[k]} if k in assign else cand[k] for k in ((c, getattr(r, c)) for c in roles) if not (k in assign and assign[k] is None)]
        return (len(opts) == len(names) and any(all(n in o for n, o in zip(p, opts)) for p in itertools.permutations(names))
                and assign.get(("solvent_id", r.solvent_id)) == T.loc[(L, lab)]["Standardized Name"])
    agree = {lab for lab, r in labs if agrees(lab, r)}
    stored_order = share(Z.sub_1_smiles, screens_of[L], ROLE, 0) >= 0.95 and share(Z.sub_2_smiles, screens_of[L], ROLE, 1) >= 0.95
    Z["lib"], Z["ok"] = L, Z.label.isin(agree) & stored_order
    Z["s1"], Z["s2"] = Z.sub_1_smiles.map(canon), Z.sub_2_smiles.map(canon)
    return Z

def flags(X, Y):
    """For each record of X, whether it is among the first n(Y) records of its key, X in released order."""
    rank = X.groupby("key", sort=False).cumcount()
    return (X.key.notna() & (rank < X.key.map(Y.key.dropna().value_counts()).fillna(0))).to_numpy()

def guard(datasets):
    """Records of two datasets joined by the two-stage rounding but not by a single rounding: stored yields <= 1e-9 apart."""
    A = pd.concat([pd.DataFrame({"ds": n, "k": D.key, "k1": D.key_single, "y": pd.to_numeric(D["yield"], errors="coerce")})
                   for n, D in datasets.items()], ignore_index=True).dropna(subset=["k"])
    worst, pairs = 0.0, 0
    for _, g in A[A.groupby("k").k1.transform("nunique") > 1].groupby("k"):
        for x, z in itertools.combinations(list(g.itertuples(index=False)), 2):
            if x.ds != z.ds and x.k1 != z.k1:
                pairs, worst = pairs + 1, max(worst, abs(x.y - z.y))
    if worst > 1e-9:
        raise SystemExit(f"STOP: the two-stage rounding joins stored yields {worst:.3g} apart")
    return dict(record_pairs_joined_by_the_two_stage_rounding=pairs, max_abs_stored_yield_difference=worst, threshold=1e-9)


RECORD_COLUMNS = ["unit", "screen", "condition_id", "sub_1_smiles", "sub_2_smiles", "yield"]
PUBLIC_COLUMNS = RECORD_COLUMNS + ["matches_zhang2025", "match_rule_version"]
COUNTS_FILE = "P2_ZHANG_OVERLAP_COUNTS_PUBLIC.json"


def keyed(D):
    for col, two in (("key", True), ("key_single", False)):
        D[col] = [key(l, c, a, b, yield_key(t, two)) if ok else None
                  for l, c, a, b, t, ok in zip(D.lib, D.label, D.s1, D.s2, D["yield"], D.ok)]
    return D


def inputs(conditions):
    T, lib_of = condition_table(conditions)
    roles = role_table()
    screens_of = {}
    for s in roles:
        screens_of.setdefault(lib_of[s.split("|")[0]], []).append(s)
    return T, lib_of, roles, screens_of


def main(out=OUT, zhang_file=RAW / "public_amide_reference.csv.gz", conditions=RAW / "DATASET_conditions.csv.gz"):
    T, lib_of, roles, screens_of = inputs(conditions)
    P = keyed(released(T, lib_of))
    Z = keyed(zhang(zhang_file, T, lib_of, roles, screens_of))
    fz = flags(P, Z)
    zcounts = Counter(Z.key.dropna())
    overlap = sum(min(n, zcounts.get(k, 0)) for k, n in Counter(P.key.dropna()).items())
    assert int(fz.sum()) == overlap
    rec = P[RECORD_COLUMNS].copy()
    rec["matches_zhang2025"], rec["match_rule_version"] = fz, VERSION
    paths = ["05_DATA/Release/00_RAW_IMMUTABLE/DATASET_analysis.csv.gz",
             "05_DATA/Validation_Set_A/never_analysed_reserve.csv.gz",
             "05_DATA/Validation_Set_B/never_opened_cross_library.csv.gz",
             "05_DATA/Results/reserve_validation/record_transfer_losses.csv",
             "05_DATA/Results/reserve_validation/decision_targets.csv"]
    result = dict(
        measurements=len(rec), zhang_overlap=int(fz.sum()),
        rule=dict(version=VERSION, text=RULE.relative_to(ROOT).as_posix(), text_sha256=sha(RULE),
                  history="PROV-1 was frozen before counting; Amendment 2 corrected two mechanical matching defects after the first run. "
                          "The final counts use PROV-1+A2."),
        environment=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__, rdkit=rdkit.__version__),
        inputs=dict(P2={f: sha(ROOT / f) for f in paths},
                    condition_dictionary={"05_DATA/Release/00_RAW_IMMUTABLE/DATASET_conditions.csv.gz": sha(conditions)},
                    role_table={"06_CODE/Core/00_common/ctb/history.py": sha(ROOT / "06_CODE/Core/00_common/ctb/history.py")},
                    Zhang=ZHANG),
        cross_check=dict(per_record_flags_reproduce_overlap=True, yield_guard=guard(dict(P2=P, Zhang=Z))))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rec.to_csv(out / "P2_RECORD_PROVENANCE.csv", index=False, lineterminator="\n")
    (out / COUNTS_FILE).write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(measurements=len(rec), zhang_overlap=int(fz.sum()), result="PASS"), indent=1))
    return result


def check(zhang_file=None, released_file=OUT / "P2_RECORD_PROVENANCE.csv", conditions=RAW / "DATASET_conditions.csv.gz"):
    """Check record order and optionally recompute the Zhang flags, without writing files."""
    T, lib_of, roles, screens_of = inputs(conditions)
    P = released(T, lib_of)
    R = pd.read_csv(released_file, dtype=str, keep_default_na=False)
    rows_equal = len(R) == len(P) and bool((R[RECORD_COLUMNS].to_numpy() == P[RECORD_COLUMNS].to_numpy()).all())
    schema_equal = list(R) == PUBLIC_COLUMNS
    pc = json.loads((Path(released_file).parent / COUNTS_FILE).read_text(encoding="utf-8"))
    flags_valid = R.matches_zhang2025.isin(["True", "False"]).all()
    count_equal = pc["measurements"] == len(R) and pc["zhang_overlap"] == int((R.matches_zhang2025 == "True").sum())
    version_equal = set(R.match_rule_version) == {pc["rule"]["version"]} == {VERSION}
    rule_equal = pc["rule"]["text_sha256"] == sha(RULE)
    out = dict(measurements=len(P), rows_equal_to_released_flags_file=rows_equal,
               public_schema_equal=schema_equal, conditions_resolved=int(P.ok.sum()),
               flags_valid=bool(flags_valid), counts_equal=bool(count_equal), rule_version_equal=version_equal,
               rule_text_equal=rule_equal)
    if zhang_file:
        Z = keyed(zhang(zhang_file, T, lib_of, roles, screens_of))
        fz = flags(keyed(P), Z)
        out.update(zhang_flags_equal=bool(((R.matches_zhang2025 == "True").to_numpy() == fz).all()),
                   zhang_overlap=int(fz.sum()))
    else:
        out["zhang"] = "no reference file supplied; flags checked against the released counts"
    ok = (rows_equal and schema_equal and flags_valid and count_equal and version_equal and rule_equal
          and out["conditions_resolved"] == N["P2"] and out.get("zhang_flags_equal", True))
    out["result"] = "PASS" if ok else "FAIL"
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zhang", help="the pinned all_HTE_with_condition.csv, plain or gzip")
    ap.add_argument("--out", help="write recomputed flags and counts to this directory (requires --zhang)")
    a = ap.parse_args()
    if a.out:
        if not a.zhang:
            ap.error("--out requires --zhang")
        main(a.out, a.zhang)
    else:
        raise SystemExit(0 if check(a.zhang)["result"] == "PASS" else 1)
