"""Verify this release as it stands: nothing is rebuilt, nothing outside the package is read, nothing is written.

    python verify_release.py [--report FILE]

Checks, each PASS or FAIL:
  1  every file matches CHECKSUMS.sha256, no file is unlisted, and no file is a link (the .git/ folder of a clone is
     repository metadata, not part of the release)
  2  DATA_MANIFEST.csv describes exactly the files of the package, with their sizes and hashes
  3  the four measurement files: 177,039 + 74 + 36,499 + 30,567 = 244,179 records, identifiers unique, yields numeric in 0-1,
     each quantified or zero by rule (a zero by rule is 0.0)
  4  the wells without a yield: each in a released pair, with status 'detected_not_quantified' or 'no_screen_record', none of
     them a released measurement; 378 of them in the fifteen arrays
  5  the condition table holds exactly the 1,192 conditions named by the records; the reagent table exactly the 401 entries
     referenced in their reagent slots; every condition names its solvent
  6  Set A: the 442-pair manifest is the frozen one, the released 420 pairs follow the stated rule, and the recorded hashes
     hold
  7  the provenance flags are aligned record by record with the released measurements and add up to the released counts
  8  every derived table matches its record (inputs, rows, hashes) and carries only released identifiers
  9  every results file is listed with the claim or panel it supports, and no results file holds an array outside the fifteen
 10  every released measurement belongs to a pair with at least one publication use (metadata/PUBLICATION_USE_BY_PAIR.csv)
 11  the numbers of the statements in results/TEXT_CLAIMS.csv follow from the released files
 12  the frozen plans released are the ones the frozen-plan README lists, with the two files it lists beside them, and their
     hash records hold
 13  no text file names an absolute path
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(PKG / "code/release"))
import package_lib as L                                                      # noqa: E402

FIFTEEN = {"Acylation|1", "Buchwald-Hartwig|1", "Buchwald-Hartwig|2", "C-O competitive reactions|1", "Chan-Lam|1", "Esterification|1",
           "Grignard|1", "Heck|1", "Hiyama|1", "Kumada|1", "Reductive-amination|1", "SNAr|1", "Sonogashira|1", "Suzuki|1", "Ullmann|1"}
ABSOLUTE = re.compile(r"(?<![A-Za-z])[A-Za-z]:\\\\?[A-Za-z0-9_]|(?<![A-Za-z])[A-Za-z]:/(?!/)|(?<![\w.])/(?:Users|home)/")
TEXT = (".md", ".csv", ".json", ".py", ".txt", ".sha256", ".svg", ".cdxml")


def files():
    return sorted(p for p in PKG.rglob("*") if p.is_file() and "__pycache__" not in p.parts and not p.relative_to(PKG).as_posix().startswith("_rebuild/")
                  and p.relative_to(PKG).parts[0] != ".git")                       # a clone's repository metadata, not part of the release


def c1_checksums():
    listed = dict(reversed(line.split("  ", 1)) for line in (PKG / L.CHECKSUMS).read_text(encoding="utf-8").splitlines())
    bad = [f"{f}: {'missing' if not (PKG / f).is_file() else 'changed'}" for f, h in listed.items() if not (PKG / f).is_file() or L.sha256(PKG / f) != h]
    present = {p.relative_to(PKG).as_posix() for p in files()}
    bad += [f"{f}: not listed" for f in sorted(present - set(listed) - {L.CHECKSUMS})]
    bad += [f"{p.relative_to(PKG).as_posix()}: a link" for p in PKG.rglob("*") if p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction())]
    return bad


def c2_manifest():
    rows = list(csv.DictReader(open(PKG / L.DATA_MANIFEST, encoding="utf-8")))
    present = {p.relative_to(PKG).as_posix() for p in files()} - {L.CHECKSUMS, L.DATA_MANIFEST}
    bad = [f"{f}: not in DATA_MANIFEST.csv" for f in sorted(present - {r["path"] for r in rows})]
    bad += [f"{r['path']}: listed but absent" for r in rows if r["path"] not in present]
    for r in rows:
        p = PKG / r["path"]
        if p.is_file() and (L.sha256(p) != r["sha256"] or p.stat().st_size != int(r["bytes"])):
            bad.append(f"{r['path']}: size or hash differs from DATA_MANIFEST.csv")
    for number in (1, 2):
        expected = f"figure_inputs/figure{number}_structures.csv"
        if expected not in present:
            bad.append(f"{expected}: required reaction structures missing")
    main = [(name, script) for name, script in L.GENERATORS if name.startswith("figure")]
    if [name for name, _ in main] != [f"figure{n}" for n in range(1, 6)]:
        bad.append("exactly five consecutive main-figure generators required")
    if {name for name in present if re.fullmatch(r"code/Figures/figure\d+\.py", name)} != {
            f"code/Figures/figure{n}.py" for n in range(1, 6)}:
        bad.append("main-figure code inventory differs from five canonical entry points")
    expected_main = {f"figures/Main/Figure{n}.{ext}" for n in range(1, 6) for ext in ("pdf", "svg", "png")}
    if {name for name in present if name.startswith("figures/Main/")} != expected_main:
        bad.append("main-figure inventory differs from the five PDF, SVG and PNG sets")
    source_main = {Path(name).parts[1] for name in present if re.match(r"source_data/Figure\d+/", name)}
    if source_main != {f"Figure{n}" for n in range(1, 6)}:
        bad.append("main-figure Source Data inventory differs from five figures")
    for name, script in main:
        number = int(name.removeprefix("figure"))
        required = {f"figures/Main/Figure{number}.{ext}" for ext in ("pdf", "svg", "png")} | {
            f"source_data/Figure{number}/Figure{number}_SourceData.xlsx",
            f"source_data/Figure{number}/Figure{number}_SourceData_provenance.json",
            f"source_data/Figure{number}/Figure{number}_readback.json"}
        bad += [f"{expected}: required main-figure output missing" for expected in sorted(required - present)]
        bad += [f"{expected}: required owned support output missing" for expected in L.SUPPORT_OUTPUTS.get(name, []) if expected not in present]
        provenance = PKG / f"source_data/Figure{number}/Figure{number}_SourceData_provenance.json"
        if provenance.is_file():
            rec = json.loads(provenance.read_text(encoding="utf-8"))
            if rec.get("generated_by") != script:
                bad.append(f"{name}: source record names a different generator")
    for expected in ("figure_inputs/selection_v3/RULE.md", "environment/fonts.json",
                     "metadata/figure_display/Fig4a_folds.csv", "metadata/figure_display/Fig4a_history_draws.csv"):
        if expected not in present:
            bad.append(f"{expected}: required display input or environment record missing")
    for name in L.DISPLAY_TABLES:
        expected = "figure_inputs/v5_reference_tables/" + name
        if expected not in present:
            bad.append(f"{expected}: required retained computed input table missing")
    reference_manifest = PKG / "figure_inputs/v5_reference_tables/MANIFEST.json"
    if reference_manifest.is_file():
        records = json.loads(reference_manifest.read_text(encoding="utf-8"))
        names = [Path(record["reference"]).name for record in records]
        if len(names) != len(set(names)) or set(names) != set(L.DISPLAY_TABLES) - {"MANIFEST.json"}:
            bad.append("retained computed input manifest differs from the explicit input inventory")
        for record in records:
            name = Path(record["reference"]).name
            expected = PKG / "figure_inputs/v5_reference_tables" / name
            if record["reference"] != L.DISPLAY_TABLE_ROOT + "/" + name or not expected.is_file() or L.sha256(expected) != record["sha256"]:
                bad.append(f"{name}: retained computed input differs from its source record")
    return bad


def c4_mask(U, mask):
    bad = []
    cell = ["unit", "screen", "subpair", "condition_id"]
    if "yield" in mask.columns or mask.duplicated(cell).any() or not mask.measurement_status.isin(L.NO_YIELD).all():
        bad.append(f"wells without a yield: columns {list(mask.columns)}, or a well twice, or a status other than the two")
    if (mask.unit == "A1").sum() != 378:
        bad.append(f"{int((mask.unit == 'A1').sum())} wells without a yield in the fifteen arrays, expected 378")
    for u, m in mask.groupby("unit"):
        mine = [U[u]] + ([U["A3"]] if u == "A1" else [])
        meas = set().union(*(set(zip(d.screen, d.subpair, d.condition_id)) for d in mine))
        if set(zip(m.screen, m.subpair, m.condition_id)) & meas:
            bad.append(f"{u}: a well without a yield is also a released measurement")
        if not set(L.pair_key(m)) <= set(L.pair_key(U[u])):
            bad.append(f"{u}: a well without a yield lies outside the released pairs")
    if not set(mask[mask.unit == "A1"].screen) <= FIFTEEN or not set(mask.unit) <= {"A1", "SA1", "SB1"}:
        bad.append("a well without a yield lies outside the fifteen arrays or the released units")
    return bad


def c5_conditions(U, K, C):
    bad = []
    used = set()
    for u in L.UNITS:
        used |= set(zip(U[u].rxn_class, U[u].condition_id))
    have = set(zip(K.rxn_class, K.condition_id))
    if have != used or len(K) != 1192:
        bad.append(f"condition table: {len(K)} rows, {len(used - have)} named conditions missing, {len(have - used)} extra")
    slots = set(K[[f"Reagent{i}_id" for i in range(1, 7)]].to_numpy().ravel()) - {""}
    if slots != set(C["Compound ID"]) or len(C) != 401:
        bad.append(f"reagent table: {len(C)} entries; slots {len(slots)}; missing {len(slots - set(C['Compound ID']))}")
    if (K["Standardized Name"] == "").any() or (K.solvent_id == "").any():
        bad.append("a condition without its solvent")
    return bad


def c6_set_a(U):
    bad = []
    rec = json.loads((PKG / L.SET_A_RELEASE).read_text(encoding="utf-8"))
    sam = L.read_text_table(L.SAM)
    frozen = {Path(f).name: h for h, f in (line.split("  ", 1) for line in (PKG / "frozen_plans/reserve_validation/PREREGISTRATION.sha256").read_text(encoding="utf-8").splitlines() if not line.startswith("#"))}
    if len(sam) != 442 or L.sha256(PKG / L.SAM) != frozen.get("reserve_target_manifest.csv") or rec["frozen_manifest_sha256"] != frozen.get("reserve_target_manifest.csv"):
        bad.append("the 442-pair manifest is not the frozen one")
    if rec["frozen_data_file_sha256"] != frozen.get("never_analysed_reserve.csv.gz"):
        bad.append("the recorded hash of the frozen data file differs from the frozen hash record")
    sa = U["SA1"]
    if L.sha256(PKG / L.UNITS["SA1"][0]) != rec["released_subset"]["sha256"]:
        bad.append("the released subset's hash differs from its record")
    keys = set(L.pair_key(sa))
    transfer = set(L.pair_key(L.read_text_table(L.TRANSFER_TARGETS)))
    best = L.numeric(sa["yield"]).groupby(L.pair_key(sa)).max()
    amide = {k for k, v in best.items() if k.startswith("Acylation|1|") and v >= L.FORMS}
    decision = set(L.pair_key(L.read_text_table(L.DECISION_TARGETS)))
    if not keys <= set(L.pair_key(sam)):
        bad.append("a released pair is not in the frozen manifest")
    if keys != transfer | decision | amide or (len(keys), len(transfer), len(decision), len(amide), len(sa)) != (420, 309, 396, 217, 36499):
        bad.append(f"release rule: {len(keys)} pairs released, {len(transfer)} transfer targets, {len(decision)} decision targets, "
                   f"{len(amide)} product-forming amide pairs")
    if (rec["released_subset"]["pairs"], rec["released_subset"]["records"]) != (len(keys), len(sa)):
        bad.append("the release record's counts differ from the file")
    return bad


def c7_provenance(U):
    bad = []
    F = L.read_text_table(L.PROVENANCE_FLAGS)
    cols = ["unit", "screen", "condition_id", "sub_1_smiles", "sub_2_smiles", "yield"]
    import pandas as pd
    R = pd.concat([U[u][cols] for u in L.UNITS], ignore_index=True)
    if len(F) != len(R) or not (F[cols].to_numpy() == R.to_numpy()).all():
        bad.append("P2_RECORD_PROVENANCE.csv is not aligned record by record with the released measurements")
    pc = json.loads((PKG / L.PROVENANCE_COUNTS).read_text(encoding="utf-8"))
    if list(F) != cols + ["matches_zhang2025", "match_rule_version"]:
        bad.append("the provenance table has an unexpected schema")
    if not F.matches_zhang2025.isin(["True", "False"]).all():
        bad.append("the Zhang flags are not Boolean values")
    tally = dict(measurements=len(F), zhang_overlap=int((F.matches_zhang2025 == "True").sum()))
    if any(pc[k] != v for k, v in tally.items()) or set(F.match_rule_version) != {pc["rule"]["version"]}:
        bad.append(f"flags add up to {tally}, which differs from the released counts or rule version")
    if pc["rule"]["text_sha256"] != L.sha256(PKG / "provenance/PROV-1+A2_PUBLIC_RULE.md"):
        bad.append("the counts record names another version of the rule text")
    return bad


def c9_results():
    bad = []
    M = L.read_text_table(L.RESULTS_MANIFEST)
    listed = set(M.file)
    have = set(L.results_files())
    bad += [f"{f}: results file not in RESULTS_MANIFEST.csv" for f in sorted(have - listed)]
    bad += [f"{f}: listed but absent" for f in sorted(listed - have)]
    for r in M.itertuples():
        if r.file in have and (L.sha256(PKG / r.file) != r.sha256 or not r.claim_or_panel.strip()):
            bad.append(f"{r.file}: hash differs from the manifest or no claim given")
    for f in sorted(have):
        t = (PKG / f).read_text(encoding="utf-8")
        named = set(re.findall(r'"?([A-Za-z][A-Za-z -]*(?:\([A-Za-z]+\))?\|\d)"?', t))
        other = {s for s in named if s not in FIFTEEN and "|" in s and not s.startswith(("http",))}
        if other:
            bad.append(f"{f}: names arrays outside the fifteen: {sorted(other)[:3]}")
    return bad


def c11_claims(U):
    values = L.text_claim_values(U, L.read_text_table(L.MASK), L.read_text_table(L.CONDITIONS), L.read_text_table(L.REAGENTS),
                                 L.read_text_table(L.PROVENANCE_FLAGS),
                                 json.loads((PKG / "source_data/Figure3/Figure3_SourceData_provenance.json").read_text(encoding="utf-8")))
    return L.check_text_claims(values)


def c12_frozen():
    bad = []
    readme = (PKG / "frozen_plans/PUBLIC_FROZEN_PLAN_README.md").read_text(encoding="utf-8")
    allow = {m.group(1): re.findall(r"`([^`]+)`", m.group(2)) for m in re.finditer(r"^\| `([^`/]+)/` \| (.+?) \| .+? \|$", readme, re.M)}
    beside = re.findall(r"^\| `([^`/]+\.(?:md|csv))` \| [^|]+ \|$", readme, re.M)
    want = {f"frozen_plans/{d}/{f}" for d, fs in allow.items() for f in fs} | {"frozen_plans/PUBLIC_FROZEN_PLAN_README.md"}
    want |= {f"frozen_plans/{f}" for f in beside}
    if sorted(f"frozen_plans/{f}" for f in beside) != sorted([L.CORRECTION, L.SET_B_FLAGS]):
        bad.append(f"the README lists {beside} beside the plans, not the correction and the Set B flags")
    have = {p.relative_to(PKG).as_posix() for p in (PKG / "frozen_plans").rglob("*") if p.is_file()}
    if want != have:
        bad.append(f"frozen plans: missing {sorted(want - have)}, not listed {sorted(have - want)}")
    for rec in (PKG / "frozen_plans").rglob("*.sha256"):
        for line in rec.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^([0-9a-f]{64})\s+\*?(\S+)", line)
            if not m:
                continue
            name = Path(m.group(2)).name
            here = rec.parent / name
            if here.is_file() and L.sha256(here) != m.group(1):
                bad.append(f"{rec.relative_to(PKG).as_posix()}: {name} does not have the recorded hash")
    lock = json.loads((PKG / "frozen_plans/matched_control/design_and_input_lock_PUBLIC.json").read_text(encoding="utf-8"))
    byname = {Path(e["file"]).name: e["sha256"] for e in lock["entries"]}
    if byname.get("ANALYSIS_PLAN.md") != L.sha256(PKG / "frozen_plans/matched_control/ANALYSIS_PLAN.md") or \
            byname.get("analysis_config.json") != L.sha256(PKG / "code/Validation/06_validation/analysis_config.json"):
        bad.append("the matched-control plan or its configuration does not have the hash its input lock records")
    return bad


def c13_paths():
    bad = []
    for p in files():
        if p.suffix in TEXT:
            t = p.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(t.splitlines(), 1):
                if ABSOLUTE.search(line):
                    bad.append(f"{p.relative_to(PKG).as_posix()}:{i}: an absolute path")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--report", help="also write the results as JSON to this file")
    a = ap.parse_args()
    U = L.load_units()
    mask, K, C = L.read_text_table(L.MASK), L.read_text_table(L.CONDITIONS), L.read_text_table(L.REAGENTS)
    checks = [("1 checksums", c1_checksums), ("2 data manifest", c2_manifest), ("3 measurements: 244,179 records", lambda: L.check_units(U)),
              ("4 wells without a yield, with their status", lambda: c4_mask(U, mask)),
              ("5 conditions (1,192) and reagent entries (401)", lambda: c5_conditions(U, K, C)), ("6 Set A release rule", lambda: c6_set_a(U)),
              ("7 provenance flags", lambda: c7_provenance(U)),
              ("8 derived tables", lambda: L.check_derived(json.loads((PKG / L.DERIVED_RECORD).read_text(encoding="utf-8")), U)),
              ("9 results", c9_results), ("10 publication use of every measurement", lambda: L.check_publication_use(U)),
              ("11 statements of the text", lambda: c11_claims(U)), ("12 frozen plans", c12_frozen), ("13 no absolute paths", c13_paths)]
    out = []
    for name, fn in checks:
        try:
            bad = fn()
        except Exception as exc:                                  # a check that cannot run has failed
            bad = [f"{type(exc).__name__}: {exc}"]
        out.append(dict(check=name, result="PASS" if not bad else "FAIL", problems=bad))
        print(f"{'PASS' if not bad else 'FAIL'}  {name}" + "".join(f"\n      {x}" for x in bad[:8]))
    ok = all(o["result"] == "PASS" for o in out)
    print(f"verify: {'PASS' if ok else 'FAIL'} ({sum(o['result'] == 'PASS' for o in out)} of {len(out)} checks)")
    if a.report:
        Path(a.report).write_text(json.dumps(dict(result="PASS" if ok else "FAIL", checks=out), indent=1, ensure_ascii=False), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
