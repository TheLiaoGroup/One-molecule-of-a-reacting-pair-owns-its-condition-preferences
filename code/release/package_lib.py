"""Shared code of the package's two entry points, rebuild_all.py and verify_release.py.

Everything here reads this package alone: the released measurements (data/), their metadata, the derived analysis tables
(derived/), the retained results (results/), the frozen plans, the provenance files, the Source Data and the figures.
"""
import gzip
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[2]
WORK = PKG / "_rebuild"                                  # every file a rebuild writes; never part of the release
UNITS = {"A1": ("data/measurements_A1_analysis_arrays.csv.gz", 177039),
         "A3": ("data/measurements_A3_hiyama_duplicate.csv.gz", 74),
         "SA1": ("data/measurements_SA1_set_A.csv.gz", 36499),
         "SB1": ("data/measurements_SB1_set_B.csv.gz", 30567)}
TOTAL = 244179
MEASURED = ("quantified", "zero_by_rule")                 # the status of a released measurement
NO_YIELD = ("detected_not_quantified", "no_screen_record")  # the status of a well without a yield
MASK, CONDITIONS, REAGENTS = "data/wells_without_yield.csv", "data/conditions.csv", "data/reagents.csv"
SAM, SET_A_RELEASE, DC1 = "metadata/set_A_frozen_manifest.csv", "metadata/set_A_release.json", "metadata/DESIGN_COUNTS.csv"
# beside the frozen plans: the provenance correction of the two out-of-array evaluation sets and its per-pair Set B flags
CORRECTION, SET_B_FLAGS = "frozen_plans/OUT_OF_ARRAY_EVALUATION_PROVENANCE_CORRECTION.md", "frozen_plans/SET_B_DOCUMENTED_PRIOR_SCORING.csv"
USE = "metadata/PUBLICATION_USE_BY_PAIR.csv"
DERIVED_RECORD, RESULTS_MANIFEST, TEXT_CLAIMS = "derived/DERIVED_FILES.json", "results/RESULTS_MANIFEST.csv", "results/TEXT_CLAIMS.csv"
PROVENANCE_FLAGS, PROVENANCE_COUNTS = "provenance/P2_RECORD_PROVENANCE.csv", "provenance/P2_ZHANG_OVERLAP_COUNTS_PUBLIC.json"
TRANSFER_TARGETS, SET_B_TARGETS = "results/reserve_validation/record_transfer_losses.csv", "results/cross_library/target_losses.csv"
DECISION_TARGETS = "results/reserve_validation/decision_targets.csv"
CHECKSUMS, DATA_MANIFEST = "CHECKSUMS.sha256", "DATA_MANIFEST.csv"
ID = re.compile(r"^P2-(A1|A3|SA1|SB1)-(\d{7})$")
FORMS = 0.10                                             # a pair forms product when its best measured yield reaches 10%
COMPOSITION = [f"Reagent{i}_id" for i in range(1, 7)] + ["solvent_id", "temp_id", "time_id"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def read_text_table(rel, **kw):
    """A package table with every field as the text it holds (empty fields as empty strings)."""
    return pd.read_csv(PKG / rel, dtype=str, keep_default_na=False, **kw)


def csv_bytes(df):
    return df.to_csv(index=False, lineterminator="\n").encode("utf-8")


def gz(b):
    """Deterministic gzip: no file name, time stamp zero."""
    return gzip.compress(b, compresslevel=9, mtime=0)


def payload(path):
    """The content of a file without its container: the uncompressed bytes of a .gz file, else the file."""
    b = Path(path).read_bytes()
    return gzip.decompress(b) if str(path).endswith(".gz") else b


# ====================================================================== the released records
def load_units():
    return {u: read_text_table(f) for u, (f, _) in UNITS.items()}


def pair_key(d):
    return d.screen + "|" + d.subpair


def numeric(s):
    return pd.to_numeric(s, errors="coerce")


def check_units(U):
    """Counts, identifiers and the shape of every record; returns a list of problems."""
    bad = []
    for u, (f, n) in UNITS.items():
        d = U[u]
        if len(d) != n:
            bad.append(f"{u}: {len(d)} records, expected {n}")
        if not (d.unit == u).all():
            bad.append(f"{u}: unit column")
        want = [f"P2-{u}-{i:07d}" for i in range(1, len(d) + 1)]
        if d.public_record_id.tolist() != want:
            bad.append(f"{u}: identifiers are not P2-{u}-0000001 onwards in file order")
        y = numeric(d["yield"])
        if y.isna().any() or (y < 0).any() or (y > 1).any():
            bad.append(f"{u}: a yield is missing or outside 0-1")
        if not d.measurement_status.isin(MEASURED).all() or not (d["yield"][d.measurement_status == "zero_by_rule"] == "0.0").all():
            bad.append(f"{u}: a measurement status is not quantified or zero_by_rule, or a zero by rule is not 0.0")
        if not (d.subpair == d.sub_1_smiles + " >> " + d.sub_2_smiles).all():
            bad.append(f"{u}: subpair is not sub_1 >> sub_2")
        if d.duplicated(["screen", "subpair", "condition_id"]).any() and u != "SB1":
            bad.append(f"{u}: a cell is measured twice")
    ids = pd.concat([U[u].public_record_id for u in UNITS])
    if ids.duplicated().any() or not ids.map(lambda s: bool(ID.match(s))).all():
        bad.append("public_record_id not globally unique or malformed")
    if sum(len(U[u]) for u in UNITS) != TOTAL:
        bad.append(f"union of the four files: {sum(len(U[u]) for u in UNITS)} records, expected {TOTAL}")
    return bad


# ====================================================================== task construction: the derived analysis tables
def derived_tables():
    """Name -> (table, inputs, transformation). The analysis code reads the measurements in the layout of the original
    analysis files; these tables are that layout, made from the released records and the condition table alone."""
    U = load_units()
    K = read_text_table(CONDITIONS)
    comp = K.set_index(["rxn_class", "condition_id"])[COMPOSITION]
    assert not comp.index.duplicated().any()

    def analysis(d):
        c = comp.reindex(list(zip(d.rxn_class, d.condition_id)))
        assert len(c) == len(d) and c.notna().all().all(), "a record names a condition missing from the condition table"
        out = d[["public_record_id", "screen", "rxn_class", "condition_id", "subpair", "sub_1_smiles", "sub_2_smiles", "yield"]].copy()
        for i in range(1, 7):
            out[f"reagent{i}_id"] = c[f"Reagent{i}_id"].to_numpy()
        for col in ("solvent_id", "temp_id", "time_id"):
            out[col] = c[col].to_numpy()
        return out.reset_index(drop=True)
    comp_note = ("; each record also carries its condition's reagent, solvent, temperature and time identifiers from "
                 f"{CONDITIONS} (joined on rxn_class and condition_id)")
    return {
        "derived/DATASET_analysis.csv.gz": (
            analysis(pd.concat([U["A1"], U["A3"]], ignore_index=True)), [UNITS["A1"][0], UNITS["A3"][0], CONDITIONS],
            "the A1 records in file order, then the A3 records in file order" + comp_note),
        "derived/set_A_analysis_format.csv.gz": (analysis(U["SA1"]), [UNITS["SA1"][0], CONDITIONS], "the SA1 records in file order" + comp_note),
        "derived/set_B_analysis_format.csv.gz": (analysis(U["SB1"]), [UNITS["SB1"][0], CONDITIONS], "the SB1 records in file order" + comp_note),
        "derived/DATASET_conditions.csv.gz": (K, [CONDITIONS], f"{CONDITIONS} unchanged, gzip-compressed (read by the provenance script)"),
    }


def build_derived(dest):
    """Write the derived tables under dest (a package root or a work directory); returns their records."""
    rec = {}
    for name, (t, inputs, how) in derived_tables().items():
        raw = csv_bytes(t)
        b = gz(raw)
        p = Path(dest) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
        rec[name] = dict(inputs={f: sha256(PKG / f) for f in inputs}, transformation=how, rows=int(len(t)),
                         sha256=sha_bytes(b), content_sha256=sha_bytes(raw))
    return rec


def check_derived(rec, U, where=PKG):
    """Each derived file against its record: input hashes, rows, file and content hashes, and only released identifiers."""
    bad = []
    public = set(pd.concat([U[u].public_record_id for u in UNITS]))
    for name, r in rec.items():
        p = Path(where) / name
        if not p.is_file():
            bad.append(f"{name}: missing")
            continue
        for f, h in r["inputs"].items():
            if sha256(PKG / f) != h:
                bad.append(f"{name}: input {f} changed")
        if sha256(p) != r["sha256"] and sha_bytes(payload(p)) != r["content_sha256"]:
            bad.append(f"{name}: content differs from its record")
        t = pd.read_csv(io.BytesIO(payload(p)), dtype=str, keep_default_na=False)
        if len(t) != r["rows"]:
            bad.append(f"{name}: {len(t)} rows, record says {r['rows']}")
        if "public_record_id" in t and not set(t.public_record_id) <= public:
            bad.append(f"{name}: identifiers outside the released records")
    return bad


# ====================================================================== the work tree: package files where the analysis code expects them
HIDDEN_STATE = ["Supplementary_Figures/FigS7_depth_interval_draws.csv", "Supplementary_Figures/FigS8_transfer_folds.csv",
                "Supplementary_Figures/FigS10_molecule_groups.csv", "Supplementary_Figures/FigS10_condition_folds.csv",
                "Supplementary_Figures/FigS10_pair_halves.csv",
                "Supplementary_Figures/FigS10_SourceData.xlsx"]      # the last is checked by the ED10 re-fit, not read by a figure
DERIVED_AT = {"derived/DATASET_analysis.csv.gz": "05_DATA/Release/00_RAW_IMMUTABLE/DATASET_analysis.csv.gz",
              "derived/DATASET_conditions.csv.gz": "05_DATA/Release/00_RAW_IMMUTABLE/DATASET_conditions.csv.gz",
              "derived/set_A_analysis_format.csv.gz": "05_DATA/Validation_Set_A/never_analysed_reserve.csv.gz",      # the original file names
              "derived/set_B_analysis_format.csv.gz": "05_DATA/Validation_Set_B/never_opened_cross_library.csv.gz"}
DISPLAY_TABLES = ["overview.xlsx", "preference.xlsx", "screen_shares.xlsx", "molecule_context.xlsx", "transfer.xlsx",
                  "preference_payload.json", "MANIFEST.json"]
DISPLAY_TABLE_ROOT = "05_DATA/Analysis/figure_display/v5_reference_tables"


def results_files():
    return sorted(p.relative_to(PKG).as_posix() for p in (PKG / "results").rglob("*")
                  if p.is_file() and p.relative_to(PKG).as_posix() not in (RESULTS_MANIFEST, TEXT_CLAIMS))


def stage_map(derived_dir):
    """(source file, path in the work tree). The code is the code under code/ as released; the measurements are the derived
    tables made by this run (derived_dir)."""
    m = [(Path(derived_dir) / k, v) for k, v in DERIVED_AT.items()]
    m += [(PKG / DC1, "05_DATA/Analysis/directional_cohorts/DESIGN_COUNTS.csv")]
    m += [(PKG / "metadata/figure_display" / name, "05_DATA/Analysis/figure_display/" + name)
          for name in ("Fig4a_folds.csv", "Fig4a_history_draws.csv")]
    m += [(PKG / "figure_inputs/v5_reference_tables" / name, DISPLAY_TABLE_ROOT + "/" + name)
          for name in DISPLAY_TABLES]
    m += [(PKG / "figure_inputs/selection_v3/RULE.md", "07_METHODS_AND_PROVENANCE/frozen_analysis_plans/figure_selection_v3/RULE.md")]
    m += [(PKG / CONDITIONS, "10_PUBLIC_RELEASE_CANDIDATE/data/conditions.csv")]
    m += [(PKG / f"figure_inputs/figure{n}_structures.csv",
           f"07_METHODS_AND_PROVENANCE/figure_inputs/figure{n}_structures.csv") for n in (1, 2)]
    m += [(PKG / r, "05_DATA/Results/" + r[len("results/"):]) for r in results_files()]
    m += [(PKG / "source_data" / h, "04_SOURCE_DATA/" + h) for h in HIDDEN_STATE]
    structures = PKG / "figures/structures"
    for p in sorted(structures.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(structures).as_posix()
        dest = "07_METHODS_AND_PROVENANCE/" + rel if rel == "FIGURE_STRUCTURES_PROMOTION_READY.csv" else "02_FIGURES/ChemDraw/" + rel
        m.append((p, dest))
    for p in sorted((PKG / "code").rglob("*")):
        r = p.relative_to(PKG / "code").as_posix()
        if p.is_file() and not r.startswith("release/") and r != "LICENSE":
            m.append((p, "06_CODE/" + r))
    m += [(PKG / "provenance/provenance_overlap_public.py", "06_CODE/Release/provenance_overlap_public.py"),
          (PKG / "provenance/PROV-1+A2_PUBLIC_RULE.md", "07_METHODS_AND_PROVENANCE/provenance_public/PROV-1+A2_PUBLIC_RULE.md"),
          (PKG / PROVENANCE_FLAGS, "07_METHODS_AND_PROVENANCE/provenance_public/P2_RECORD_PROVENANCE.csv"),
          (PKG / PROVENANCE_COUNTS, "07_METHODS_AND_PROVENANCE/provenance_public/P2_ZHANG_OVERLAP_COUNTS_PUBLIC.json")]
    return m


def stage(tree, derived_dir):
    """Copy (never link) every staged file into the work tree."""
    for src, dest in stage_map(derived_dir):
        q = Path(tree) / dest
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_bytes(Path(src).read_bytes())


# ====================================================================== the generators and where their outputs are delivered
GENERATORS = [(f"figure{n}", f"06_CODE/Figures/figure{n}.py") for n in range(1, 5)] + \
             [(f"ed{n:02d}", f"06_CODE/Figures/ed{n:02d}.py") for n in range(1, 11)]
SUPPORT_OUTPUTS = {
    "figure1": ["source_data/Figure1/figure1_payload.json"],
    "figure2": ["source_data/Figure2/figure2_selection.json"],
    "figure4": ["source_data/Figure4/figure4_selection.json"] + [f"source_data/Figure4/SetA_{name}_bootstrap_draws.csv"
                               for name in ("whole_array", "owner_block")],
}
REFIT = "06_CODE/Checks/verify_ed10_refit.py"
NESTED = "06_CODE/Checks/nested_owner.py"
NULLCHK = "06_CODE/Checks/role_null_checks.py"
SITABLES = "06_CODE/Checks/si_tables_checks.py"
FROZENREADS = "06_CODE/Checks/frozen_reads_recomputed.py"


def delivered(file_in_provenance):
    """The package file of an output named in a provenance record (None for the plot read-back, which is not released:
    its hash is part of the provenance record, which is compared)."""
    f = file_in_provenance
    for a, b in (("02_FIGURES/Main/", "figures/Main/"), ("02_FIGURES/Supplementary_Figures/", "figures/Supplementary_Figures/"), ("04_SOURCE_DATA/", "source_data/")):
        if f.startswith(a):
            return b + f[len(a):]
    return None


def same_payload(a, b):
    """Byte identity; failing that, identity of the content inside the container (zip members, gzip stream, PNG pixels),
    with the reason. Returns (status, note)."""
    A, B = Path(a).read_bytes(), Path(b).read_bytes()
    if A == B:
        return "identical", ""
    if str(a).endswith((".xlsx", ".zip")):
        za, zb = zipfile.ZipFile(io.BytesIO(A)), zipfile.ZipFile(io.BytesIO(B))
        if sorted(za.namelist()) == sorted(zb.namelist()) and all(za.read(n) == zb.read(n) for n in za.namelist()):
            return "payload identical", "every workbook part is identical; the zip container differs"
    if str(a).endswith(".gz") and gzip.decompress(A) == gzip.decompress(B):
        return "payload identical", "the uncompressed content is identical; the gzip container differs"
    if str(a).endswith(".png"):
        from PIL import Image
        ia, ib = Image.open(io.BytesIO(A)), Image.open(io.BytesIO(B))
        if ia.size == ib.size and ia.mode == ib.mode and ia.tobytes() == ib.tobytes() and ia.info.get("dpi") == ib.info.get("dpi"):
            return "payload identical", "pixels and resolution are identical; the PNG encoding differs"
    return "differs", ""


def flat(x, pre=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, f"{pre}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flat(v, f"{pre}[{i}]")
    else:
        yield pre, x


def substitutions():
    """The public stand-ins for internal inputs, by work-tree path: the hash a locked provenance record names, and the
    package file that stands in for it with how it was made."""
    sub = {}
    rec = json.loads((PKG / DERIVED_RECORD).read_text(encoding="utf-8"))
    for name, r in rec.items():
        for h in r.get("locked_provenance_sha256", []):
            sub[(DERIVED_AT[name], h)] = (name, r["transformation"])
    for r in read_text_table(RESULTS_MANIFEST).itertuples():
        for h in filter(None, r.locked_provenance_sha256.split(";")):
            sub[("05_DATA/Results/" + r.file[len("results/"):], h)] = (r.file, r.transformation)
    return sub


NOT_RELEASED_FILES = {
    "fold_registry": {"05_DATA/Analysis/06_METADATA/frozen_folds.json"},
    "registry_integrity": {"05_DATA/Analysis/06_METADATA/input_lock.json",
                           "07_METHODS_AND_PROVENANCE/analysis_runtime/PATH_RELOCATIONS.json"},
    "phase2_code": {"06_CODE/Directional_Ownership/phase2_section7.py"},
    "frozen": {"05_DATA/Results/validation/matched/fold_assignments.csv"},
}
RELEASED_STATE = {"04_SOURCE_DATA/" + h for h in HIDDEN_STATE} | {
    "05_DATA/Analysis/figure_display/Fig4a_folds.csv", "05_DATA/Analysis/figure_display/Fig4a_history_draws.csv"}


def input_record_file(leaves, key):
    """Find the file identity of a leaf, including legacy metadata beside file/hash."""
    parent = key.rsplit("/", 1)[0]
    return leaves.get(parent + "/file")


def compare_provenance(locked, rebuilt, tree):
    """Everything but the inputs must be equal. In the inputs: a hash may change only where a declared public stand-in was
    read; an entry may be missing only for the fold registry, its integrity files, the Phase 2 source and the matched-control
    fold copy, which the public rebuild does not read; an entry may be added only for a released file of the package.
    Returns (problems, explained differences)."""
    bad, notes = [], []
    L, R = json.loads(Path(locked).read_text(encoding="utf-8")), json.loads(Path(rebuilt).read_text(encoding="utf-8"))
    for k in sorted(set(L) | set(R)):
        if k != "inputs" and L.get(k) != R.get(k):
            bad.append(f"field {k} differs")
    a, b = dict(flat(L.get("inputs", {}))), dict(flat(R.get("inputs", {})))
    sub = substitutions()
    for k in sorted(set(a) | set(b)):
        if a.get(k) == b.get(k):
            continue
        if k in a and k in b and k.endswith("sha256"):
            fk = k[:-len("sha256")] + "file"
            f = b.get(fk) if fk in b else None
            if f is not None and a.get(fk) == f and (f, a[k]) in sub and sha256(Path(tree) / f) == b[k]:
                notes.append(f"{f}: read {sub[(f, a[k])][0]} in place of the internal file ({sub[(f, a[k])][1]})")
                continue
        role = re.split(r"[/\[]", k.lstrip("/"), 1)[0]
        if k in a and k not in b and input_record_file(a, k) in NOT_RELEASED_FILES.get(role, set()):
            notes.append(f"inputs{k}: not read by the public rebuild")
            continue
        if k in b and k not in a and k.startswith("/released"):
            fk = k[:-len("sha256")] + "file" if k.endswith("/sha256") else k
            f = b.get(fk)
            if not k.endswith(("/file", "/sha256")) or not isinstance(f, str) or f not in RELEASED_STATE \
                    or not (Path(tree) / f).is_file() or (k.endswith("/sha256") and sha256(Path(tree) / f) != b[k]):
                bad.append(f"inputs{k}: {str(b[k])[:60]} is not a released file of this package")
            elif k.endswith("/file"):
                notes.append(f"inputs{k}: read the released {f}")
            continue
        bad.append(f"inputs{k}: {str(a.get(k))[:60]} -> {str(b.get(k))[:60]}")
    return bad, sorted(set(notes))


# ====================================================================== statements of the text recomputed from the package
def set_b_summary(L, seed=20261007, n_boot=2000):
    """The frozen Set B estimator's summary (partner_external.summarise, as the frozen cross_library_validation.py calls it),
    restated: the pooled D, the 95% intervals of 2,000 whole-array draws and, from the same generator next, 2,000 draws of the
    owning building blocks within arrays, and the arrays with a positive D among those where D is defined."""
    D = lambda d: (d.sse_n.sum() - d.sse_e.sum()) / d.sst.sum() if d.sst.sum() > 0 else np.nan
    rng = np.random.default_rng(seed)
    L = L.assign(gain=L.sse_n - L.sse_e)
    A = L.groupby("screen")[["gain", "sst"]].sum()
    ia = rng.integers(0, len(A), (n_boot, len(A)))
    num, den = A.gain.to_numpy()[ia].sum(1), A.sst.to_numpy()[ia].sum(1)
    whole = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    num, den = np.zeros(n_boot), np.zeros(n_boot)
    for _, g in L.groupby("screen"):
        c = g.groupby("electrophile")[["gain", "sst"]].sum()
        ic = rng.integers(0, len(c), (n_boot, len(c)))
        num += c.gain.to_numpy()[ic].sum(1)
        den += c.sst.to_numpy()[ic].sum(1)
    clus = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    per = L.groupby("screen").apply(lambda g: D(g), include_groups=False)
    return dict(D=float(D(L)), ci_whole_array=[float(np.nanpercentile(whole, q)) for q in (2.5, 97.5)],
                ci_owner_cluster=[float(np.nanpercentile(clus, q)) for q in (2.5, 97.5)], positive=int((per > 0).sum()),
                arrays=int(per.notna().sum()))


def text_claim_values(U, mask, K, C, flags, fig2_prov):
    """Claim id -> the numbers it states, as printed, recomputed from the package."""
    A1, A3 = U["A1"], U["A3"]
    y1 = numeric(A1["yield"])
    m1 = mask[mask.unit == "A1"]
    cells = pd.concat([A1[["screen", "subpair", "condition_id"]], m1[["screen", "subpair", "condition_id"]]])
    designed = sum(g.subpair.nunique() * g.condition_id.nunique() for _, g in cells.groupby("screen"))
    mols = set(A1.sub_1_smiles) | set(A1.sub_2_smiles)
    hy = A1[(A1.screen == "Hiyama|1") & (A1.condition_id == "C43")].set_index("subpair")["yield"].astype(float)
    dup = A3.set_index("subpair")["yield"].astype(float)
    r_raw = float(np.corrcoef(hy.reindex(dup.index), dup)[0, 1])
    r_pattern = fig2_prov["panels"]["A"]["noise_calibration"]["duplicate_agreement"]
    p_pattern = fig2_prov["panels"]["A"]["noise_calibration"]["duplicate_pattern_pearson"]
    st = pd.concat([U[u].measurement_status for u in UNITS] + [mask.measurement_status])
    names = dict(quantified="quantified", zero_by_rule="zero by rule", detected_not_quantified="product detected but not quantified",
                 no_screen_record="no screen record")
    used = set(zip(pd.concat([U[u].rxn_class for u in UNITS]), pd.concat([U[u].condition_id for u in UNITS])))
    slots = set(K[[f"Reagent{i}_id" for i in range(1, 7)]].to_numpy().ravel()) - {""}
    sa = U["SA1"]
    c = lambda n: f"{n:,}"
    sbf = read_text_table(SET_B_FLAGS)
    scored, single = sbf.documented_prior_pair_level_scoring == "True", sbf.documented_prior_single_role_losses == "True"
    keys = set(pair_key(sbf))
    flagged = set(pair_key(sbf[scored]))
    if keys != set(pair_key(U["SB1"])) or len(sbf) != len(keys):
        flagged = set()                                  # the flags must cover the released Set B pairs, once each
    tl = pd.read_csv(PKG / SET_B_TARGETS, float_precision="round_trip")
    kept = set_b_summary(tl[~pair_key(tl).isin(flagged)])
    f3 = lambda v: f"{v:.3f}".replace("-", "−")
    ci = lambda v: f"{f3(v[0])}–{f3(v[1])}"
    return {
        "T01": [c(len(A1)), c(A1.subpair.groupby(A1.screen).nunique().sum()), c(len(mols)), f"{100 * len(A1) / designed:.1f}%", c(designed)],
        "T02": [c(len(m1)), f"not quantifiable, {c(int((m1.measurement_status == 'detected_not_quantified').sum()))}",
                f"no screen record, {c(int((m1.measurement_status == 'no_screen_record').sum()))}", c(len(A3))],
        "T03": [c(int((A1.check_passed == "False").sum() + (A3.check_passed == "False").sum()))],
        "T04": [c(dup.size), f"r = {r_raw:.2f}", f"is {r_pattern:.2f}", f"Pearson r = {p_pattern:.2f}"],
        "T05": [c(TOTAL), c(len(A1)), c(len(A3)), c(len(sa)), c(sa.subpair.groupby(sa.screen).nunique().sum()), c(len(U["SB1"])),
                c(U["SB1"].subpair.groupby(U["SB1"].screen).nunique().sum())],
        "T06": [names[s] for s in MEASURED + NO_YIELD] if set(st) == set(MEASURED + NO_YIELD)
                else ["statuses in the package differ from the four"],
        "T07": [c(len(K)) if set(zip(K.rxn_class, K.condition_id)) == used and len(K) == len(used) else "conditions table and records disagree",
                c(len(slots)) if slots == set(C["Compound ID"]) else "reagent table and reagent slots disagree"],
        "T08": [c(len(read_text_table(SAM)))],
        "T09": [c(int((flags.matches_zhang2025 == "True").sum())) if len(flags) == TOTAL
                else "provenance record count differs from the released measurement count"],
        "T12": [c(len(keys) - len(flagged)), f3(kept['D']),
                f"{f3(kept['ci_owner_cluster'][0])} to {f3(kept['ci_owner_cluster'][1])}",
                f"{f3(kept['ci_whole_array'][0])} to {f3(kept['ci_whole_array'][1])}",
                f"{kept['positive']}/{kept['arrays']}"],
        "T13": [f"{c(len(flagged))} of the {c(len(keys))} pairs", f"({c(int(U['SB1'].pipe(pair_key).isin(flagged).sum()))} measurements)",
                f"{c(int(single.sum()))} with single-role losses" if not (single & ~scored).any() else "single-role pairs outside the flagged"],
    }


def check_text_claims(values):
    """Every recomputed number must be printed in its statement."""
    bad = []
    T = read_text_table(TEXT_CLAIMS)
    if set(T.claim_id) != set(values):
        bad.append(f"text claims: listed {sorted(T.claim_id)}, recomputed {sorted(values)}")
    for r in T.itertuples():
        for v in values.get(r.claim_id, []):
            if v not in r.statement:
                bad.append(f"{r.claim_id}: recomputed {v!r} is not in the statement")
    return bad


# ====================================================================== publication use of every released measurement
def publication_use(U):
    """One row per released pair: the retained results that use its measurements, by the rule each use follows."""
    rows = []
    tt = read_text_table(TRANSFER_TARGETS)
    transfer = set(pair_key(tt))
    decision = set(pair_key(read_text_table(DECISION_TARGETS)))
    tb = set(pair_key(read_text_table(SET_B_TARGETS)))
    for u, d in U.items():
        g = d.assign(y=numeric(d["yield"])).groupby(["screen", "subpair"], sort=False).agg(n=("y", "size"), best=("y", "max")).reset_index()
        for r in g.itertuples():
            k, uses = f"{r.screen}|{r.subpair}", []
            if u == "A1":
                uses.append("Fig. 2A: the product-forming gate reads every cell of every pair of the fifteen screens")
            if u == "A3":
                uses.append("Methods repeatability and Fig. 2A noise calibration: the duplicate (C46) of a C43 cell")
            if u == "SA1" and k in transfer:
                uses.append("Fig. 4A-B: record-transfer target of Set A")
            if u == "SA1" and k in decision:
                uses.append("Results: target of the three-condition decision on Set A (e3)")
            if u == "SA1" and r.screen == "Acylation|1" and r.best >= FORMS:
                uses.append("Fig. S5: product-forming amide pair of Set A")
            if u == "SB1" and k in tb:
                uses.append("Fig. 4A-B: record-transfer target of Set B")
            rows.append(dict(unit=u, screen=r.screen, subpair=r.subpair, measurements=int(r.n), uses="; ".join(uses)))
    return pd.DataFrame(rows)


def check_publication_use(U):
    bad = []
    mine = publication_use(U)
    shipped = read_text_table(USE).astype({"measurements": int})
    if not mine.reset_index(drop=True).equals(shipped.reset_index(drop=True)):
        bad.append("PUBLICATION_USE_BY_PAIR.csv differs from the uses recomputed from the package")
    if (mine.uses == "").any():
        bad.append(f"{int((mine.uses == '').sum())} released pairs serve no retained result")
    if int(mine.measurements.sum()) != TOTAL:
        bad.append(f"publication-use table covers {int(mine.measurements.sum())} measurements")
    if A3_unmatched(U):
        bad.append("an A3 record has no C43 partner cell in A1")
    return bad


def A3_unmatched(U):
    c43 = set(pair_key(U["A1"][(U["A1"].screen == "Hiyama|1") & (U["A1"].condition_id == "C43")]))
    return sorted(set(pair_key(U["A3"])) - c43)
