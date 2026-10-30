"""Independent numerical and quantitative-artist checks for main Figures 1 and 2.

References come from measured yields, condition metadata, frozen cohorts and
individual prediction losses. No drawing module or figure-data helper is used.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from rdkit import Chem

TOL = 1e-9
CN, AMIDE = "Buchwald-Hartwig|1", "Acylation|1"
HALIDE, AMINE = "N#Cc1ccnc(Cl)c1", "Cc1cc(C)nc(N)c1"
RAW = "05_DATA/Release/00_RAW_IMMUTABLE/DATASET_analysis.csv.gz"
RULE = "07_METHODS_AND_PROVENANCE/frozen_analysis_plans/figure_selection_v3/RULE.md"
RULE_SHA = "318b4babc7d8918fcc1705d762861ebc40278111ddd82e093b10b3b32f89a721"
NS = "{http://www.w3.org/2000/svg}"
ORDER = [CN, "Buchwald-Hartwig|2", "C-O competitive reactions|1", "Heck|1", "Hiyama|1",
         "Kumada|1", "SNAr|1", "Sonogashira|1", "Suzuki|1", "Ullmann|1", AMIDE,
         "Chan-Lam|1", "Esterification|1", "Grignard|1", "Reductive-amination|1"]
NAMES = {
    HALIDE: "2-chloro-4-cyanopyridine", AMINE: "2-amino-4,6-dimethylpyridine",
    "CNc1cccc(C)c1": "N,3-dimethylaniline", "COc1ccc(N)cc1": "4-methoxyaniline",
    "COc1ccc(N)cc1OC": "3,4-dimethoxyaniline", "COc1ccc(N)cn1": "5-amino-2-methoxypyridine",
    "Cc1ccc(N)c(C)c1": "2,4-dimethylaniline", "Cc1nccn1-c1ccc(N)cc1": "4-(2-methylimidazol-1-yl)aniline",
    "CC(=O)c1ccc(Cl)c(C)c1": "4-chloro-3-methylacetophenone",
    "COc1ccccc1OS(=O)(=O)C(F)(F)F": "2-methoxyphenyl triflate",
    "COc1cnc(Cl)nc1": "2-chloro-5-methoxypyrimidine",
    "C[Si](C)(C)c1ccccc1OS(=O)(=O)C(F)(F)F": "2-(trimethylsilyl)phenyl triflate",
    "Cc1ccc(OS(=O)(=O)C(F)(F)F)cc1": "4-methylphenyl triflate",
    "Cc1ccccc1OS(=O)(=O)C(F)(F)F": "2-methylphenyl triflate",
    "Cc1ccnc(OS(=O)(=O)C(F)(F)F)c1": "4-methylpyridin-2-yl triflate",
    "FC(F)(F)c1cnc(Cl)nc1": "2-chloro-5-(trifluoromethyl)pyrimidine",
    "N#Cc1cc(Br)ccc1F": "5-bromo-2-fluorobenzonitrile",
}
BLUE, ORANGE, PINK, GREEN = "#0072B2", "#D55E00", "#CC79A7", "#009E73"


class Evidence:
    def __init__(self):
        self.checks = []
        self.differences = {}

    def require(self, name, condition, detail=""):
        assert bool(condition), f"{name}: {detail}"
        self.checks.append(dict(check=name, status="PASS", detail=str(detail)))

    def close(self, name, actual, expected):
        a, b = np.asarray(actual, float), np.asarray(expected, float)
        self.require(name + " shape", a.shape == b.shape, f"{a.shape} / {b.shape}")
        finite = np.isfinite(b)
        difference = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
        self.differences[name] = difference
        self.require(name, np.allclose(a, b, atol=TOL, rtol=0, equal_nan=True),
                     f"maximum absolute difference {difference:.6g}")

    def table(self, name, actual, expected, keys, numeric=(), text=()):
        self.require(name + " unique keys", not actual.duplicated(keys).any())
        self.require(name + " row count", len(actual) == len(expected), f"{len(actual)} / {len(expected)}")
        a, b = actual.set_index(keys), expected.set_index(keys)
        self.require(name + " identities", set(a.index) == set(b.index))
        a = a.reindex(b.index)
        if numeric:
            self.close(name + " numerical values", a[list(numeric)], b[list(numeric)])
        if text:
            self.require(name + " labels", a[list(text)].fillna("").astype(str).equals(
                b[list(text)].fillna("").astype(str)))

    def result(self, **details):
        return dict(checks=self.checks, maximum_differences=self.differences, **details)


def _path(root, *names):
    for name in names:
        path = root / name
        if path.is_file():
            return path
    raise AssertionError("Missing independent input: " + " or ".join(names))


def _result(root, name):
    return _path(root, "05_DATA/Results/" + name, "results/" + name)


def _raw(root, evidence):
    path = root / RAW
    columns = ["screen", "condition_id", "subpair", "yield"]
    if path.is_file():
        raw = pd.read_csv(path, usecols=columns)
    else:
        raw = pd.read_csv(_path(root, "data/measurements_A1_analysis_arrays.csv.gz"), usecols=columns)
    raw["yield"] = pd.to_numeric(raw["yield"], errors="coerce")
    a1 = raw[raw.screen.isin(ORDER) & raw["yield"].notna()
             & ~(raw.screen.eq("Hiyama|1") & raw.condition_id.eq("C46"))].copy()
    evidence.require("A1 measurement universe", len(a1) == 177039 and a1.screen.nunique() == 15,
                     f"{len(a1)} numeric measurements; {a1.screen.nunique()} arrays")
    evidence.require("A1 pair-condition uniqueness", not a1.duplicated(["screen", "subpair", "condition_id"]).any())
    return a1


def _pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    u, v = x[mask], y[mask]
    u, v = u - u.mean(), v - v.mean()
    return float(np.dot(u, v) / np.sqrt(np.dot(u, u) * np.dot(v, v)))


def _patterns(raw, screen):
    pivot = raw[raw.screen.eq(screen)].pivot(index="subpair", columns="condition_id", values="yield")
    pivot = pivot[sorted(pivot.columns, key=lambda value: int(value[1:]))]
    usable = np.nanmax(pivot.to_numpy(float), axis=1) >= 0.10
    pivot = pivot.loc[usable]
    yields = pivot.to_numpy(float)
    signature = np.nanmean(yields, axis=0)
    residuals = np.full_like(yields, np.nan)
    for i, row in enumerate(yields):
        mask = np.isfinite(row) & np.isfinite(signature)
        x, y = signature[mask], row[mask]
        xmean, ymean = x.mean(), y.mean()
        centered = x - xmean
        slope = np.dot(centered, y - ymean) / np.dot(centered, centered)
        residuals[i, mask] = 100 * (y - ymean - slope * centered)
    pairs = np.array(pivot.index)
    return dict(y=yields, z=residuals, pairs=pairs, conditions=list(pivot.columns),
                a=np.array([pair.split(" >> ")[0] for pair in pairs]),
                b=np.array([pair.split(" >> ")[1] for pair in pairs]))


def _metadata(root, conditions, evidence):
    metadata = pd.read_csv(_path(root, "10_PUBLIC_RELEASE_CANDIDATE/data/conditions.csv", "data/conditions.csv"))
    metadata = metadata[metadata.reaction_type.eq("buchwald") & metadata.condition_id.isin(conditions)].copy()
    evidence.require("95 unique condition compositions", len(metadata) == metadata.condition_id.nunique() == 95)
    metadata = metadata.rename(columns={"Reagent1_Standardized Name": "precatalyst",
                                       "Reagent2_Standardized Name": "added_ligand",
                                       "Reagent3_Standardized Name": "base", "Standardized Name": "solvent"})
    metadata["added_ligand"] = metadata.added_ligand.fillna("")
    def metal(name):
        if "PEPPSI" in name.upper():
            return "Pd"
        found = re.findall(r"Pd|Cu|Ni", name)
        assert len(set(found)) == 1, name
        return found[0]
    metadata["metal"] = metadata.precatalyst.map(metal)
    components = ["metal", "precatalyst", "added_ligand", "base", "solvent", "condition_id"]
    metadata = metadata.sort_values(components, key=lambda c: c.str.casefold(), kind="stable").reset_index(drop=True)
    metadata["column"] = np.arange(1, 96)
    def ligand(row):
        name = row.added_ligand or row.precatalyst
        lower = name.casefold()
        if any(s in lower for s in ("peppsi", "ipr", "ipent")):
            return "NHC"
        if any(s in lower for s in ("phos", "phosph", "phosp", "catacxium", "p(tbu)", "p(o-tolyl)",
                                    "dppf", "dtbpf", "binap", "cypf", "biphep")):
            return "Phosphine"
        if row.added_ligand in {"BTMPO", "DMPAO", "2,6-difluoroanilino(oxo)acetic acid", "trans-1,2-Cy-(NHMe)2",
                               "dipivaloylmethane", "Me4Phen", "L-proline"}:
            return "N/O"
        return "Other" if row.added_ligand else "None"
    metadata["ligand_family"] = [ligand(row) for row in metadata.itertuples()]
    classes = {"NaOtBu": "Alkoxide", "KOtBu": "Alkoxide", "Cs2CO3": "Carbonate", "K2CO3": "Carbonate",
               "K3PO4": "Phosphate", "LiHMDS": "Amide"}
    metadata["base_class"] = metadata.base.map(classes).fillna("Other")
    evidence.require("metal recipe counts", metadata.metal.value_counts().to_dict() == {"Pd": 87, "Cu": 7, "Ni": 1})
    return metadata


def _preference_reference(root, evidence):
    rule = _path(root, RULE, "design/figure_selection_v3/RULE.md")
    evidence.require("frozen display rule identity", hashlib.sha256(rule.read_bytes()).hexdigest() == RULE_SHA)
    raw = _raw(root, evidence)
    patterns = {screen: _patterns(raw, screen) for screen in (CN, AMIDE)}
    rows = []
    for screen, data in patterns.items():
        for role, molecules in ((1, data["a"]), (2, data["b"])):
            for molecule in sorted(set(molecules)):
                ix = np.flatnonzero(molecules == molecule)
                if len(ix) >= 4:
                    values = [_pearson(data["z"][i], data["z"][j]) for i, j in itertools.combinations(ix, 2)]
                    rows.append(dict(screen=screen, role=role, molecule=molecule,
                                     product_forming_partners=len(ix), agreement=float(np.mean(values))))
    agreements = pd.DataFrame(rows)
    counts = agreements.groupby(["screen", "role"]).size().to_dict()
    evidence.require("four-partner molecule eligibility", counts == {(CN, 1): 19, (CN, 2): 23, (AMIDE, 1): 42, (AMIDE, 2): 47})
    medians = agreements.groupby(["screen", "role"]).agreement.agg(median="median", molecules="size").reset_index()
    for role, focal in ((1, AMINE), (2, HALIDE)):
        group = agreements[agreements.screen.eq(CN) & agreements.role.eq(role)]
        distance = abs(group.agreement - group.agreement.median())
        evidence.require(f"role {role} median-nearest focal", distance.eq(distance.min()).sum() == 1
                         and group.loc[distance.idxmin(), "molecule"] == focal)
    metadata = _metadata(root, patterns[CN]["conditions"], evidence)
    order = metadata.condition_id.tolist()
    data = patterns[CN]
    indices = [data["conditions"].index(c) for c in order]
    partners, correlations = {}, []
    for role, prefix, fixed, changing, focal in (
        ("amine", "a", data["b"], data["a"], HALIDE),
        ("halide", "h", data["a"], data["b"], AMINE),
    ):
        ix = sorted(np.flatnonzero(fixed == focal), key=lambda i: (NAMES[changing[i]].casefold(), changing[i]))
        records = []
        for n, i in enumerate(ix, 1):
            records.append(dict(id=f"{prefix}{n}", name=NAMES[changing[i]], smiles=changing[i],
                                pair=data["pairs"][i], focal_smiles=focal,
                                pattern_points=data["z"][i, indices], yield_fraction=data["y"][i, indices],
                                positive_conditions=int((data["y"][i] > 0).sum()),
                                shared=changing[i] == (AMINE if role == "amine" else HALIDE)))
        partners[role] = records
        for i, row in enumerate(records):
            for j in range(i):
                other = records[j]
                correlations.append(dict(role=role, id_i=row["id"], id_j=other["id"], name_i=row["name"],
                                         name_j=other["name"], smiles_i=row["smiles"], smiles_j=other["smiles"],
                                         r=_pearson(row["pattern_points"], other["pattern_points"])))
    evidence.require("17 name-ordered partner identities", len(partners["amine"]) == 7 and len(partners["halide"]) == 10)
    shared = next(row for row in partners["amine"] if row["shared"])
    candidates = []
    for row in partners["amine"]:
        if not row["shared"]:
            r = _pearson(shared["pattern_points"], row["pattern_points"])
            candidates.append(dict(id=row["id"], name=row["name"], smiles=row["smiles"], r_to_shared=r,
                                   distance_to_0_27=abs(r - .27), positive_conditions=row["positive_conditions"]))
    candidates.sort(key=lambda row: (row["distance_to_0_27"], -row["positive_conditions"], row["smiles"]))
    evidence.require("Rule 1 unique candidate and selection", len(candidates) == 6 and candidates[0]["id"] == "a6")
    correlations = pd.DataFrame(correlations)
    means = correlations.groupby("role").r.mean().to_dict()
    best = float(100 * np.nanmax(shared["yield_fraction"]))
    return dict(raw=raw, patterns=patterns, agreements=agreements, medians=medians, metadata=metadata,
                order=order, partners=partners, correlations=correlations, means=means, candidates=candidates,
                selected=candidates[0], best=best)


def _check_payload(root, ref, evidence):
    path = _path(root, "04_SOURCE_DATA/Figure2/figure2_payload.json", "source_data/Figure2/figure2_payload.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    evidence.require("Figure 2 payload rule and condition order", payload["rule_sha256"] == RULE_SHA
                     and payload["condition_order"] == ref["order"])
    for role in ("amine", "halide"):
        actual, expected = payload[role + "_rows"], ref["partners"][role]
        evidence.require(role + " payload partner identities", len(actual) == len(expected))
        for a, b in zip(actual, expected):
            for field in ("id", "name", "smiles", "pair", "focal_smiles", "positive_conditions", "shared"):
                evidence.require(f"{role} {b['id']} payload {field}", a[field] == b[field])
            for field in ("pattern_points", "yield_fraction"):
                evidence.close(f"{role} {b['id']} payload {field}", a[field], b[field])
        evidence.close(role + " payload mean r", [payload["mean_correlations"][role]], [ref["means"][role]])
    evidence.table("payload all correlations", pd.DataFrame(payload["correlations"]), ref["correlations"],
                   ["role", "id_i", "id_j"], ["r"], ["name_i", "name_j", "smiles_i", "smiles_j"])
    selection = payload["selected_second_amine"]
    for field in ("id", "smiles", "name", "positive_conditions"):
        evidence.require("payload Rule 1 selected molecule " + field, selection[field] == ref["selected"][field])
    evidence.close("payload Rule 1 selected distance and r", [selection["r_to_shared"], selection["distance_to_0_27"]],
                   [ref["selected"]["r_to_shared"], ref["selected"]["distance_to_0_27"]])
    evidence.table("payload Rule 1 all candidates", pd.DataFrame(payload["rule1_candidates"]), pd.DataFrame(ref["candidates"]),
                   ["id"], ["r_to_shared", "distance_to_0_27", "positive_conditions"], ["name", "smiles"])
    evidence.require("payload shared chemical pair", payload["shared_pair"] == AMINE + " >> " + HALIDE)
    evidence.close("payload best observed yield", [payload["shared_pair_best_yield_pct"]], [ref["best"]])
    return payload


def _condition_table(actual, ref, evidence, name):
    columns = ["condition_id", "metal", "precatalyst", "added_ligand", "base", "solvent", "ligand_family", "base_class"]
    evidence.table(name, actual, ref["metadata"], ["column"], text=columns)


def _barcode_rows(ref):
    selected = {"a2", ref["selected"]["id"]}
    return pd.DataFrame([
        dict(amine_id=row["id"], halide_id="h2", name=row["name"], pair=row["pair"], column=i + 1,
             condition_id=condition, pattern_points=value, displayed_color_points=np.clip(value, -30, 30))
        for row in ref["partners"]["amine"] if row["id"] in selected
        for i, (condition, value) in enumerate(zip(ref["order"], row["pattern_points"]))])


def _audit2(root, tables, ref, evidence):
    _check_payload(root, ref, evidence)
    evidence.table("Figure 2A all 66 correlations", tables["A_correlations"], ref["correlations"],
                   ["role", "id_i", "id_j"], ["r"], ["name_i", "name_j", "smiles_i", "smiles_j"])
    evidence.require("Figure 2A 21 and 45 r counts", tables["A_correlations"].role.value_counts().to_dict() == {"halide": 45, "amine": 21})
    _condition_table(tables["B_condition_metadata"], ref, evidence, "Figure 2B metadata-only column order")
    evidence.table("Figure 2B two selected barcodes", tables["B_actual_barcode_cells"], _barcode_rows(ref),
                   ["amine_id", "column"], ["pattern_points", "displayed_color_points"],
                   ["halide_id", "name", "pair", "condition_id"])
    evidence.table("Figure 2 Rule 1 all six candidates", tables["Rule1_candidates"], pd.DataFrame(ref["candidates"]),
                   ["id"], ["r_to_shared", "distance_to_0_27", "positive_conditions"], ["name", "smiles"])
    evidence.require("Figure 2 Rule 1 candidate order", tables["Rule1_candidates"].id.tolist() == [r["id"] for r in ref["candidates"]])
    d = tables["C_molecule_agreement"]
    molecule_rows = d[d.row_type.eq("molecule")]
    evidence.table("Figure 2C all four eligible molecule groups", molecule_rows, ref["agreements"],
                   ["screen", "role", "molecule"], ["product_forming_partners", "agreement"])
    median_rows = d[d.row_type.eq("median of the group")].rename(columns={"agreement": "median"})
    evidence.table("Figure 2C agreement table group medians", median_rows, ref["medians"],
                   ["screen", "role"], ["median", "molecules"])
    evidence.table("Figure 2C displayed group medians", tables["C_group_medians"], ref["medians"],
                   ["screen", "role"], ["median", "molecules"])
    for label, molecule in (("B", HALIDE), ("C", AMINE)):
        selected = molecule_rows[molecule_rows.example.eq(label)]
        evidence.require("Figure 2C highlighted " + label, len(selected) == 1 and selected.molecule.iloc[0] == molecule)
    partners = pd.DataFrame([row for role in ("amine", "halide") for row in ref["partners"][role]])
    evidence.table("Figure 2A molecular structure identities", tables["A_partner_structures"], partners,
                   ["id"], text=["name", "smiles"])
    evidence.table("Figure 2A abbreviated halide identities", tables["A_native_abbreviations"],
                   partners[partners.id.str.startswith("h")], ["id"], text=["smiles"])
    for row in tables["A_native_abbreviations"].itertuples():
        expected = Chem.MolFromSmiles(row.smiles)
        expanded = Chem.MolFromSmiles(row.expanded_smiles)
        evidence.require("Figure 2A expanded abbreviation " + row.id,
                         expected is not None and expanded is not None
                         and Chem.MolToSmiles(expected) == Chem.MolToSmiles(expanded))
    bands = []
    for component in ("metal", "ligand_family"):
        values = ref["metadata"][component].tolist()
        start = 0
        for stop in range(1, 96):
            if stop == 95 or values[stop] != values[start]:
                bands.append(dict(component=component, name=values[start], first_column=start + 1, last_column=stop))
                start = stop
    evidence.table("Figure 2B composition-band ranges", tables["B_composition_bands"], pd.DataFrame(bands),
                   ["component", "first_column"], ["last_column"], ["name"])
    _ledger2(tables, ref, evidence)


def _ledger2(tables, ref, evidence):
    expected = []
    def add(panel, value, display, field, **other):
        expected.append(dict(panel=panel, value=value, display=display, source_sheet_or_field=field, **other))
    for role, n, old in (("amine", 21, "B"), ("halide", 45, "C")):
        value = ref["means"][role]
        add("A", value, f"{value:.2f}", "Fig1BC_r:mean; existing panel " + old)
        add("A", n, str(n), "Fig1BC_r:pairwise correlation count", role=role)
    med = ref["medians"].set_index(["screen", "role"])
    for screen, role in ((CN, 2), (CN, 1), (AMIDE, 1), (AMIDE, 2)):
        row = med.loc[(screen, role)]
        add("C", row["median"], f"{row['median']:.2f}", "Fig1D:agreement; row_type=median of the group", screen=screen, role=role)
        add("C", row.molecules, str(int(row.molecules)), "Fig1D:molecules; row_type=median of the group", screen=screen, role=role)
    for v in (-.6, 0, .6):
        add("A", v, str(v), "correlation color scale endpoint/tick")
    for v in (-30, 0, 30):
        add("B", v, str(v), "BARCODE_CMAP:preference color scale percentage points")
    for v in (-.2, 0, .2, .4, .6, .8):
        add("C", v, f"{v:g}", "agreement axis tick")
    for role in ("amine", "halide"):
        for row in ref["partners"][role]:
            for occurrence in ("matrix row label", "matrix column label", "structure identifier"):
                add("A", int(row["id"][1:]), row["id"], "subpair/name-based identifier", smiles=row["smiles"], occurrence=occurrence)
    for label in ("a2", "h2"):
        add("A", 2, label, "fixed molecule identifier in heading")
    for label in ("a2", "h2", ref["selected"]["id"], "h2"):
        add("B", int(label[1:]), label, "shared or Rule1 molecule identifier")
    a, b = tables["Printed_numbers"].reset_index(drop=True).copy(), pd.DataFrame(expected)
    a["ledger_row"], b["ledger_row"] = np.arange(len(a)), np.arange(len(b))
    for column in ("role", "screen", "smiles", "occurrence"):
        if column not in b:
            b[column] = np.nan
    evidence.table("Figure 2 complete printed-number ledger", a, b, ["ledger_row"], ["value"],
                   ["panel", "display", "source_sheet_or_field", "role", "screen", "smiles", "occurrence"])


def _board_reference(ref):
    cells, barcodes = [], []
    horizontal = ["a1", "a3", "a4", "a2", "a5", "a6", "a7"]
    vertical = ["h1", "h3", "h4", "h5", "h2", "h6", "h7", "h8", "h9", "h10"]
    for h in ref["partners"]["halide"]:
        for a in ref["partners"]["amine"]:
            if not (h["shared"] or a["shared"]):
                continue
            record = a if h["shared"] else h
            shared = h["shared"] and a["shared"]
            xx = 3 + (horizontal.index(a["id"]) if h["shared"] else 3) * 18.8
            yy = 41 + (4 if h["shared"] else vertical.index(h["id"])) * 14.7
            cell = dict(board_row=h["id"], board_column=a["id"], pair=a["smiles"] + " >> " + h["smiles"],
                        halide_name=h["name"], amine_name=a["name"], measured=True, highlighted=True,
                        measured_conditions=int(np.isfinite(record["yield_fraction"]).sum()),
                        arm="shared" if shared else "horizontal" if h["shared"] else "vertical",
                        x_mm=xx, y_mm=yy, width_mm=18.8, height_mm=14.7)
            cells.append(cell)
            for i, c in enumerate(ref["order"]):
                barcodes.append(dict(**cell, condition=c, condition_position=i + 1,
                                     yield_fraction=record["yield_fraction"][i], preference_points=record["pattern_points"][i]))
    return pd.DataFrame(cells), pd.DataFrame(barcodes)


def _shares(patterns):
    e, a, b = patterns["z"], patterns["a"], patterns["b"]
    variance = float(np.mean(np.nanmean(e ** 2, axis=1)))
    result = []
    for same, other in ((a, b), (b, a)):
        products = []
        for molecule in sorted(set(same)):
            for i, j in itertools.permutations(np.flatnonzero(same == molecule), 2):
                finite = np.isfinite(e[i]) & np.isfinite(e[j])
                if other[i] != other[j] and finite.any():
                    products.append(float(np.mean(e[i, finite] * e[j, finite])))
        result.append(float(np.mean(products) / variance))
    return result


def _molecule_t(patterns, molecule, role):
    e = patterns["z"]
    molecules = patterns["a" if role == 1 else "b"]
    ix = np.flatnonzero(molecules == molecule)
    products = [float(np.mean(e[i, ok] * e[j, ok])) for i, j in itertools.permutations(ix, 2)
                for ok in [np.isfinite(e[i]) & np.isfinite(e[j])] if ok.any()]
    assert products, f"No repeated product-forming partners for {molecule}"
    return float(np.mean(products) / np.mean(np.nanmean(e ** 2, axis=1)))


def _overview_reference(root, ref, evidence):
    patterns = {screen: ref["patterns"].get(screen) or _patterns(ref["raw"], screen) for screen in ORDER}
    anatomy = pd.read_csv(_result(root, "display_inputs/Anatomy_arrays.csv")).set_index("screen")
    role_source = _path(root, "06_CODE/Core/00_common/ctb/history.py", "code/ctb/history.py")
    role_tree = ast.parse(role_source.read_text(encoding="utf-8"))
    roles = next(ast.literal_eval(node.value) for node in role_tree.body if isinstance(node, ast.Assign)
                 and any(getattr(target, "id", None) == "ROLE" for target in node.targets))
    arrays = []
    for n, screen in enumerate(ORDER, 1):
        s1, s2 = _shares(patterns[screen])
        owner = 1 if s1 >= s2 else 2
        evidence.require("independent owner identity " + screen, owner == int(anatomy.loc[screen, "owner_role"]))
        evidence.close("independent role shares " + screen, [s1, s2], anatomy.loc[screen, ["share1", "share2"]])
        arrays.append(dict(screen=screen, plot_row=n, owner_role=owner, share1=s1, share2=s2,
                           role1=roles[screen][0][0].replace("–", "-"), role2=roles[screen][1][0].replace("–", "-"),
                           owner_share=(s1, s2)[owner - 1], partner_share=(s1, s2)[2 - owner]))
    arrays = pd.DataFrame(arrays)
    cohorts = pd.read_csv(_path(root, "05_DATA/Analysis/directional_cohorts/DESIGN_COUNTS.csv", "design/DESIGN_COUNTS.csv"))
    primary = cohorts[(cohorts["set"] == "G1") & cohorts.array_clear_owner]
    molecules = []
    for molecule, group in primary.groupby("molecule"):
        owned, partner = [], []
        for row in group.itertuples():
            value = _molecule_t(patterns[row.array], molecule, int(row.role))
            (owned if row.context == "owner" else partner).append(value)
        molecules.append(dict(molecule=molecule, T_O=float(np.mean(owned)), T_N=float(np.mean(partner))))
    molecules = pd.DataFrame(molecules)
    evidence.require("overview 23 repeated-context molecules", len(molecules) == 23)
    triples = sorted(cohorts.loc[cohorts.triple_context, "molecule"].unique(),
                     key=lambda m: (Chem.MolFromSmiles(m).GetNumHeavyAtoms(), m))
    evidence.require("overview existing M1 identity", len(triples) == 8 and triples[0] == "Cc1ccc(N)cc1")
    m1 = dict(plot_label="M1", smiles=triples[0], **{
        field: _molecule_t(patterns[screen], triples[0], 1)
        for field, screen in (("T_amide", AMIDE), ("T_BH1", CN), ("T_BH2", "Buchwald-Hartwig|2"))})
    transfer = []
    for name, file in (("Set A", "reserve_validation/record_transfer_losses.csv"), ("Set B", "cross_library/target_losses.csv")):
        frame = pd.read_csv(_result(root, file))
        evidence.require(name + " target identities", len(frame) == (309 if name == "Set A" else 535)
                         and not frame.duplicated(["screen", "subpair"]).any())
        for screen, group in [("all", frame)] + list(frame.groupby("screen")):
            denominator = float(np.add.reduce(group.sst.to_numpy(float)))
            assert denominator > 0
            owner = 1 - float(np.add.reduce(group.sse_e.to_numpy(float))) / denominator
            partner = 1 - float(np.add.reduce(group.sse_n.to_numpy(float))) / denominator
            owner_role_name = roles[screen][int(anatomy.loc[screen, "owner_role"]) - 1][0] if screen != "all" else "mixed roles"
            owner_colour = "#30343B" if screen == "all" else _transfer_colour(owner_role_name)
            transfer.append(dict(set=name, row_type="pooled" if screen == "all" else "array", screen=screen,
                                 owner=owner, partner=partner, targets=len(group), arrays=group.screen.nunique(),
                                 owner_colour=owner_colour,
                                 owner_display_pct=100 * owner, partner_display_pct=100 * partner,
                                 owner_minus_partner_display_points=100 * (owner - partner)))
    pairs = ref["raw"][["screen", "subpair"]].drop_duplicates()
    partner_sets = {}
    for screen, pair in pairs.itertuples(index=False, name=None):
        a, b = pair.split(" >> ")
        partner_sets.setdefault((screen, a), set()).add(b)
        partner_sets.setdefault((screen, b), set()).add(a)
    recurring = sum(len(partners) > 1 for partners in partner_sets.values())
    recurrence = 100 * recurring / len(partner_sets)
    gaps = []
    for screen in ORDER:
        y = ref["raw"][ref["raw"].screen.eq(screen)].pivot(index="subpair", columns="condition_id", values="yield").to_numpy(float)
        best = int(np.nanargmax(np.nanmean(y, axis=0)))
        measured = np.isfinite(y[:, best])
        gaps.append(float(100 * np.mean(np.nanmax(y[measured], axis=1) - y[measured, best])))
    return dict(arrays=arrays, anatomy=anatomy, molecules=molecules, m1=m1, transfer=pd.DataFrame(transfer),
                recurrence_pct=recurrence, recurrence_n=recurring, recurrence_den=len(partner_sets),
                median_gap=float(np.median(gaps)), pairs=len(pairs))


def _audit1(root, tables, ref, evidence):
    _check_payload(root, ref, evidence)
    cells, barcodes = _board_reference(ref)
    evidence.require("Figure 1 reciprocal cross has 16 pairs", len(cells) == 16)
    evidence.table("Figure 1 all actual barcode slots", tables["Barcodes"], barcodes,
                   ["board_row", "board_column", "condition_position"], ["yield_fraction", "preference_points"],
                   ["pair", "condition", "halide_name", "amine_name", "measured", "highlighted"])
    evidence.table("Figure 1 reciprocal cross cells", tables["Board_cells"], cells,
                   ["board_row", "board_column"], ["measured_conditions", "x_mm", "y_mm", "width_mm", "height_mm"],
                   ["pair", "halide_name", "amine_name", "measured", "highlighted", "arm"])
    finite = int(np.isfinite(barcodes.yield_fraction).sum())
    evidence.require("Figure 1 observed and missing slots", len(barcodes) == 1520 and finite == 1480 and len(barcodes) - finite == 40)
    _condition_table(tables["Condition_order"], ref, evidence, "Figure 1 shared metadata-only condition order")
    evidence.table("Figure 1 correlation miniature source", tables["Thumbnail_correlations"], ref["correlations"],
                   ["role", "id_i", "id_j"], ["r"], ["name_i", "name_j", "smiles_i", "smiles_j"])
    overview = _overview_reference(root, ref, evidence)
    evidence.table("Figure 1 all 15 array miniature values", tables["Thumbnail_arrays"], overview["arrays"], ["screen"],
                   ["plot_row", "owner_role", "share1", "share2", "owner_share", "partner_share"], ["role1", "role2"])
    evidence.table("Figure 1 M1 context miniature source", tables["Thumbnail_amine"], pd.DataFrame([overview["m1"]]),
                   ["plot_label"], ["T_amide", "T_BH1", "T_BH2"], ["smiles"])
    evidence.table("Figure 1 all 21 prediction miniature values", tables["Thumbnail_transfer"], overview["transfer"],
                   ["set", "row_type", "screen"], ["owner", "partner", "targets", "arrays", "owner_display_pct",
                                                     "partner_display_pct", "owner_minus_partner_display_points"], ["owner_colour"])
    refs = pd.DataFrame([dict(reference=n, author=a, journal=j, year=y) for n, a, j, y in
                         [(1, "Ingoglia", "Tetrahedron", 2019), (2, "Kutchukian", "Chem. Sci.", 2016),
                          (6, "Angello", "Science", 2022), (7, "Wang", "Nature", 2024), (16, "Rinehart", "Science", 2023)]])
    evidence.table("Figure 1 literature positions", tables["Literature_tags"], refs, ["reference"], ["year"], ["author", "journal"])
    manuscript = _path(root, "01_MANUSCRIPT/Manuscript_Nature.md", "manuscript/Manuscript_Nature.md").read_text(encoding="utf-8")
    for row in refs.itertuples():
        citation = re.search(rf"(?m)^{row.reference}\. (.+)$", manuscript)
        evidence.require("Figure 1 reference identity " + str(row.reference), citation is not None
                         and row.author in citation[1] and row.journal in citation[1] and str(row.year) in citation[1])
    structure_rows = pd.read_csv(_path(root, "07_METHODS_AND_PROVENANCE/figure_inputs/figure1_structures.csv",
                                       "design/figure_inputs/figure1_structures.csv"))
    selected = structure_rows.set_index("role")
    evidence.require("Figure 1 exact two illustrated reactants", selected.loc["halide", "smiles"] == HALIDE
                     and selected.loc["N-H partner", "smiles"] == AMINE
                     and selected.loc[["halide", "N-H partner"], "source_pair"].eq(AMINE + " >> " + HALIDE).all())
    _ledger1(tables, ref, overview, refs, evidence)
    return overview


def _ledger1(tables, ref, overview, refs, evidence):
    m = overview["molecules"]
    pooled = overview["transfer"][overview["transfer"].row_type.eq("pooled")]
    evidence.require("Figure 1 rounded transfer advantage", all(round(r.owner / r.partner) == 3 for r in pooled.itertuples()))
    expected = [
        ("importance", "63%", round(overview["recurrence_pct"])),
        ("importance", "17.4 points", round(overview["median_gap"], 1)),
        ("A", "91%", ref["best"]), ("A", "95", len(ref["order"])),
        ("B", "15", len(ORDER)), ("B", "14", len({s.split("|")[0] for s in ORDER})),
        ("B", "177,039", len(ref["raw"])), ("B", "844", int(pooled.targets.sum())),
        ("badges", "0.27", ref["means"]["amine"]), ("badges", "0.07", ref["means"]["halide"]),
        ("badges", "10 of 10", int(overview["anatomy"].electrophile_role.notna().sum())),
        ("badges", "0.26", float(m.T_O.mean())), ("badges", "0.06", float(m.T_N.mean())),
        ("badges", "23", len(m)), ("badges", "3×", 3), ("badges", "844", int(pooled.targets.sum())),
    ]
    for row in refs.itertuples():
        expected += [("prior_work", str(row.year), row.year), ("prior_work", f"({row.reference})", row.reference)]
    expected += [("badges", f"Fig. {n}", n) for n in range(2, 6)]
    expected += [("badges", "10 of 10, denominator", 10), ("A", "H2N, chemical subscript 2", 2)]
    a, b = tables["Printed_numbers"].reset_index(drop=True).copy(), pd.DataFrame(expected, columns=["panel", "display", "value"])
    a["ledger_row"], b["ledger_row"] = np.arange(len(a)), np.arange(len(b))
    evidence.table("Figure 1 complete printed-number ledger", a, b, ["ledger_row"], ["value"], ["panel", "display"])


def _axes(readback):
    axes = readback["axes"]
    return list(axes.values()) if isinstance(axes, dict) else axes


def _artist(readback, kind, gid):
    found = [record for axis in _axes(readback) for record in axis.get(kind, []) if record.get("gid") == gid]
    assert len(found) == 1, f"Quantitative artist {gid}: expected one, found {len(found)}"
    return found[0]


def _audit_artists2(root, tables, readback, ref, evidence):
    images = {record.get("gid") for axis in _axes(readback) for record in axis.get("images", [])}
    expected_images = {"A_amine_correlation_matrix", "A_halide_correlation_matrix", "B_a2_actual_pattern",
                       f"B_{ref['selected']['id']}_actual_pattern"}
    evidence.require("Figure 2 complete quantitative-image inventory", images == expected_images)
    for role in ("amine", "halide"):
        n = len(ref["partners"][role])
        expected = np.full((n, n), np.nan)
        for row in ref["correlations"][ref["correlations"].role.eq(role)].itertuples():
            expected[int(row.id_i[1:]) - 1, int(row.id_j[1:]) - 1] = row.r
        artist = _artist(readback, "images", f"A_{role}_correlation_matrix")
        evidence.close(f"Figure 2A {role} correlation matrix artist", artist["array"], expected)
        evidence.close(f"Figure 2A {role} correlation colour limits", artist["clim"], [-.6, .6])
        evidence.close(f"Figure 2A {role} matrix extent", artist["extent"], [-.5, n - .5, n - .5, -.5])
        mean_labels = [text["text"] for text in readback["texts"] if text.get("gid") == f"A_{role}_mean"]
        evidence.require(f"Figure 2A {role} mean and correlation count label",
                         mean_labels == [f"Mean r {ref['means'][role]:.2f}\n{n * (n - 1) // 2} pairs"])
    for row in ref["partners"]["amine"]:
        if row["id"] not in {"a2", ref["selected"]["id"]}:
            continue
        artist = _artist(readback, "images", f"B_{row['id']}_actual_pattern")
        evidence.close("Figure 2B barcode artist " + row["id"], artist["array"], row["pattern_points"] [None, :])
        evidence.close("Figure 2B barcode colour limits " + row["id"], artist["clim"], [-30, 30])
        axis = next(axis for axis in _axes(readback) if artist in axis.get("images", []))
        missing = np.flatnonzero(~np.isfinite(row["pattern_points"]))
        hatches = [record for record in axis["patches"]]
        evidence.require("Figure 2B missing-value hatch count " + row["id"], len(hatches) == len(missing))
        for patch, index in zip(hatches, missing):
            evidence.close("Figure 2B missing-value hatch position " + row["id"],
                           list(patch["xy"]) + [patch["width"], patch["height"]], [index - .5, -.5, 1, 1])
    medians = ref["medians"].set_index(["screen", "role"])
    expected_collections, expected_lines = set(), set()
    for y, (screen, role) in enumerate(((CN, 2), (CN, 1), (AMIDE, 1), (AMIDE, 2))):
        group = ref["agreements"][ref["agreements"].screen.eq(screen) & ref["agreements"].role.eq(role)].sort_values("molecule")
        jitter = .22 * np.sin(np.arange(len(group)) * 2.4)
        gid = f"C_{screen}_{role}_molecules"
        expected_collections.add(gid)
        evidence.close("Figure 2C all molecule point artists " + gid,
                       _artist(readback, "collections", gid)["offsets"], np.c_[group.agreement, y + jitter])
        gid = f"C_{screen}_{role}_median"
        expected_lines.add(gid)
        line = _artist(readback, "lines", gid)
        median = medians.loc[(screen, role), "median"]
        evidence.close("Figure 2C median line " + gid, line["x"] + line["y"], [median, median, y - .36, y + .36])
        median_labels = [text["text"] for text in readback["texts"] if text.get("gid") == f"C_median_{screen}_{role}"]
        evidence.require("Figure 2C displayed median " + gid, median_labels == [f"{median:.2f}"])
        if screen == CN:
            molecule = HALIDE if role == 2 else AMINE
            index = list(group.molecule).index(molecule)
            gid = f"C_{screen}_{role}_example"
            expected_collections.add(gid)
            evidence.close("Figure 2C selected example point " + gid,
                           _artist(readback, "collections", gid)["offsets"], [[group.agreement.iloc[index], y + jitter[index]]])
    actual_collections = {r["gid"] for a in _axes(readback) for r in a.get("collections", []) if r.get("gid")}
    actual_lines = {r["gid"] for a in _axes(readback) for r in a.get("lines", []) if r.get("gid")}
    evidence.require("Figure 2C complete point and median inventory", actual_collections == expected_collections and actual_lines == expected_lines)


def _barcode_colour(value):
    if not np.isfinite(value):
        return "url(#missing)"
    v = np.clip(float(value), -30, 30) / 30
    white = np.full(3, 255.0)
    end = np.array([89, 89, 89] if v < 0 else [214, 168, 39], float)
    return "#" + "".join(f"{int(c):02X}" for c in np.round(white + abs(v) * (end - white)))


def _correlation_colour(value):
    v = max(-1, min(1, value / .6))
    target = np.array([178, 24, 43] if v >= 0 else [100, 100, 100])
    return "#" + "".join(f"{int(c):02X}" for c in np.round(255 + abs(v) * (target - 255)))


def _svg_number(record, field):
    return float(record.get(field, "0"))


def _svg_shapes(nodes, expected, evidence, name):
    evidence.require(name + " count", len(nodes) == len(expected), f"{len(nodes)} / {len(expected)}")
    unmatched = list(nodes)
    for item in expected:
        kind, attributes = item
        matches = []
        for node in unmatched:
            if node.tag != NS + kind:
                continue
            ok = True
            for key, value in attributes.items():
                if isinstance(value, str):
                    ok &= node.get(key, "1" if key == "opacity" else "") == value
                else:
                    ok &= abs(_svg_number(node, key) - value) <= TOL
            if ok:
                matches.append(node)
        evidence.require(name + " quantitative identity", bool(matches), str(item))
        unmatched.remove(matches[0])
    evidence.require(name + " no unaccounted quantitative shape", not unmatched)


def _role_colour(name):
    name = str(name).lower()
    if "halide" in name or "electrophile" in name:
        return BLUE
    if "acid" in name:
        return PINK
    if "amine" in name or "n-h" in name or "n–h" in name:
        return ORANGE
    return GREEN


def _transfer_colour(name):
    name = str(name).replace("–", "-")
    if "halide" in name:
        return BLUE
    if name in ("N-H partner", "N-H nucleophile", "amine"):
        return ORANGE
    return PINK if name == "carboxylic acid" else GREEN


def _master_structure(path):
    root = ET.parse(path).getroot()
    polygons, labels = [], []
    def matrix(text):
        found = re.fullmatch(r"matrix\(([^)]*)\)", text.strip())
        assert found, "Unsupported ChemDraw matrix"
        a, b, c, d, e, f = map(float, found[1].replace(",", " ").split())
        return np.array([[a, c, e], [b, d, f]])
    for node in root:
        transform = matrix(node.get("transform", ""))
        if node.tag == NS + "path":
            tokens = re.findall(r"[MLZ]|-?[\d.]+(?:e-?\d+)?", node.get("d"))
            points, i = [], 0
            while i < len(tokens):
                if tokens[i] in {"M", "L"}:
                    points.append([float(tokens[i + 1]), float(tokens[i + 2]), 1])
                    i += 3
                else:
                    assert tokens[i] == "Z"
                    i += 1
            polygons.append((transform @ np.asarray(points).T).T)
        elif node.tag == NS + "text":
            labels.append(dict(text=node.text.strip(), x=transform[0, 2], y=transform[1, 2],
                               size=float(node.get("font-size").removesuffix("px")) * transform[0, 0]))
        else:
            raise AssertionError("Unexpected ChemDraw element " + node.tag)
    return polygons, labels


def _check_structures(root, svg, readback, evidence):
    evidence.require("Figure 1 exactly two chemical drawing groups",
                     {r["id"] for r in readback["structures"]} == {"focal-halide", "focal-amine"})
    for role, x in (("halide", 3), ("amine", 31)):
        master = _path(root, f"02_FIGURES/ChemDraw/Fig1_a_{role}.svg", f"structures/Fig1_a_{role}.svg")
        polygons, labels = _master_structure(master)
        scale = 8 * 25.4 / 72 / round(max(label["size"] for label in labels), 6)
        group = svg.find(f".//{NS}g[@id='focal-{role}']")
        assert group is not None, "Missing focal structure " + role
        drawn_polygons = group.findall(NS + "polygon")
        evidence.require("Figure 1 native bond count " + role, len(drawn_polygons) == len(polygons))
        for i, (drawn, expected) in enumerate(zip(drawn_polygons, polygons)):
            actual = np.array([[float(v) for v in p.split(",")] for p in drawn.get("points").split()])
            evidence.close(f"Figure 1 native bonds {role} {i}", actual, np.round(expected * scale + (x, 72), 6))
        texts = group.findall(NS + "text")
        evidence.require("Figure 1 native atom-label count " + role, len(texts) == len(labels))
        for drawn, expected in zip(texts, labels):
            evidence.require("Figure 1 native atom-label identity " + role, drawn.text == expected["text"])
            evidence.close("Figure 1 native atom-label position " + role,
                           [_svg_number(drawn, "x"), _svg_number(drawn, "y")],
                           [round(x + expected["x"] * scale, 5), round(72 + expected["y"] * scale, 5)])


def _audit_artists1(root, tables, readback, ref, overview, evidence):
    cells, barcodes = _board_reference(ref)
    svg = ET.parse(_path(root, "02_FIGURES/Main/Figure1.svg", "figures/Figure1.svg")).getroot()
    actual = {record["id"]: record for record in readback["cells"]}
    wanted = {f"cell-{r.board_row}-{r.board_column}" for r in cells.itertuples()}
    evidence.require("Figure 1 complete 16-cell artist inventory", set(actual) == wanted)
    for cell in cells.itertuples():
        gid = f"cell-{cell.board_row}-{cell.board_column}"
        values = barcodes[barcodes.board_row.eq(cell.board_row) & barcodes.board_column.eq(cell.board_column)].sort_values("condition_position")
        fills = [_barcode_colour(v) for v in values.preference_points]
        evidence.require("Figure 1 barcode artist " + gid, actual[gid]["fills"] == fills and actual[gid]["opacity"] == "1")
        group = svg.find(f".//{NS}g[@id='{gid}']")
        assert group is not None, "Figure 1 missing SVG barcode " + gid
        rectangles = group.findall(NS + "rect")
        evidence.require("Figure 1 SVG 95 rectangles " + gid, len(rectangles) == 95)
        for i, (node, fill) in enumerate(zip(rectangles, fills)):
            evidence.require("Figure 1 SVG condition colour " + gid, node.get("fill") == fill)
            expected = [round(cell.x_mm + .9 + i * 17 / 95, 5), round(cell.y_mm + 3.8, 5), round(17 / 95 + .002, 5), 8.8]
            evidence.close("Figure 1 SVG condition placement " + gid,
                           [_svg_number(node, k) for k in ("x", "y", "width", "height")], expected)
    _check_structures(root, svg, readback, evidence)
    badges = svg.find(f".//{NS}g[@id='badges']")
    assert badges is not None
    rectangles, circles, contextual_lines = [], [], []
    def expected_rect(x, y, width, height, fill, opacity="1"):
        return "rect", dict(x=round(x, 5), y=round(y, 5), width=round(width, 5), height=round(height, 5), fill=fill, opacity=opacity)
    for role, xx in (("amine", 2), ("halide", 7)):
        n = len(ref["partners"][role])
        for r in ref["correlations"][ref["correlations"].role.eq(role)].itertuples():
            i, j = int(r.id_i[1:]) - 1, int(r.id_j[1:]) - 1
            rectangles.append(expected_rect(xx + j * 4.2 / n, 202 + i * 8.4 / n, 4.2 / n, 8.4 / n, _correlation_colour(r.r)))
    for n, row in enumerate(overview["arrays"].itertuples()):
        for field, offset, owned in (("owner_share", -.16, True), ("partner_share", .16, False)):
            value = 100 * getattr(row, field)
            role = int(row.owner_role) if owned else 3 - int(row.owner_role)
            role_name = getattr(row, f"role{role}")
            rectangles.append(expected_rect(49 + min(0, value) * .18, 199 + n * .88 + offset,
                                            abs(value) * .18, .24, _role_colour(role_name), "1" if owned else ".28"))
    m1 = overview["m1"]
    xs = [92, 96, 100]
    ys = [211 - (m1[k] + .5) / 1.2 * 12 for k in ("T_amide", "T_BH1", "T_BH2")]
    for i in range(2):
        contextual_lines.append(("line", dict(x1=xs[i], y1=ys[i], x2=xs[i + 1], y2=ys[i + 1], stroke=ORANGE)))
    for x, y in zip(xs, ys):
        circles.append(("circle", dict(cx=x, cy=y, r=.55, fill=ORANGE)))
    for row in overview["transfer"].itertuples():
        xx = 137 + (100 * row.partner + 20) / 50 * 9.5
        yy = 212 - (100 * row.owner + 20) / 50 * 13
        radius = .6 if row.row_type == "pooled" else .35
        fill = "#20252A" if row.row_type == "pooled" else row.owner_colour
        if row.set == "Set A":
            circles.append(("circle", dict(cx=xx, cy=yy, r=radius, fill=fill)))
        else:
            rectangles.append(expected_rect(xx - radius, yy - radius, 2 * radius, 2 * radius, fill))
    _svg_shapes(badges.findall(NS + "rect"), rectangles, evidence, "Figure 1 quantitative badge rectangles")
    _svg_shapes(badges.findall(NS + "circle"), circles, evidence, "Figure 1 quantitative badge circles")
    actual_context = [node for node in badges.findall(NS + "line") if node.get("stroke") == ORANGE]
    _svg_shapes(actual_context, contextual_lines, evidence, "Figure 1 M1 miniature slopes")
    visible = " ".join(record["text"] for record in readback["texts"])
    for text in ("63%", "17.4", "best 91%", "95 conditions", "177,039 measurements", "15 arrays", "14 transformations",
                 "r 0.27 vs 0.07", "10 of 10", "0.26 vs 0.06", "23 molecules", "3× more", "844 new pairs"):
        evidence.require("Figure 1 scientific badge text " + text, text in visible)


def audit_tables(number, root: Path, tables: dict[str, pd.DataFrame]) -> dict:
    assert number in (1, 2), "This module checks Figure 1 or Figure 2"
    evidence = Evidence()
    ref = _preference_reference(Path(root), evidence)
    if number == 1:
        overview = _audit1(Path(root), tables, ref, evidence)
        return evidence.result(figure=1, barcode_cells=16, barcode_slots=1520, finite_slots=1480, missing_slots=40,
                               recurrence_n=overview["recurrence_n"], recurrence_den=overview["recurrence_den"],
                               recurrence_pct=overview["recurrence_pct"], median_gap_pp=overview["median_gap"],
                               independently_recomputed_previews=4)
    _audit2(Path(root), tables, ref, evidence)
    return evidence.result(figure=2, correlations={"amine": 21, "halide": 45}, condition_columns=95,
                           selected_second_amine=ref["selected"]["id"], mean_correlations=ref["means"],
                           eligible_molecule_groups=[19, 23, 42, 47])


def audit_artists(number, root: Path, tables: dict[str, pd.DataFrame], readback: dict) -> dict:
    assert number in (1, 2), "This module checks Figure 1 or Figure 2"
    evidence = Evidence()
    ref = _preference_reference(Path(root), evidence)
    if number == 1:
        overview = _overview_reference(Path(root), ref, evidence)
        _audit_artists1(Path(root), tables, readback, ref, overview, evidence)
    else:
        _audit_artists2(Path(root), tables, readback, ref, evidence)
    return evidence.result(figure=number, quantitative_artists="independently recomputed")


def fault_tests(number, root: Path, tables: dict[str, pd.DataFrame], readback: dict) -> list[dict]:
    """Positive controls require their unmodified baseline to pass first."""
    assert number in (1, 2)
    audit_tables(number, root, tables)
    audit_artists(number, root, tables, readback)
    if number == 1:
        corrupted = copy.deepcopy(readback)
        first, second = corrupted["cells"][:2]
        first["fills"], second["fills"] = second["fills"], first["fills"]
        label = "Figure 1 swapped actual barcode artists"
        expected = "Figure 1 barcode artist " + first["id"]
        run = lambda: audit_artists(1, root, tables, corrupted)
        target = "04_SOURCE_DATA/Figure1/Figure1_readback.json:cells:" + first["id"]
    else:
        corrupted = {key: frame.copy(deep=True) for key, frame in tables.items()}
        frame = corrupted["B_actual_barcode_cells"]
        selected = frame.amine_id.ne("a2")
        frame.loc[selected, "amine_id"] = "a1"
        label = "Figure 2 wrong Rule 1 selected molecule"
        expected = "Figure 2B two selected barcodes identities"
        run = lambda: audit_tables(2, root, corrupted)
        target = "04_SOURCE_DATA/Figure2/Figure2_SourceData.xlsx:B_actual_barcode_cells:amine_id"
    try:
        run()
    except AssertionError as error:
        message = str(error)
        assert expected in message, f"Positive control failed for an unrelated reason: {message}"
        return [dict(fault=label, rejected=True, new_issue=message, status="PASS", baseline="PASS", target=target)]
    raise AssertionError(label + ": corrupted sample was accepted")
