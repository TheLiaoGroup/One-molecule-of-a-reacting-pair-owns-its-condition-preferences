"""Figure 2 descriptive data and frozen display Rule 1."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
import preference_data as established

RULE = ROOT / "07_METHODS_AND_PROVENANCE/frozen_analysis_plans/figure_selection_v3/RULE.md"
RULE_SHA256 = "318b4babc7d8918fcc1705d762861ebc40278111ddd82e093b10b3b32f89a721"
SOURCE = ROOT / "04_SOURCE_DATA/Figure2/Figure2_SourceData.xlsx"
STRUCTURES = ROOT / "02_FIGURES/ChemDraw/v3/f2"
MEASUREMENT_SOURCE = "05_DATA/Release/00_RAW_IMMUTABLE/DATASET_analysis.csv.gz"
CONDITION_SOURCE = "10_PUBLIC_RELEASE_CANDIDATE/data/conditions.csv"
CALCULATION_SOURCE = "06_CODE/Figures/preference_data.py:measurement_tables/load_data"


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _structure_paths(rows):
    manifest = ROOT / "02_FIGURES/ChemDraw/structures_manifest.csv"
    known = {}
    with manifest.open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            known.setdefault(record["smiles"], ROOT / "02_FIGURES/ChemDraw" / (record["name"] + ".svg"))
    for record in rows:
        path = known.get(record["smiles"], STRUCTURES / (record["id"] + ".svg"))
        record["structure_svg"] = relative(path)
        record["existing_structure"] = record["smiles"] in known


def extract():
    assert hashlib.sha256(RULE.read_bytes()).hexdigest() == RULE_SHA256
    sheets, metadata, blocks, best = established.load_data()
    order = metadata.condition_id.tolist()
    assert len(order) == len(set(order)) == 95
    rows_by_role = {}
    correlations = []
    for old_panel, role, prefix, focal in (
        ("B", "amine", "a", established.HALIDE),
        ("C", "halide", "h", established.AMINE),
    ):
        cells = blocks[old_panel]["cells"]
        row_specs = cells[["partner", "partner_smiles", "pair"]].drop_duplicates()
        row_specs = row_specs.sort_values(["partner", "partner_smiles"],
                                          key=lambda col: col.str.casefold() if col.name == "partner" else col,
                                          kind="stable")
        records = []
        for i, row in enumerate(row_specs.itertuples(index=False), 1):
            group = cells[cells.pair.eq(row.pair)].set_index("condition_id").loc[order]
            records.append(dict(id=f"{prefix}{i}", name=row.partner, smiles=row.partner_smiles,
                                pair=row.pair, focal_smiles=focal,
                                pattern_points=group.pattern_points.tolist(),
                                yield_fraction=group.yield_fraction.tolist(),
                                positive_conditions=int((group.yield_fraction > 0).sum()),
                                shared=row.partner_smiles == (established.AMINE if role == "amine" else established.HALIDE)))
        _structure_paths(records)
        for i in range(len(records)):
            for j in range(i):
                r = established.correlation(np.asarray(records[i]["pattern_points"]),
                                            np.asarray(records[j]["pattern_points"]))
                correlations.append(dict(role=role, id_i=records[i]["id"], id_j=records[j]["id"],
                                         name_i=records[i]["name"], name_j=records[j]["name"],
                                         smiles_i=records[i]["smiles"], smiles_j=records[j]["smiles"], r=r,
                                         source=CALCULATION_SOURCE + ":Fig1BC_r"))
        rows_by_role[role] = records
        assert abs(np.mean([r["r"] for r in correlations if r["role"] == role]) - blocks[old_panel]["mean"]) < 2e-15
    shared = next(row for row in rows_by_role["amine"] if row["shared"])
    candidates = []
    for record in rows_by_role["amine"]:
        if record["shared"]:
            continue
        r = established.correlation(np.asarray(shared["pattern_points"]), np.asarray(record["pattern_points"]))
        candidates.append(dict(id=record["id"], name=record["name"], smiles=record["smiles"],
                               r_to_shared=r, distance_to_0_27=abs(r - .27),
                               positive_conditions=record["positive_conditions"]))
    candidates.sort(key=lambda row: (row["distance_to_0_27"], -row["positive_conditions"], row["smiles"]))
    selection = dict(**candidates[0], rule=1, rule_sha256=RULE_SHA256)
    agreement = sheets["Fig1D"].copy()
    agreement["source"] = CALCULATION_SOURCE + ":Fig1D"
    means = {role: float(np.mean([r["r"] for r in correlations if r["role"] == role]))
             for role in rows_by_role}
    payload = dict(schema="figure2-v3-shared-focal-v1", rule_sha256=RULE_SHA256,
                   condition_order=order, condition_metadata=metadata.to_dict("records"),
                   amine_rows=rows_by_role["amine"], halide_rows=rows_by_role["halide"],
                   correlations=correlations, mean_correlations=means,
                   selected_second_amine=selection, rule1_candidates=candidates,
                   shared_pair=shared["pair"], shared_pair_best_yield_pct=best,
                   shared_structure_svgs={"amine": relative(ROOT / "02_FIGURES/ChemDraw/Fig1_a_amine.svg"),
                                          "halide": relative(ROOT / "02_FIGURES/ChemDraw/Fig1_a_halide.svg")},
                   sources=dict(measurements=MEASUREMENT_SOURCE, conditions=CONDITION_SOURCE,
                                calculation=CALCULATION_SOURCE,
                                correlations=CALCULATION_SOURCE + ":Fig1BC_r",
                                agreement=CALCULATION_SOURCE + ":Fig1D"))
    assert len(candidates) == 6 and len(rows_by_role["amine"]) == 7 and len(rows_by_role["halide"]) == 10
    return clean(payload), agreement


if __name__ == "__main__":
    data, _ = extract()
    print(json.dumps({"rule1_selection": data["selected_second_amine"],
                      "mean_correlations": data["mean_correlations"],
                      "missing_structures": [{k: row[k] for k in ("id", "name", "smiles", "structure_svg")}
                                             for role in ("amine_rows", "halide_rows") for row in data[role]
                                             if not row["existing_structure"]]}, indent=2))
