"""Retained scientific extraction, independent of figure rendering."""
import re
import itertools
import numpy as np
import pandas as pd
import figure_common as common

ROOT = common.ROOT


RAW = ROOT / "05_DATA/Release/00_RAW_IMMUTABLE/DATASET_analysis.csv.gz"


CONDITIONS = ROOT / "10_PUBLIC_RELEASE_CANDIDATE/data/conditions.csv"


HALIDE = "N#Cc1ccnc(Cl)c1"


AMINE = "Cc1cc(C)nc(N)c1"


CN, AMIDE = "Buchwald-Hartwig|1", "Acylation|1"


NAME = {
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


ROLE_NAMES = {CN: ("N–H partner", "aryl (pseudo)halide"), AMIDE: ("N–H partner", "carboxylic acid")}


def ligand_family(precatalyst, added):
    name = added if added else precatalyst
    lower = name.casefold()
    if any(token in lower for token in ("peppsi", "ipr", "ipent")):
        return "NHC"
    if any(token in lower for token in ("phos", "phosph", "phosp", "catacxium", "p(tbu)", "p(o-tolyl)",
                                         "dppf", "dtbpf", "binap", "cypf", "biphep")):
        return "Phosphine"
    if added in {"BTMPO", "DMPAO", "2,6-difluoroanilino(oxo)acetic acid", "trans-1,2-Cy-(NHMe)2",
                 "dipivaloylmethane", "Me4Phen", "L-proline"}:
        return "N/O"
    return "Other" if added else "None"


def base_class(name):
    if name in {"NaOtBu", "KOtBu"}:
        return "Alkoxide"
    if name in {"Cs2CO3", "K2CO3"}:
        return "Carbonate"
    if name == "K3PO4":
        return "Phosphate"
    if name == "LiHMDS":
        return "Amide"
    return "Other"


def correlation(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def measurement_tables():
    """Reproduce the established descriptive extraction from released measurements."""
    raw = pd.read_csv(RAW, usecols=["screen", "condition_id", "subpair", "sub_1_smiles", "sub_2_smiles", "yield"])
    raw = raw[raw.screen.isin((CN, AMIDE))].copy()
    raw["yield"] = pd.to_numeric(raw["yield"], errors="coerce")
    raw = raw[raw["yield"].notna()]
    assert (raw.subpair == raw.sub_1_smiles + " >> " + raw.sub_2_smiles).all()
    arrays, agreements = {}, []
    for screen, group in raw.groupby("screen"):
        pivot = group.pivot(index="subpair", columns="condition_id", values="yield")
        pivot = pivot[sorted(pivot.columns, key=lambda key: int(key[1:]))]
        matrix = pivot.to_numpy(float)
        active = np.nanmax(matrix, axis=1) >= 0.10
        matrix = matrix[active]
        signature = np.nanmean(matrix, axis=0)
        residual = np.full_like(matrix, np.nan)
        # The same normal-equation OLS arithmetic as anatomy.pattern preserves precision.
        for i in range(len(matrix)):
            observed = np.isfinite(matrix[i]) & np.isfinite(signature)
            design = np.c_[np.ones(observed.sum()), signature[observed]]
            inverse = np.linalg.pinv(design.T @ design)
            coefficients = inverse @ design.T @ matrix[i, observed]
            residual[i, observed] = matrix[i, observed] - design @ coefficients
        pairs = np.asarray(pivot.index[active])
        parts = [np.array([pair.split(" >> ")[side] for pair in pairs]) for side in (0, 1)]
        arrays[screen] = dict(yield_fraction=matrix, pattern_points=100 * residual,
                              pairs=pairs, parts=parts, conditions=list(pivot.columns))
        for role, molecules in enumerate(parts, 1):
            for molecule in sorted(set(molecules)):
                indices = np.where(molecules == molecule)[0]
                if len(indices) < 4:
                    continue
                values = [correlation(100 * residual[i], 100 * residual[j])
                          for i, j in itertools.combinations(indices, 2)]
                agreements.append(dict(screen=screen, array=common.array_label(screen, mathtext=False), role=role,
                                       role_name=ROLE_NAMES[screen][role - 1], molecule=molecule,
                                       product_forming_partners=len(indices), agreement=float(np.mean(values)),
                                       example="B" if screen == CN and molecule == HALIDE else
                                               "C" if screen == CN and molecule == AMINE else "",
                                       row_type="molecule"))
    agreement = pd.DataFrame(agreements)
    for role, expected in ((1, AMINE), (2, HALIDE)):
        group = agreement[agreement.screen.eq(CN) & agreement.role.eq(role)]
        distance = (group.agreement - group.agreement.median()).abs()
        assert distance.eq(distance.min()).sum() == 1
        assert group.loc[distance.idxmin(), "molecule"] == expected
    medians = [dict(screen=screen, array=common.array_label(screen, mathtext=False), role=role,
                    role_name=ROLE_NAMES[screen][role - 1], agreement=group.agreement.median(),
                    molecules=len(group), row_type="median of the group")
               for (screen, role), group in agreement.groupby(["screen", "role"], sort=False)]
    sheets = {"Fig1D": pd.concat([agreement, pd.DataFrame(medians)], ignore_index=True)}
    r_rows = []
    array = arrays[CN]
    for panel, focal, side, other in (("B", HALIDE, 1, AMINE), ("C", AMINE, 0, HALIDE)):
        partner = array["parts"][1 - side]
        indices = list(np.where(array["parts"][side] == focal)[0])
        indices.sort(key=lambda i: (partner[i] != other, NAME[partner[i]].casefold()))
        assert partner[indices[0]] == other
        cells = [dict(row=k + 1, partner=NAME[partner[i]], partner_smiles=partner[i], pair=array["pairs"][i],
                      condition_id=condition, column=j + 1, yield_fraction=array["yield_fraction"][i, j],
                      pattern_points=array["pattern_points"][i, j],
                      colour_value_points=float(np.clip(array["pattern_points"][i, j], -30, 30)))
                 for k, i in enumerate(indices) for j, condition in enumerate(array["conditions"])]
        sheets[f"Fig1{panel}"] = pd.DataFrame(cells)
        correlations = [dict(panel=panel, row_i=i + 1, row_j=j + 1, partner_i=NAME[partner[indices[i]]],
                             partner_j=NAME[partner[indices[j]]],
                             r=correlation(array["pattern_points"][indices[i]], array["pattern_points"][indices[j]]))
                        for i, j in itertools.combinations(range(len(indices)), 2)]
        r_rows.extend(correlations + [dict(panel=panel, partner_i=f"mean of the {len(correlations)} r",
                                          r=float(np.mean([row["r"] for row in correlations])))])
    sheets["Fig1BC_r"] = pd.DataFrame(r_rows)
    shared_pair = AMINE + " >> " + HALIDE
    best = 100 * raw.loc[raw.screen.eq(CN) & raw.subpair.eq(shared_pair), "yield"].max()
    sheets["Fig1A"] = pd.DataFrame([
        dict(molecule=NAME[HALIDE], smiles=HALIDE, role="aryl halide", kept_in="B", chemdraw_file="Fig1_a_halide.cdxml"),
        dict(molecule=NAME[AMINE], smiles=AMINE, role="N–H partner", kept_in="C", chemdraw_file="Fig1_a_amine.cdxml"),
        dict(molecule="shared pair", smiles=shared_pair, role="pair", best_yield_pct=best)])
    return sheets


def load_data():
    sheets = measurement_tables()
    cond_ids = set(sheets["Fig1B"].condition_id)
    assert cond_ids == set(sheets["Fig1C"].condition_id) and len(cond_ids) == 95
    metadata = pd.read_csv(CONDITIONS)
    metadata = metadata[metadata.reaction_type.eq("buchwald") & metadata.condition_id.isin(cond_ids)].copy()
    assert len(metadata) == 95 and not metadata.condition_id.duplicated().any()
    metadata = metadata.rename(columns={"Reagent1_Standardized Name": "precatalyst",
                                        "Reagent2_Standardized Name": "added_ligand",
                                        "Reagent3_Standardized Name": "base", "Standardized Name": "solvent"})
    metadata["added_ligand"] = metadata.added_ligand.fillna("")
    metadata["metal"] = ["Pd" if "PEPPSI" in name else re.search(r"Pd|Cu|Ni", name).group()
                         for name in metadata.precatalyst]
    sort_columns = ["metal", "precatalyst", "added_ligand", "base", "solvent", "condition_id"]
    metadata = metadata.sort_values(sort_columns, key=lambda s: s.str.casefold(), kind="stable").reset_index(drop=True)
    metadata["column"] = np.arange(1, len(metadata) + 1)
    metadata["ligand_family"] = [ligand_family(p, a) for p, a in zip(metadata.precatalyst, metadata.added_ligand)]
    metadata["base_class"] = metadata.base.map(base_class)
    assert metadata.metal.value_counts().to_dict() == {"Pd": 87, "Cu": 7, "Ni": 1}
    order = metadata.condition_id.tolist()
    blocks = {}
    for panel in ("B", "C"):
        cells = sheets[f"Fig1{panel}"].copy()
        rows = cells[["row", "partner", "partner_smiles", "pair"]].drop_duplicates().sort_values("row")
        matrix = cells.pivot(index="row", columns="condition_id", values="colour_value_points").loc[rows.row, order]
        assert matrix.shape == (7 if panel == "B" else 10, 95)
        rr = sheets["Fig1BC_r"]
        correlations = rr[rr.panel.eq(panel) & rr.row_i.notna()].copy()
        mean_row = rr[rr.panel.eq(panel) & rr.row_i.isna()]
        assert len(mean_row) == 1
        mean = float(mean_row.r.iloc[0])
        assert abs(correlations.r.mean() - mean) < 1e-12
        cells["original_column"] = cells.column
        cells["column"] = cells.condition_id.map(dict(zip(order, range(1, 96))))
        blocks[panel] = dict(rows=rows, matrix=matrix, correlations=correlations, mean=mean,
                             cells=cells.sort_values(["row", "column"]).reset_index(drop=True))
    assert blocks["B"]["matrix"].columns.tolist() == blocks["C"]["matrix"].columns.tolist() == order
    shared = sheets["Fig1A"].loc[sheets["Fig1A"].role.eq("pair"), "best_yield_pct"]
    assert len(shared) == 1 and f"{shared.iloc[0]:.0f}" == "91"
    assert set(sheets["Fig1A"].loc[sheets["Fig1A"].role.ne("pair"), "smiles"]) == {HALIDE, AMINE}
    return sheets, metadata, blocks, float(shared.iloc[0])


def segments(values):
    start = 0
    result = []
    for stop in range(1, len(values) + 1):
        if stop == len(values) or values[stop] != values[start]:
            result.append((start, stop, values[start]))
            start = stop
    return result
