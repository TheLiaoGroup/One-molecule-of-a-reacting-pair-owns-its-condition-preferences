"""Independent numerical checks for the revised main Figures 2 and 3.

The observed workbooks and artist records are checked against immutable yields,
closed-form least-squares calculations, and the existing analysis outputs.
No figure generator is imported.
"""
from functools import lru_cache
from pathlib import Path
import ast
import hashlib
import itertools
import json
import re

import numpy as np
import pandas as pd
from rdkit import Chem

TOL = 1e-9
ORDER = ["Buchwald-Hartwig|1", "Buchwald-Hartwig|2", "C-O competitive reactions|1", "Heck|1", "Hiyama|1",
         "Kumada|1", "SNAr|1", "Sonogashira|1", "Suzuki|1", "Ullmann|1", "Acylation|1", "Chan-Lam|1",
         "Esterification|1", "Grignard|1", "Reductive-amination|1"]
ADDITIVES = ["AcOH", "MsOH", "PTSA", "TFA", "none", "DIPEA", "TEA", "2,6-lutidine", "Py", "DBU",
             "Cs2CO3", "K2CO3", "KHCO3", "KOAc", "KOH", "KOtBu", "NaOMe", "NaOtAm"]
SOLVENTS = ["DMSO", "THF", "t-AmOH", "NMP", "CPME", "MeCN"]
CLASS_SIZES = [("Acids", 4), ("No additive", 1), ("Amine bases", 5),
               ("Carbonates, bicarbonate and acetate", 4), ("Hydroxide and alkoxides", 4)]
HALIDE = "Cc1cc(F)ccc1C#N"
NUC = "c1cc2[nH]ccc2cn1"
PRODUCT = "Cc1cc(-n2ccc3cnccc32)ccc1C#N"


def _path(root, *names):
    found = [root / name for name in names if (root / name).is_file()]
    assert found, "Missing independent input: " + " or ".join(names)
    return found[0]


def _result(root, name):
    return _path(root, "05_DATA/Results/" + name, "results/" + name)


def _close(actual, expected, label, tol=TOL):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape, f"{label}: shape {a.shape} != {b.shape}"
    assert np.isfinite(a).all() and np.isfinite(b).all(), f"{label}: non-finite value"
    deviation = float(np.max(np.abs(a - b))) if a.size else 0.0
    assert deviation <= tol, f"{label}: max difference {deviation:.5g} > {tol}"
    return deviation


def _canonical(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    assert molecule is not None, f"Invalid molecule: {smiles}"
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


@lru_cache(maxsize=2)
def _raw(root_string):
    root = Path(root_string)
    full = root / "05_DATA/Release/00_RAW_IMMUTABLE/DATASET_analysis.csv.gz"
    columns = ["screen", "condition_id", "subpair", "sub_1_smiles", "sub_2_smiles", "product_smiles", "yield"]
    if full.is_file():
        frame = pd.read_csv(full, usecols=columns, low_memory=False)
        assert len(frame) == 191357
    else:
        frame = pd.concat([pd.read_csv(_path(root, f"data/{name}"), usecols=columns, low_memory=False)
                           for name in ("measurements_A1_analysis_arrays.csv.gz", "measurements_A3_hiyama_duplicate.csv.gz")], ignore_index=True)
    frame["y"] = pd.to_numeric(frame["yield"], errors="coerce")
    duplicate = (frame.screen == "Hiyama|1") & (frame.condition_id == "C46")
    a1 = frame[frame.screen.isin(ORDER) & frame.y.notna() & ~duplicate].copy()
    a3 = frame[duplicate & frame.y.notna()].copy()
    assert len(a1) == 177039 and a1.screen.nunique() == 15 and len(a3) == 74
    return frame, a1, a3


def _lines(yield_matrix, condition_means):
    residual = np.full_like(yield_matrix, np.nan)
    fitted = residual.copy()
    leverage = residual.copy()
    for i, row in enumerate(yield_matrix):
        finite = np.isfinite(row) & np.isfinite(condition_means)
        x, y = condition_means[finite], row[finite]
        xx = np.sum((x - x.mean()) ** 2)
        assert len(x) >= 3 and xx > 0
        slope = np.sum((x - x.mean()) * (y - y.mean())) / xx
        fitted[i, finite] = y.mean() + slope * (x - x.mean())
        residual[i, finite] = y - fitted[i, finite]
        leverage[i, finite] = 1 / len(x) + (x - x.mean()) ** 2 / xx
    return residual, fitted, leverage


def _patterns(frame, product_forming=True):
    table = frame.pivot(index="subpair", columns="condition_id", values="y")
    yields = table.to_numpy(float)
    keep = np.nanmax(yields, axis=1) >= 0.10 if product_forming else np.ones(len(yields), dtype=bool)
    selected = yields[keep]
    residual, fitted, leverage = _lines(selected, pd.DataFrame(selected).mean(axis=0).to_numpy())
    pairs = table.index[keep]
    return dict(table=table, yields=yields, keep=keep, E=residual, F=fitted, H=leverage,
                a=np.asarray([s.split(" >> ")[0] for s in pairs]), b=np.asarray([s.split(" >> ")[1] for s in pairs]))


def _molecule_t(patterns, role):
    residual = patterns["E"]
    molecules = patterns["a" if role == 1 else "b"]
    variance = np.mean(np.nanmean(residual ** 2, axis=1))
    values = {}
    for molecule in sorted(set(molecules)):
        products = []
        for i, j in itertools.permutations(np.flatnonzero(molecules == molecule), 2):
            finite = np.isfinite(residual[i]) & np.isfinite(residual[j])
            if finite.any():
                products.append(float(np.mean(residual[i, finite] * residual[j, finite])))
        if products:
            values[_canonical(molecule)] = (float(np.mean(products) / variance), len(products))
    return values


@lru_cache(maxsize=2)
def _anatomy(root_string):
    root = Path(root_string)
    _, a1, a3 = _raw(root_string)
    calibration = _patterns(a1[a1.screen == "Hiyama|1"], product_forming=False)
    column = list(calibration["table"].columns).index("C43")
    replicate = a3.set_index("subpair").y.reindex(calibration["table"].index).to_numpy(float)
    fitted = calibration["F"][:, column]
    e43 = calibration["E"][:, column]
    e46 = replicate - fitted
    assert np.isfinite(e43).all() and np.isfinite(e46).all()
    mean_square = np.mean((e43 ** 2 + e46 ** 2) / 2)
    agreement = np.mean(e43 * e46) / mean_square
    k = (1 - agreement) * mean_square / np.mean(np.maximum(0.07 * np.clip(fitted, 0, 1), 0.012) ** 2 * (1 - calibration["H"][:, column]))
    summary = json.loads(_result(root, "display_inputs/Anatomy.json").read_text(encoding="utf-8"))
    _close([agreement, k], [summary["share"]["hiyama_duplicate_reliability"], summary["share"]["noise_scale"]], "Independent noise calibration")
    rows = []
    for screen in ORDER:
        group = a1[a1.screen == screen]
        p = _patterns(group)
        e, a, b = p["E"], p["a"], p["b"]
        variance = np.mean(np.nanmean(e ** 2, axis=1))
        shares = []
        for same, other in ((a, b), (b, a)):
            products = []
            for molecule in sorted(set(same)):
                for i, j in itertools.permutations(np.flatnonzero(same == molecule), 2):
                    finite = np.isfinite(e[i]) & np.isfinite(e[j])
                    if other[i] != other[j] and finite.any():
                        products.append(float(np.mean(e[i, finite] * e[j, finite])))
            assert products
            shares.append(float(np.mean(products) / variance))
        sig2 = k * np.maximum(0.07 * np.clip(p["F"], 0, 1), 0.012) ** 2 * (1 - p["H"])
        noise = np.mean(np.nansum(sig2, axis=1) / np.isfinite(e).sum(axis=1)) / variance
        owner = int(np.argmax(shares)) + 1
        rows.append(dict(screen=screen, share1=shares[0], share2=shares[1], share=sum(shares), noise=noise,
                         combination=1 - sum(shares) - noise, owner_role=owner, owner_fraction=max(shares) / sum(shares),
                         pairs=len(p["yields"]), product_forming=int(p["keep"].sum())))
    values = pd.DataFrame(rows).set_index("screen")
    result = pd.read_csv(_result(root, "display_inputs/Anatomy_arrays.csv")).set_index("screen").loc[ORDER]
    columns = ["share1", "share2", "share", "noise", "combination", "owner_fraction"]
    deviation = _close(values[columns], result[columns], "Independent anatomy versus analysis")
    assert np.array_equal(values.owner_role, result.owner_role)
    assert int(values.pairs.sum()) == 2207 and int(values.product_forming.sum()) == 1539
    return values, float(k), deviation


def _ledger(tables, expected):
    numbers = tables["Numbers"]
    assert len(numbers) == len(expected)
    keys = [(row.panel, row.item, row.quantity) for row in numbers.itertuples()]
    assert len(set(keys)) == len(keys) and set(keys) == set(expected), "Ledger entries missing, duplicated, or unexpected"
    for row, key in zip(numbers.itertuples(), keys):
        value, display = expected[key]
        _close([row.value], [value], "Ledger " + repr(key))
        if display is not None:
            _close([row.plotted_value], [display], "Displayed ledger " + repr(key))
    return len(numbers)


def _figure2(root, tables):
    values, noise_scale, deviation = _anatomy(str(root))
    observed = tables["Fig2A_Arrays"]
    assert observed.screen.tolist() == ORDER and observed.plot_row.tolist() == list(range(1, 16))
    observed = observed.set_index("screen")
    columns = ["share1", "share2", "share", "noise", "combination", "owner_fraction"]
    _close(observed[columns], values[columns], "Figure 2A arrays")
    assert np.array_equal(observed.owner_role, values.owner_role)
    assert int((observed[["share1", "share2"]] < 0).sum().sum()) == 3
    for column, position in (("owner_chip_column", "left"), ("partner_chip_column", "right"),
                             ("owner_share_bar_position", "upper"), ("partner_share_bar_position", "lower")):
        assert observed[column].eq(position).all()
    role_source = _path(root, "06_CODE/Core/00_common/ctb/history.py", "code/ctb/history.py")
    role_ast = ast.parse(role_source.read_text(encoding="utf-8"))
    roles = next(ast.literal_eval(node.value) for node in role_ast.body if isinstance(node, ast.Assign)
                 and any(getattr(target, "id", None) == "ROLE" for target in node.targets))
    for row in observed.itertuples():
        for side in (1, 2):
            expected = roles[row.Index][side - 1][0].replace("\u2013", "-")
            assert getattr(row, f"role{side}") == expected
        owner = int(row.owner_role)
        partner = 3 - owner
        _close([row.owner_share, row.partner_share],
               [getattr(row, f"share{owner}"), getattr(row, f"share{partner}")],
               "Figure 2A owner/partner copied shares " + row.Index)
        assert row.owner_display_role == getattr(row, f"role{owner}")
        assert row.partner_display_role == getattr(row, f"role{partner}")
    conditions = pd.read_csv(_path(root, "10_PUBLIC_RELEASE_CANDIDATE/data/conditions.csv", "data/conditions.csv"))
    conditions = conditions[conditions.rxn_class == "SNAr"].copy()
    conditions["base_or_acid"] = conditions["Reagent3_Standardized Name"].fillna("none")
    assert len(conditions) == conditions.condition_id.nunique() == 47
    assert set(conditions.base_or_acid) == set(ADDITIVES) and set(conditions["Standardized Name"]) == set(SOLVENTS)
    assert all(conditions[f"Reagent{i}_id"].isna().all() for i in (1, 2, 4, 5, 6))
    conditions["matrix_column"] = conditions.base_or_acid.map({s: i + 1 for i, s in enumerate(ADDITIVES)})
    conditions["matrix_row"] = conditions["Standardized Name"].map({s: i + 1 for i, s in enumerate(SOLVENTS)})
    assert len(conditions[["matrix_row", "matrix_column"]].drop_duplicates()) == 47
    cells = tables["Fig2B_ObservedCells"].set_index("condition_id")
    reference = conditions.set_index("condition_id")
    assert set(cells.index) == set(reference.index) and len(cells) == 47
    _close(cells.loc[reference.index, ["matrix_row", "matrix_column"]], reference[["matrix_row", "matrix_column"]], "Figure 2B matrix positions", 0)
    assert cells.loc[reference.index, "base_or_acid"].tolist() == reference.base_or_acid.tolist()
    assert cells.loc[reference.index, "Standardized Name"].tolist() == reference["Standardized Name"].tolist()
    column_table = tables["Fig2B_Columns"]
    assert column_table.additive_name.tolist() == ADDITIVES and column_table.matrix_column.tolist() == list(range(1, 19))
    assert column_table.additive_class.tolist() == [label for label, size in CLASS_SIZES for _ in range(size)]
    classes = dict(zip(ADDITIVES, [label for label, size in CLASS_SIZES for _ in range(size)]))
    assert cells.loc[reference.index, "additive_class"].tolist() == reference.base_or_acid.map(classes).tolist()
    copied_conditions = tables["Fig2B_Conditions"].set_index("condition_id")
    assert len(copied_conditions) == 47 and set(copied_conditions.index) == set(reference.index)
    for column in reference.columns:
        actual = copied_conditions.loc[reference.index, column].fillna("<missing>").astype(str)
        expected_column = reference[column].fillna("<missing>").astype(str)
        assert actual.tolist() == expected_column.tolist(), "Copied condition field differs: " + column
    raw, _, _ = _raw(str(root))
    pair = raw[(raw.screen == "SNAr|1") & (raw.sub_1_smiles == NUC) & (raw.sub_2_smiles == HALIDE)]
    assert len(pair) == pair.condition_id.nunique() == 47 and set(pair.condition_id) == set(reference.index)
    assert pair.product_smiles.nunique() == 1 and pair.product_smiles.iloc[0] == PRODUCT
    measurements = tables["Fig2B_Measurements"].set_index("condition_id")
    assert set(measurements.index) == set(pair.condition_id) and len(measurements) == 47
    _close(pd.to_numeric(measurements.loc[pair.condition_id, "yield"]), pd.to_numeric(pair["yield"]), "Figure 2B immutable measured yields")
    reaction = tables["Fig2B_Reaction"].set_index("role")
    assert all(_canonical(reaction.loc[role, "smiles"]) == _canonical(smiles)
               for role, smiles in (("halide", HALIDE), ("N-H nucleophile", NUC), ("product", PRODUCT)))
    expected = {( "A", key, column): (float(row[column]), 100 * float(row[column]))
                for key, row in values.iterrows() for column in ("share", "combination", "noise", "share1", "share2")}
    for quantity, column in (("median molecular share", "share"), ("median owner fraction", "owner_fraction")):
        value = float(values[column].median())
        expected[("A", "15 arrays", quantity)] = value, round(100 * value)
    for column in ("share1", "share2"):
        value = float(values.loc["SNAr|1", column])
        expected[("B", "SNAr|1", column)] = value, round(100 * value, 1)
    expected[("B", "condition matrix", "measured combinations")] = 47, 47
    count = _ledger(tables, expected)
    return dict(numerical_ledger_rows=count, independent_arrays=15, independent_noise_scale=noise_scale,
                analysis_max_difference=deviation, observed_combinations=47, additive_columns=18, solvents=6,
                signed_negative_shares=3)


@lru_cache(maxsize=2)
def _directional(root_string):
    root = Path(root_string)
    _, a1, _ = _raw(root_string)
    cohorts = pd.read_csv(_path(root, "05_DATA/Analysis/directional_cohorts/DESIGN_COUNTS.csv", "design/DESIGN_COUNTS.csv"))
    primary = cohorts[(cohorts.set == "G1") & cohorts.array_clear_owner]
    arrays = sorted(set(primary.array) | set(cohorts.loc[cohorts.triple_context, "array"]))
    lookup = {(molecule, array, role): result for array in arrays for role in (1, 2)
              for molecule, result in _molecule_t(_patterns(a1[a1.screen == array]), role).items()}
    result_t = pd.read_csv(_result(root, "directional/phase1_T.csv")).set_index(["molecule", "array", "role"])
    context_rows = []
    molecules = []
    for molecule, group in primary.groupby("molecule"):
        owner, nonowner = [], []
        for row in group.itertuples():
            key = molecule, row.array, int(row.role)
            value, count = lookup[key]
            ref = result_t.loc[key]
            _close([value], [ref["T"]], "Independent per-array T " + repr(key))
            assert count == ref.n_ordered_pair_pairs
            context_rows.append(dict(molecule=molecule, array=row.array, role=int(row.role), context=row.context, T=value, ordered_pairs_of_pairs=count))
            (owner if row.context == "owner" else nonowner).append(value)
        to, tn = np.mean(owner), np.mean(nonowner)
        molecules.append(dict(molecule=molecule, stratum=group.stratum.iloc[0], T_O=to, T_N=tn, D=to - tn))
    a = pd.DataFrame(molecules)
    a = pd.concat([a[a.stratum == group].sort_values(["D", "molecule"], ascending=[False, True]) for group in ("amine", "acid")], ignore_index=True)
    assert len(a) == 23 and len(context_rows) == 55
    tri = sorted(cohorts.loc[cohorts.triple_context, "molecule"].unique(), key=lambda m: (Chem.MolFromSmiles(m).GetNumHeavyAtoms(), m))
    b = pd.DataFrame([dict(plot_label=f"M{i}", smiles=molecule,
                          **{column: lookup[(molecule, array, 1)][0] for column, array in
                             (("T_amide", "Acylation|1"), ("T_BH1", "Buchwald-Hartwig|1"), ("T_BH2", "Buchwald-Hartwig|2"))})
                      for i, molecule in enumerate(tri, 1)])
    assert len(b) == 8
    return a, b, pd.DataFrame(context_rows), cohorts


def _figure3(root, tables):
    a, b, contexts, cohorts = _directional(str(root))
    source_a = tables["Fig3A_Molecules"]
    assert source_a.molecule.tolist() == a.molecule.tolist() and source_a.stratum.tolist() == a.stratum.tolist()
    assert source_a.plot_row.tolist() == list(range(1, 24))
    deviation = _close(source_a[["T_O", "T_N", "D"]], a[["T_O", "T_N", "D"]], "Figure 3A independent values")
    result_d = pd.read_csv(_result(root, "directional/phase1_D.csv")).set_index("molecule").loc[a.molecule]
    _close(a[["T_O", "T_N", "D"]], result_d[["T_O", "T_N", "D"]], "Independent D versus analysis")
    for column in ("owner_arrays", "nonowner_arrays"):
        assert source_a[column].tolist() == result_d[column].tolist()
    _close(source_a.dTanimoto, result_d.dTanimoto, "Retained directional descriptor versus analysis")
    source_contexts = tables["Fig3A_Contexts"].rename(columns={"smiles": "molecule"}).set_index(["molecule", "array", "role"])
    reference_contexts = contexts.set_index(["molecule", "array", "role"])
    assert len(source_contexts) == len(reference_contexts) == 55 and set(source_contexts.index) == set(reference_contexts.index)
    _close(source_contexts.loc[reference_contexts.index, ["T", "ordered_pairs_of_pairs"]], reference_contexts[["T", "ordered_pairs_of_pairs"]], "Figure 3A all 55 per-array contexts")
    assert source_contexts.loc[reference_contexts.index, "context"].tolist() == reference_contexts.context.tolist()
    source_b = tables["Fig3B_EightAmines"]
    assert source_b.plot_label.tolist() == b.plot_label.tolist() and source_b.smiles.tolist() == b.smiles.tolist()
    columns = ["T_amide", "T_BH1", "T_BH2"]
    _close(source_b[columns], b[columns], "Figure 3B independent values")
    cn = b[["T_BH1", "T_BH2"]].mean(axis=1)
    differences = b.T_amide - cn
    _close(source_b.CN_mean, cn, "Figure 3B C-N mean")
    _close(source_b.R, differences, "Figure 3B signed differences")
    assert set(b.loc[differences < 0, "plot_label"]) == {"M3", "M8"}
    partner_counts = cohorts.set_index(["molecule", "array", "role"]).n_pf_partners
    for row in source_b.itertuples():
        assert row.heavy_atoms == Chem.MolFromSmiles(row.smiles).GetNumHeavyAtoms()
        assert pd.notna(row.amide_above_CN_mean) and row.amide_above_CN_mean in (0, 1, False, True)
        assert pd.notna(row.amide_above_both) and row.amide_above_both in (0, 1, False, True)
        assert bool(row.amide_above_CN_mean) == bool(row.R > 0)
        assert bool(row.amide_above_both) == bool(row.T_amide > max(row.T_BH1, row.T_BH2))
        for column, array in (("partners_amide", "Acylation|1"), ("partners_BH1", "Buchwald-Hartwig|1"), ("partners_BH2", "Buchwald-Hartwig|2")):
            assert getattr(row, column) == partner_counts.loc[(row.smiles, array, 1)]
    summary = json.loads(_result(root, "directional/phase1_summary.json").read_text(encoding="utf-8"))
    summaries = {("A", "mean"): float(a.D.mean()),
                 ("C", "mean_amide_minus_BH1"): float((b.T_amide - b.T_BH1).mean()),
                 ("C", "mean_amide_minus_BH2"): float((b.T_amide - b.T_BH2).mean()), ("C", "mean_R"): float(differences.mean())}
    for (section, field), value in summaries.items():
        _close([value], [summary[section][field]], "Independent summary " + field)
    expected = {("A", row.molecule, column): (float(getattr(row, column)), None)
                for row in a.itertuples() for column in ("T_O", "T_N")}
    expected.update({("B", row.plot_label, column): (float(getattr(row, column)), None)
                     for row in b.itertuples() for column in columns})
    expected.update({("B", "mean line", column): (float(b[column].mean()), None) for column in columns})
    for panel, section, field in (("A", "A", "mean"), ("B", "C", "mean_amide_minus_BH1"), ("B", "C", "mean_amide_minus_BH2")):
        value = summaries[(section, field)]
        expected[(panel, field, "mean difference")] = value, round(value, 3 if panel == "A" else 2)
    count = _ledger(tables, expected)
    raw_summary = tables["Fig3_RawSummary"].set_index(["section", "quantity"])
    assert ("C", "mean_R") in raw_summary.index
    for (section, quantity), row in raw_summary.iterrows():
        _close([row.value], [summary[section][quantity]], "Retained raw summary " + section + "." + quantity)
    manifest = pd.read_csv(_path(root, "02_FIGURES/ChemDraw/structures_manifest.csv", "structures/structures_manifest.csv")).set_index("name")
    specification = pd.read_csv(_path(root, "07_METHODS_AND_PROVENANCE/FIGURE_STRUCTURES_PROMOTION_READY.csv", "structures/FIGURE_STRUCTURES_PROMOTION_READY.csv"))
    specification = specification[specification.label.isin(b.plot_label)].set_index("label")
    assert len(specification) == 8 and specification.index.is_unique
    checked = []
    for row in b.itertuples():
        key = f"Fig2_b_{row.plot_label}"
        assert _canonical(manifest.loc[key, "smiles"]) == _canonical(specification.loc[row.plot_label, "smiles"]) == _canonical(row.smiles)
        for extension in ("svg", "cdxml"):
            path = _path(root, f"02_FIGURES/ChemDraw/{key}.{extension}", f"structures/{key}.{extension}")
            assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest.loc[key, f"{extension}_sha256"]
            if extension == "cdxml":
                drawn = Chem.MolsFromCDXML(path.read_text(encoding="utf-8"))
                assert len(drawn) == 1 and Chem.MolToSmiles(drawn[0], isomericSmiles=True) == _canonical(row.smiles)
        checked.append(row.plot_label)
    for frame in (source_a, source_b):
        assert not frame.name.astype(str).str.contains(r"(?:^|[-,(])n[-,]", regex=True).any()
    names = source_a.set_index("molecule").name
    assert names.loc["CNc1ccc(OC)cc1"] == "4-methoxy-N-methylaniline" and names.loc["CCNc1ccccc1"] == "N-ethylaniline"
    return dict(numerical_ledger_rows=count, independent_molecules=23, independent_per_array_contexts=55,
                independent_slope_values=24, analysis_max_difference=deviation, verified_cdxml_labels=checked)


def audit_values(number, root, source_path):
    root = Path(root).resolve()
    tables = pd.read_excel(source_path, sheet_name=None)
    if number == 2:
        return _figure2(root, tables)
    assert number == 3
    return _figure3(root, tables)


def self_test_values(number, root, source_path):
    """A coherent workbook change must still fail against independent yields."""
    tables = pd.read_excel(source_path, sheet_name=None)
    bad = {name: frame.copy(deep=True) for name, frame in tables.items()}
    if number == 2:
        first = bad["Fig2A_Arrays"].index[0]
        key = bad["Fig2A_Arrays"].loc[first, "screen"]
        for column, delta in (("share1", 0.01), ("share", 0.01), ("combination", -0.01)):
            bad["Fig2A_Arrays"].loc[first, column] += delta
            selected = (bad["Numbers"].panel == "A") & (bad["Numbers"].item == key) & (bad["Numbers"].quantity == column)
            bad["Numbers"].loc[selected, "value"] += delta
            bad["Numbers"].loc[selected, "plotted_value"] += 100 * delta
        function = _figure2
    else:
        assert number == 3
        first = bad["Fig3A_Molecules"].index[0]
        key = bad["Fig3A_Molecules"].loc[first, "molecule"]
        bad["Fig3A_Molecules"].loc[first, ["T_O", "D"]] += 0.01
        selected = (bad["Numbers"].panel == "A") & (bad["Numbers"].item == key) & (bad["Numbers"].quantity == "T_O")
        bad["Numbers"].loc[selected, "value"] += 0.01
        function = _figure3
    try:
        function(Path(root).resolve(), bad)
    except AssertionError:
        return dict(coherent_source_and_ledger_mutation_rejected=True)
    raise AssertionError("Planted coherent numerical change was accepted")


def audit_artists(number, root, source_path, readback):
    """Check every quantitative mark against independent values in data coordinates."""
    root = Path(root).resolve()
    axes = readback["axes"]
    if number == 2:
        values, _, _ = _anatomy(str(root))
        composition = next(axis for axis in axes if np.allclose(axis["xlim"], [0, 100]) and len(axis.get("patches", [])) == 45)
        shares = next(axis for axis in axes if np.allclose(axis["xlim"], [-8, 50]) and len(axis.get("patches", [])) == 30)
        for i, key in enumerate(ORDER):
            row = values.loc[key]
            center = 22 + 5.1 * i + (2.5 if i >= 10 else 0)
            left = 0.0
            for j, column in enumerate(("share", "combination", "noise")):
                patch = composition["patches"][3 * i + j]
                _close([patch["width"], patch["height"], *patch["xy"]], [100 * row[column], 3.5, left, center - 1.75], "Figure 2A composition artist")
                left += 100 * row[column]
            for j, side in enumerate((int(row.owner_role), 3 - int(row.owner_role))):
                patch = shares["patches"][2 * i + j]
                y = center + (-0.9 if j == 0 else 0.9)
                _close([patch["width"], patch["height"], *patch["xy"]], [100 * row[f"share{side}"], 1.5, 0, y - 0.75], "Figure 2A signed share artist")
        dots = [collection for axis in axes for collection in axis.get("collections", []) if collection.get("gid") == "B|47_measured_combinations"]
        assert len(dots) == 1
        tables = pd.read_excel(source_path, sheet_name=None)
        cells = tables["Fig2B_ObservedCells"]
        observed = {tuple(point) for point in dots[0]["offsets"]}
        expected = set(map(tuple, cells[["matrix_column", "matrix_row"]].to_numpy()))
        assert len(dots[0]["offsets"]) == len(observed) == len(expected) == 47 and observed == expected
        return dict(composition_bars=45, signed_role_bars=30, measured_dots=47)
    assert number == 3
    a, b, _, _ = _directional(str(root))
    dumbbell = next(axis for axis in axes if np.allclose(axis["xlim"], [-0.52, 0.82]))
    lines = dumbbell["lines"][1:]
    assert len(lines) == 69
    for i, row in enumerate(a.itertuples()):
        y = 23 + 4.2 * i + (6 if i >= 19 else 0)
        connector, nonowner, owner = lines[3 * i:3 * i + 3]
        _close(connector["x"], [row.T_N, row.T_O], "Figure 3A connector values")
        _close(connector["y"], [y, y], "Figure 3A connector row")
        for mark, value in ((nonowner, row.T_N), (owner, row.T_O)):
            _close(mark["x"], [value], "Figure 3A point value")
            _close(mark["y"], [y], "Figure 3A point row")
            assert mark["marker"] == ("o" if row.stratum == "amine" else "s")
        assert nonowner["mfc"] in ("white", "#ffffff", "#FFFFFF") and owner["mfc"] != nonowner["mfc"]
    slope = next(axis for axis in axes if np.allclose(axis["xlim"], [-0.1, 2.1]))
    trajectories = [line for line in slope["lines"] if line["x"] == [0, 1, 2]]
    assert len(trajectories) == 9
    values = b[["T_amide", "T_BH1", "T_BH2"]].to_numpy(float)
    for line, expected in zip(trajectories, [*values, values.mean(axis=0)]):
        _close(line["y"], expected, "Figure 3B trajectory")
    leaders = [line for line in slope["lines"] if line["x"] == [2.02, 2.22]]
    assert len(leaders) == 8
    _close([line["y"][0] for line in leaders], values[:, 2], "Figure 3B leader endpoints")
    for first, second in itertools.combinations(leaders, 2):
        assert np.prod(np.asarray(first["y"]) - np.asarray(second["y"])) >= 0, "Crossed endpoint leaders"
    return dict(context_points=46, context_connectors=23, molecule_trajectories=8, mean_trajectory=1, noncrossing_leader_pairs=28)
