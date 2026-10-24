"""Independent numerical and artist checks for formal Figures 3 and 5.

Only raw measurements, frozen role/fold records and retained target losses are
inputs to the calculations below. No figure generator or figure helper is
imported. Closed-form anatomy equations retain the v2 independent check's 1e-9
criterion; descriptive selection uses the frozen normal-equation definition.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

TOL = 1e-9
KS = (1, 2, 3, 4, 6)
ORDER = ["Buchwald-Hartwig|1", "Buchwald-Hartwig|2", "C-O competitive reactions|1", "Heck|1", "Hiyama|1",
         "Kumada|1", "SNAr|1", "Sonogashira|1", "Suzuki|1", "Ullmann|1", "Acylation|1", "Chan-Lam|1",
         "Esterification|1", "Grignard|1", "Reductive-amination|1"]
RULE_FILE = "07_METHODS_AND_PROVENANCE/frozen_analysis_plans/figure_selection_v3/RULE.md"
RULE_SHA = "1eef02cbdb55db159135883820e4a83e353e7da0f057a2fd258f96082143c02d"
LOSS_FILES = {
    "Set A": "reserve_validation/record_transfer_losses.csv",
    "Set B": "cross_library/target_losses.csv",
    "k2": "directional/phase2_setAB_k2.csv",
    "within": "directional/phase3_transport.csv",
    "depth": "primary/anatomy_depth.csv",
}
ADDITIVES = ["AcOH", "MsOH", "PTSA", "TFA", "none", "DIPEA", "TEA", "2,6-lutidine", "Py", "DBU",
             "Cs2CO3", "K2CO3", "KHCO3", "KOAc", "KOH", "KOtBu", "NaOMe", "NaOtAm"]
SOLVENTS = ["DMSO", "THF", "t-AmOH", "NMP", "CPME", "MeCN"]
CLASS_SIZES = [("Acids", 4), ("No additive", 1), ("Amine bases", 5),
               ("Carbonates, bicarbonate and acetate", 4), ("Hydroxide and alkoxides", 4)]
HALIDE = "Cc1cc(F)ccc1C#N"
NUCLEOPHILE = "c1cc2[nH]ccc2cn1"


def _path(root, *names):
    for name in names:
        candidate = Path(root) / name
        if candidate.is_file():
            return candidate
    raise AssertionError("Missing independent input: " + " or ".join(names))


def _result(root, name):
    return _path(root, "05_DATA/Results/" + name, "results/" + name)


def _close(actual, expected, label, tol=TOL):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape, f"{label}: shape {a.shape} != {b.shape}"
    assert np.isfinite(a).all() and np.isfinite(b).all(), f"{label}: non-finite value"
    difference = float(np.max(np.abs(a - b))) if a.size else 0.0
    assert difference <= tol, f"{label}: max difference {difference:.5g} > {tol}"
    return difference


def _nullable_close(actual, expected, label):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b)), label + ": missingness differs"
    return _close(a[np.isfinite(b)], b[np.isfinite(b)], label)


def _canonical(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    assert molecule is not None, f"Invalid molecule: {smiles}"
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def _rule(root):
    actual = hashlib.sha256(_path(root, RULE_FILE, "figure_inputs/selection_v3/RULE.md").read_bytes()).hexdigest()
    assert actual == RULE_SHA, "Frozen Figure v3 selection rule changed"


def _frame(actual, expected, keys, numeric=(), strings=(), label="Source rows"):
    """Check identities before aligning values, including duplicate-row rejection."""
    assert len(actual) == len(expected), f"{label}: row count differs"
    assert not actual.duplicated(keys).any() and not expected.duplicated(keys).any(), label + ": duplicate identities"
    a, b = actual.set_index(keys), expected.set_index(keys)
    assert set(a.index) == set(b.index), label + ": identities differ"
    a = a.loc[b.index]
    if numeric:
        _nullable_close(a[list(numeric)], b[list(numeric)], label)
    for column in strings:
        assert a[column].fillna("<missing>").astype(str).tolist() == b[column].fillna("<missing>").astype(str).tolist(), label + ": " + column


@lru_cache(maxsize=2)
def _raw(root_string):
    root = Path(root_string)
    columns = ["screen", "condition_id", "subpair", "sub_1_smiles", "sub_2_smiles", "yield"]
    full = root / "05_DATA/Release/00_RAW_IMMUTABLE/DATASET_analysis.csv.gz"
    if full.is_file():
        header = pd.read_csv(full, nrows=0).columns
        raw = pd.read_csv(full, usecols=columns, low_memory=False)
        # The public materializer contains exactly A1+A3, rather than the
        # private immutable workbook's additional non-analysis arrays.
        expected_rows = 177113 if "public_record_id" in header else 191357
        assert len(raw) == expected_rows, "Independent immutable measurement count"
    else:
        raw = pd.concat([pd.read_csv(_path(root, "data/" + name), usecols=columns, low_memory=False)
                         for name in ("measurements_A1_analysis_arrays.csv.gz", "measurements_A3_hiyama_duplicate.csv.gz")], ignore_index=True)
    raw["y"] = pd.to_numeric(raw["yield"], errors="coerce")
    assert raw.subpair.eq(raw.sub_1_smiles + " >> " + raw.sub_2_smiles).all()
    duplicate = raw.screen.eq("Hiyama|1") & raw.condition_id.eq("C46")
    a1 = raw[raw.screen.isin(ORDER) & raw.y.notna() & ~duplicate].copy()
    a3 = raw[duplicate & raw.y.notna()].copy()
    assert len(a1) == 177039 and a1.screen.nunique() == 15 and len(a3) == 74
    return raw, a1, a3


def _patterns(frame, product_forming=True):
    table = frame.pivot(index="subpair", columns="condition_id", values="y")
    matrix = table.to_numpy(float)
    keep = np.nanmax(matrix, axis=1) >= 0.10 if product_forming else np.ones(len(matrix), dtype=bool)
    selected = matrix[keep]
    signature = pd.DataFrame(selected).mean(axis=0).to_numpy()
    residual = np.full_like(selected, np.nan)
    fitted, leverage = residual.copy(), residual.copy()
    for i, row in enumerate(selected):
        finite = np.isfinite(row) & np.isfinite(signature)
        x, y = signature[finite], row[finite]
        xx = np.sum((x - x.mean()) ** 2)
        assert len(x) >= 3 and xx > 0
        slope = np.sum((x - x.mean()) * (y - y.mean())) / xx
        fitted[i, finite] = y.mean() + slope * (x - x.mean())
        residual[i, finite] = y - fitted[i, finite]
        leverage[i, finite] = 1 / len(x) + (x - x.mean()) ** 2 / xx
    pairs = table.index[keep]
    return dict(table=table, E=residual, F=fitted, H=leverage, yields=matrix, keep=keep,
                a=np.asarray([pair.split(" >> ")[0] for pair in pairs]),
                b=np.asarray([pair.split(" >> ")[1] for pair in pairs]))


@lru_cache(maxsize=2)
def _anatomy(root_string):
    """Retain the independent v2 closed-form noise and covariance equations."""
    root = Path(root_string)
    _, a1, a3 = _raw(root_string)
    calibration = _patterns(a1[a1.screen.eq("Hiyama|1")], product_forming=False)
    column = list(calibration["table"].columns).index("C43")
    replicate = a3.set_index("subpair").y.reindex(calibration["table"].index).to_numpy(float)
    fitted = calibration["F"][:, column]
    e43, e46 = calibration["E"][:, column], replicate - fitted
    assert np.isfinite(e43).all() and np.isfinite(e46).all()
    mean_square = np.mean((e43 ** 2 + e46 ** 2) / 2)
    agreement = np.mean(e43 * e46) / mean_square
    k = (1 - agreement) * mean_square / np.mean(np.maximum(0.07 * np.clip(fitted, 0, 1), 0.012) ** 2 * (1 - calibration["H"][:, column]))
    summary = json.loads(_result(root, "display_inputs/Anatomy.json").read_text(encoding="utf-8"))
    _close([agreement, k], [summary["share"]["hiyama_duplicate_reliability"], summary["share"]["noise_scale"]], "Independent noise calibration")
    rows = []
    for screen in ORDER:
        patterns = _patterns(a1[a1.screen.eq(screen)])
        e, a, b = patterns["E"], patterns["a"], patterns["b"]
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
        sig2 = k * np.maximum(0.07 * np.clip(patterns["F"], 0, 1), 0.012) ** 2 * (1 - patterns["H"])
        noise = np.mean(np.nansum(sig2, axis=1) / np.isfinite(e).sum(axis=1)) / variance
        owner = int(np.argmax(shares)) + 1
        rows.append(dict(screen=screen, share1=shares[0], share2=shares[1], share=sum(shares), noise=noise,
                         combination=1 - sum(shares) - noise, owner_role=owner,
                         owner_share=max(shares), partner_share=min(shares), owner_fraction=max(shares) / sum(shares),
                         pairs=len(patterns["yields"]), product_forming=int(patterns["keep"].sum())))
    values = pd.DataFrame(rows)
    retained = pd.read_csv(_result(root, "display_inputs/Anatomy_arrays.csv"))
    columns = ["share1", "share2", "share", "noise", "combination", "owner_fraction", "owner_share", "partner_share", "owner_role", "pairs", "product_forming"]
    _frame(retained, values, ["screen"], columns, label="Independent anatomy versus retained analysis")
    assert values.pairs.sum() == 2207 and values.product_forming.sum() == 1539
    return values, float(k)


def _descriptive(frame):
    pivot = frame.pivot(index="subpair", columns="condition_id", values="y")
    pivot = pivot[sorted(pivot.columns, key=lambda key: int(key[1:]))]
    matrix = pivot.to_numpy(float)
    active = np.nanmax(matrix, axis=1) >= 0.10
    matrix = matrix[active]
    signature = np.nanmean(matrix, axis=0)
    residual = np.full_like(matrix, np.nan)
    for i, row in enumerate(matrix):
        observed = np.isfinite(row) & np.isfinite(signature)
        design = np.c_[np.ones(observed.sum()), signature[observed]]
        coefficients = np.linalg.pinv(design.T @ design) @ design.T @ row[observed]
        residual[i, observed] = row[observed] - design @ coefficients
    return pivot.index[active].to_numpy(), pivot.columns.to_numpy(), 100 * residual


@lru_cache(maxsize=2)
def _owners(root_string):
    root = Path(root_string)
    _rule(root)
    raw, _, _ = _raw(root_string)
    raw = raw[raw.y.notna()]
    values, _ = _anatomy(root_string)
    role_table = pd.read_csv(_result(root, "primary/history_roles.csv")).set_index("screen")
    all_candidates, chosen = [], []
    for row in values.itertuples():
        role = int(row.owner_role)
        group = raw[raw.screen.eq(row.screen)]
        pairs, _, residual = _descriptive(group)
        identity = np.asarray([pair.split(" >> ")[role - 1] for pair in pairs])
        local = []
        for smiles in sorted(group[f"sub_{role}_smiles"].unique()):
            indices = np.flatnonzero(identity == smiles)
            correlations = []
            for i, j in itertools.combinations(indices, 2):
                observed = np.isfinite(residual[i]) & np.isfinite(residual[j])
                if observed.sum() < 2 or np.std(residual[i, observed]) == 0 or np.std(residual[j, observed]) == 0:
                    correlations.append(np.nan)
                else:
                    correlations.append(float(np.corrcoef(residual[i, observed], residual[j, observed])[0, 1]))
            local.append(dict(screen=row.screen, owner_role=role, owner_role_name=role_table.loc[row.screen, f"role_{role}"],
                              smiles=smiles, product_forming_partners=len(indices), pairwise_correlations=correlations,
                              mean_pairwise_r=float(np.mean(correlations)) if correlations else np.nan))
        threshold = next((k for k in (4, 3, 2) if any(c["product_forming_partners"] >= k for c in local)), None)
        assert threshold == (3 if row.screen == "Hiyama|1" else 4), "Unexpected Rule 2 threshold: " + row.screen
        eligible = [c for c in local if c["product_forming_partners"] >= threshold and np.isfinite(c["mean_pairwise_r"])]
        assert eligible, "Rule 2 has no finite candidate: " + row.screen
        median = float(np.median([c["mean_pairwise_r"] for c in eligible]))
        for candidate in local:
            candidate.update(threshold_used=threshold, eligible=candidate in eligible, role_median_r=median,
                             absolute_distance_to_median=abs(candidate["mean_pairwise_r"] - median), selected=False)
        selected = min(eligible, key=lambda c: (c["absolute_distance_to_median"], -c["product_forming_partners"], c["smiles"]))
        selected["selected"] = True
        chosen.append({key: value for key, value in selected.items() if key != "pairwise_correlations"})
        all_candidates.extend(local)
    return pd.DataFrame(chosen), pd.DataFrame(all_candidates)


def _reaction_product(halide, nucleophile):
    a, b = Chem.MolFromSmiles(halide), Chem.MolFromSmiles(nucleophile)
    leaving = [atom for atom in a.GetAtoms() if atom.GetSymbol() == "F" and atom.GetDegree() == 1 and atom.GetNeighbors()[0].GetIsAromatic()]
    nitrogen = [atom for atom in b.GetAtoms() if atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0]
    assert len(leaving) == len(nitrogen) == 1, "Ambiguous depicted SNAr edit"
    x, n = leaving[0], nitrogen[0]
    molecule = Chem.RWMol(Chem.CombineMols(a, b))
    target_n = molecule.GetAtomWithIdx(a.GetNumAtoms() + n.GetIdx())
    target_n.SetNumExplicitHs(max(0, target_n.GetNumExplicitHs() - 1))
    molecule.AddBond(x.GetNeighbors()[0].GetIdx(), a.GetNumAtoms() + n.GetIdx(), Chem.BondType.SINGLE)
    molecule.RemoveAtom(x.GetIdx())
    product = molecule.GetMol()
    Chem.SanitizeMol(product)
    return Chem.MolToSmiles(product, isomericSmiles=True)


@lru_cache(maxsize=2)
def _snar(root_string):
    root = Path(root_string)
    conditions = pd.read_csv(_path(root, "10_PUBLIC_RELEASE_CANDIDATE/data/conditions.csv", "data/conditions.csv"))
    conditions = conditions[conditions.rxn_class.eq("SNAr")].copy()
    conditions["base_or_acid"] = conditions["Reagent3_Standardized Name"].fillna("none")
    assert len(conditions) == conditions.condition_id.nunique() == 47
    assert set(conditions.base_or_acid) == set(ADDITIVES) and set(conditions["Standardized Name"]) == set(SOLVENTS)
    assert all(conditions[f"Reagent{i}_id"].isna().all() for i in (1, 2, 4, 5, 6))
    assert not conditions.duplicated(["base_or_acid", "Standardized Name"]).any()
    classes = {name: label for name, label in zip(ADDITIVES, [label for label, size in CLASS_SIZES for _ in range(size)])}
    conditions["matrix_column"] = conditions.base_or_acid.map({name: i + 1 for i, name in enumerate(ADDITIVES)})
    conditions["matrix_row"] = conditions["Standardized Name"].map({name: i + 1 for i, name in enumerate(SOLVENTS)})
    conditions["additive_class"] = conditions.base_or_acid.map(classes)
    raw, _, _ = _raw(root_string)
    pair = raw[raw.screen.eq("SNAr|1") & raw.sub_1_smiles.eq(NUCLEOPHILE) & raw.sub_2_smiles.eq(HALIDE)].copy()
    assert len(pair) == pair.condition_id.nunique() == 47 and set(pair.condition_id) == set(conditions.condition_id)
    product = _reaction_product(HALIDE, NUCLEOPHILE)
    return conditions, pair, product


def _structures(root, chosen, product):
    owner_folder = _path(root, "02_FIGURES/ChemDraw/v3/f3/structures_manifest.csv", "structures/v3/f3/structures_manifest.csv").parent
    manifest = pd.read_csv(owner_folder / "structures_manifest.csv").set_index("name")
    assert len(manifest) == 15 and manifest.index.is_unique
    tasks = [(owner_folder, manifest, f"Fig3_A_owner{i:02d}", row.smiles) for i, row in enumerate(chosen.itertuples(), 1)]
    base = _path(root, "02_FIGURES/ChemDraw/structures_manifest.csv", "structures/structures_manifest.csv").parent
    reaction_manifest = pd.read_csv(base / "structures_manifest.csv").set_index("name")
    tasks.extend((base, reaction_manifest, "Fig2_B_" + key, smiles)
                 for key, smiles in (("halide", HALIDE), ("nucleophile", NUCLEOPHILE), ("product", product)))
    for folder, records, key, smiles in tasks:
        assert _canonical(records.loc[key, "smiles"]) == _canonical(smiles), "Structure identity: " + key
        for extension in ("svg", "cdxml"):
            file = folder / (key + "." + extension)
            assert hashlib.sha256(file.read_bytes()).hexdigest() == records.loc[key, extension + "_sha256"], "Structure bytes: " + key
            if extension == "cdxml":
                drawn = Chem.MolsFromCDXML(file.read_text(encoding="utf-8"))
                assert len(drawn) == 1 and Chem.MolToSmiles(drawn[0], isomericSmiles=True) == _canonical(smiles), "CDXML connectivity: " + key
    structural_input = pd.read_csv(_path(root, "07_METHODS_AND_PROVENANCE/figure_inputs/figure2_structures.csv", "figure_inputs/figure2_structures.csv"))
    assert structural_input.role.tolist() == ["halide", "N-H partner", "product"]
    assert structural_input.source_screen.eq("SNAr|1").all()
    assert structural_input.source_pair.eq(NUCLEOPHILE + " >> " + HALIDE).all()
    for role, smiles in (("halide", HALIDE), ("N-H partner", NUCLEOPHILE), ("product", product)):
        assert _canonical(structural_input.set_index("role").loc[role, "smiles"]) == _canonical(smiles)
    return len(tasks)


def _ledger(actual, expected):
    assert len(actual) == len(expected), "Printed-number count differs from independent specification"
    _close(actual.value.to_numpy(float), [row["value"] for row in expected], "Every independently derived printed number")
    for i, record in enumerate(expected):
        for key, value in record.items():
            if key == "value":
                continue
            observed = actual.iloc[i][key]
            if value == "":
                assert pd.isna(observed) or observed == "", f"Printed row {i}: {key}"
            elif isinstance(value, (int, float)):
                _close([observed], [value], f"Printed row {i}: {key}")
            else:
                assert str(observed) == str(value), f"Printed row {i}: {key}"
    return len(actual)


def _figure3(root, tables):
    values, scale = _anatomy(str(root))
    chosen, candidates = _owners(str(root))
    arrays = tables["Fig3A_arrays"]
    assert arrays.screen.tolist() == ORDER and arrays.plot_row.tolist() == list(range(1, 16))
    numeric = ["share1", "share2", "share", "combination", "noise", "owner_role", "owner_share", "partner_share", "owner_fraction", "pairs", "product_forming"]
    _frame(arrays, values, ["screen"], numeric, label="Figure 3 signed shares and variance composition")
    _frame(arrays, chosen, ["screen"], ["threshold_used", "product_forming_partners", "mean_pairwise_r"], ["smiles"], "Figure 3 owner illustrations")
    role_table = pd.read_csv(_result(root, "primary/history_roles.csv")).set_index("screen")
    expected_halides, bromides = [], []
    raw, _, _ = _raw(str(root))
    queries = [(name, Chem.MolFromSmarts(pattern)) for name, pattern in
               (("I", "cI"), ("Br", "cBr"), ("Cl", "cCl"), ("F", "cF"), ("sulfonate", "cOS(=O)(=O)"))]
    for row in arrays.itertuples():
        role_names = [str(role_table.loc[row.screen, f"role_{side}"]).replace("\u2013", "-") for side in (1, 2)]
        assert [row.role1, row.role2] == role_names
        halide_role = [side for side in (1, 2) if "halide" in role_names[side - 1]]
        expected_halides.append(bool(halide_role))
        assert bool(row.halide_array) == bool(halide_role)
        if halide_role:
            assert int(row.owner_role) == halide_role[0]
            molecules = [Chem.MolFromSmiles(s) for s in raw.loc[raw.screen.eq(row.screen), f"sub_{halide_role[0]}_smiles"].unique()]
            assert molecules and all(m is not None for m in molecules)
            groups = {next((name for name, pattern in queries if m.HasSubstructMatch(pattern)), "other") for m in molecules}
            is_bromide = groups == {"Br"}
        else:
            is_bromide = False
        assert bool(row.aryl_bromide_only) == is_bromide
        if is_bromide:
            bromides.append(row.screen)
    assert sum(expected_halides) == 10 and set(bromides) == {"Buchwald-Hartwig|2", "Heck|1", "Sonogashira|1", "Suzuki|1"}
    assert int((values.partner_share < 0).sum()) == 3
    selection_numeric = ["owner_role", "product_forming_partners", "mean_pairwise_r", "threshold_used", "eligible", "role_median_r", "absolute_distance_to_median", "selected"]
    _frame(tables["Rule2_selected"], chosen, ["screen", "smiles"], selection_numeric, ["owner_role_name"], "Independent Rule 2 selections")
    _frame(tables["Rule2_all_candidates"], candidates, ["screen", "smiles"], selection_numeric, ["owner_role_name"], "Independent Rule 2 candidates")
    observed_candidates = tables["Rule2_all_candidates"].set_index(["screen", "smiles"])
    for row in candidates.itertuples():
        observed = json.loads(observed_candidates.loc[(row.screen, row.smiles), "pairwise_correlations"])
        _nullable_close(observed, row.pairwise_correlations, "Rule 2 full pairwise correlations")
    selection = json.loads(_path(root, "04_SOURCE_DATA/Figure3/figure3_selection.json", "source_data/Figure3/figure3_selection.json").read_text(encoding="utf-8"))
    assert selection["rule_sha256"] == RULE_SHA
    _frame(pd.DataFrame(selection["selection"]), chosen, ["screen", "smiles"], selection_numeric, label="Rule 2 JSON selected rows")
    _frame(pd.DataFrame(selection["all_candidates"]), candidates, ["screen", "smiles"], selection_numeric, label="Rule 2 JSON candidate rows")
    conditions, pair, product = _snar(str(root))
    numeric_columns = [column for column in conditions if pd.api.types.is_numeric_dtype(conditions[column])]
    text_columns = [column for column in conditions if column not in numeric_columns and column != "condition_id"]
    _frame(tables["Fig3B_conditions"], conditions, ["condition_id"], numeric_columns, text_columns, "Figure 3 47 measured SNAr cells")
    _frame(tables["Fig3B_measurements"], pair, ["screen", "condition_id"], ["yield"], ["sub_1_smiles", "sub_2_smiles"], "Figure 3 raw SNAr yields")
    assert tables["Fig3B_measurements"].product_smiles.map(_canonical).eq(product).all(), "Figure 3 SNAr measured product connectivity"
    structure_count = _structures(root, chosen, product)
    assert arrays.structure_file.tolist() == [f"Fig3_A_owner{i:02d}.svg" for i in range(1, 16)]
    expected = []
    def add(panel, quantity, value, display):
        expected.append(dict(panel=panel, quantity=quantity, value=float(value), display=display))
    for row in values.itertuples():
        for field in ("share", "combination", "noise", "owner_share", "partner_share"):
            add("A", f"{row.screen}: {field}", 100 * getattr(row, field), "bar length")
    add("A", "halide arrays with halide owner", 10, "10")
    add("A", "median molecular share", 100 * values.share.median(), "33%")
    add("A", "median owner fraction", 100 * values.owner_fraction.median(), "80%")
    snar = values.set_index("screen").loc["SNAr|1"]
    for field, display in (("owner_share", "45.6%"), ("partner_share", "0.3%")):
        add("A", "SNAr " + field, 100 * snar[field], display)
    add("B", "measured combinations", 47, "47")
    for ticks, field in (((0, 100), "composition"), ((0, 20, 40), "role share")):
        for tick in ticks:
            add("A", field + " axis tick", tick, str(tick))
    ledger_rows = _ledger(tables["Printed_numbers"], expected)
    return dict(independent_arrays=15, signed_negative_partner_shares=3, independent_noise_scale=scale,
                rule2_candidates=len(candidates), rule2_selected=15, rule2_thresholds={row.screen: int(row.threshold_used) for row in chosen.itertuples()},
                snar_measured_cells=47, verified_cdxml_structures=structure_count, printed_numbers=ledger_rows)


def _score(frame, owner, partner):
    assert len(frame) and np.isfinite(frame[["sst", owner, partner]].to_numpy()).all()
    assert frame.sst.sum() > 0
    return float(1 - frame[owner].sum() / frame.sst.sum()), float(1 - frame[partner].sum() / frame.sst.sum())


@lru_cache(maxsize=2)
def _transfer(root_string):
    root = Path(root_string)
    _rule(root)
    frames = {key: pd.read_csv(_result(root, path)) for key, path in LOSS_FILES.items()}
    roles = pd.read_csv(_result(root, "display_inputs/Anatomy_arrays.csv")).set_index("screen").owner_role
    rows_a, rows_b = [], []
    for name, targets, arrays in (("Set A", 309, 6), ("Set B", 535, 13)):
        frame = frames[name]
        assert len(frame) == targets and frame.screen.nunique() == arrays and not frame.duplicated(["screen", "subpair"]).any()
        owner, partner = _score(frame, "sse_e", "sse_n")
        rows_a.append(dict(set=name, row_type="pooled", screen="all", owner=owner, partner=partner, targets=len(frame), arrays=arrays))
        for screen, group in frame.groupby("screen"):
            owner, partner = _score(group, "sse_e", "sse_n")
            rows_a.append(dict(set=name, row_type="array", screen=screen, owner=owner, partner=partner, targets=len(group), arrays=1))
        matched = frames["k2"][frames["k2"]["set"].eq(name)]
        _frame(matched, frame, ["screen", "subpair"], ["sst", "sse_e", "sse_n"], ["electrophile"], "Equal-depth cohort identity: " + name)
        owner, partner = _score(matched, "sse_e_k2", "sse_n_k2")
        rows_b.append(dict(set=name, cohort="Out-of-array: " + name, owner=owner, partner=partner, targets=len(matched), arrays=matched.screen.nunique()))
    internal = frames["within"]
    assert len(frames["k2"]) == 844 and len(internal) == 908
    assert not frames["k2"].duplicated(["set", "screen", "subpair"]).any() and not internal.duplicated(["screen", "subpair"]).any()
    owner, partner = _score(internal, "sse_owner", "sse_partner")
    rows_b.append(dict(set="Analysis arrays", cohort="Within-array held-out", owner=owner, partner=partner, targets=len(internal), arrays=internal.screen.nunique()))
    a, b = pd.DataFrame(rows_a), pd.DataFrame(rows_b)
    for frame in (a, b):
        frame["owner_display_pct"] = 100 * frame.owner
        frame["partner_display_pct"] = 100 * frame.partner
        frame["owner_minus_partner_display_points"] = 100 * (frame.owner - frame.partner)
    depth = frames["depth"].copy()
    assert len(depth) == 1247 and not depth.duplicated(["screen", "pair", "role"]).any()
    assert np.isfinite(depth[["sst", *[f"sse_{k}" for k in KS]]].to_numpy()).all()
    depth["side"] = ["owner" if int(row.role) == int(roles[row.screen]) else "partner" for row in depth.itertuples()]
    common_ids = depth.groupby(["screen", "pair"]).side.nunique()
    common_ids = common_ids[common_ids.eq(2)].index
    depth["common_target"] = pd.MultiIndex.from_frame(depth[["screen", "pair"]]).isin(common_ids)
    shared = depth[depth.common_target]
    assert len(common_ids) == 310 and len(shared) == 620
    assert shared.groupby(["screen", "pair"]).sst.nunique().eq(1).all()
    counts = shared.groupby("screen").pair.nunique().sort_index()
    assert counts.to_dict() == {"Acylation|1": 236, "Buchwald-Hartwig|1": 4, "Kumada|1": 3, "Sonogashira|1": 1, "Suzuki|1": 66}
    selected = counts[counts.ge(20)].index.tolist()
    assert selected == ["Acylation|1", "Suzuki|1"]
    pooled, thin = [], []
    for side, group in shared.groupby("side"):
        for k in KS:
            pooled.append(dict(side=side, partners=k, explained_pct=100 * (1 - group[f"sse_{k}"].sum() / group.sst.sum()),
                               targets=len(group), arrays=group.screen.nunique(), sum_sse=float(group[f"sse_{k}"].sum()), sum_sst=float(group.sst.sum())))
    for (screen, side), group in shared.groupby(["screen", "side"]):
        if screen in selected:
            for k in KS:
                thin.append(dict(screen=screen, side=side, partners=k, common_targets=group.pair.nunique(),
                                 explained_pct=100 * (1 - group[f"sse_{k}"].sum() / group.sst.sum()),
                                 sum_sse=float(group[f"sse_{k}"].sum()), sum_sst=float(group.sst.sum())))
    folds = pd.read_csv(_path(root, "05_DATA/Analysis/figure_display/Fig4a_folds.csv", "metadata/figure_display/Fig4a_folds.csv"))
    assert not folds.duplicated(["screen", "subpair"]).any()
    pair_lists, molecule_lists = {}, {}
    for screen, group in folds.groupby("screen"):
        ordered = group.sort_values("subpair", kind="stable")
        pair_lists[screen] = list(zip(ordered.subpair, ordered.fold))
        molecule_lists[screen] = {role: sorted({pair.split(" >> ")[role - 1] for pair in ordered.subpair}) for role in (1, 2)}
    records = []
    for row in depth.itertuples():
        pair, fold = pair_lists[row.screen][row.pair]
        molecule = pair.split(" >> ")[row.role - 1]
        assert molecule_lists[row.screen][row.role][row.molecule] == molecule
        records.append(dict(screen=row.screen, role=row.role, side=row.side, target_pair=pair, molecule=molecule, fold=int(fold),
                            sst=row.sst, **{f"sse_{k}": getattr(row, f"sse_{k}") for k in KS},
                            pair_index=row.pair, molecule_index=row.molecule, common_target=row.common_target))
    return dict(frames=frames, a=a, b=b, pooled=pd.DataFrame(pooled), thin=pd.DataFrame(thin), counts=counts,
                selected=selected, depth=depth, records=pd.DataFrame(records))


def _figure5(root, tables):
    data = _transfer(str(root))
    frames, a, b = data["frames"], data["a"], data["b"]
    numeric = ["owner", "partner", "targets", "arrays", "owner_display_pct", "partner_display_pct", "owner_minus_partner_display_points"]
    _frame(tables["A_array_and_pooled"], a, ["set", "row_type", "screen"], numeric, label="Figure 5 per-array and pooled SSE/SST")
    assert tables["B_equal_depth"]["set"].tolist() == ["Set A", "Set B", "Analysis arrays"], "Figure 5 separate equal-depth cohort order/count"
    _frame(tables["B_equal_depth"], b, ["set"], numeric, ["cohort"], "Figure 5 separate equal-depth cohorts")
    curve_values = ["explained_pct", "sum_sse", "sum_sst", "targets", "arrays"]
    _frame(tables["C_pooled_curves"], data["pooled"], ["side", "partners"], curve_values, label="Figure 5 all-five-array common-target curves")
    _frame(tables["C_selected_array_curves"], data["thin"], ["screen", "side", "partners"], ["explained_pct", "common_targets", "sum_sse", "sum_sst"], label="Figure 5 frozen Rule 3 thin curves")
    counts = pd.DataFrame([dict(screen=key, common_targets=int(value), thin_curve=value >= 20) for key, value in data["counts"].items()])
    _frame(tables["C_selection_counts"], counts, ["screen"], ["common_targets", "thin_curve"], label="Rule 3 counts")
    selection = json.loads(_path(root, "04_SOURCE_DATA/Figure5/figure5_selection.json", "source_data/Figure5/figure5_selection.json").read_text(encoding="utf-8"))
    assert selection["rule_sha256"] == RULE_SHA and selection["threshold_common_targets"] == 20 and selection["common_targets"] == 310
    assert selection["array_counts"] == {key: int(value) for key, value in data["counts"].items()}
    assert selection["pooled_arrays"] == data["counts"].index.tolist()
    assert selection["omitted_thin_arrays"] == data["counts"][data["counts"].lt(20)].index.tolist()
    assert [(row["screen"], row["targets"]) for row in selection["selected_arrays"]] == [(key, int(data["counts"][key])) for key in data["selected"]]
    for name, sheet in (("Set A", "A_SetA_target_errors"), ("Set B", "A_SetB_target_errors")):
        _frame(tables[sheet], frames[name], ["screen", "subpair"], ["sse_e", "sse_n", "sst"], ["electrophile"], sheet)
        assert tables[sheet].source_file.eq("05_DATA/Results/" + LOSS_FILES[name]).all()
    renamed = frames["k2"].rename(columns={"electrophile": "owner_molecule", "sse_e": "sse_owner", "sse_n": "sse_partner",
        "sse_e_k2": "sse_owner_two_partners", "sse_n_k2": "sse_partner_two_partners", "subsets_e": "owner_subsets", "subsets_n": "partner_subsets",
        "history_e": "owner_history_pairs", "history_n": "partner_history_pairs"})
    target_numeric = ["sse_owner", "sse_partner", "sst", "sse_owner_two_partners", "owner_subsets", "owner_history_pairs",
                      "sse_partner_two_partners", "partner_subsets", "partner_history_pairs"]
    _frame(tables["AB_equal_depth_targets"], renamed, ["set", "screen", "subpair"], target_numeric, ["owner_molecule"], "All 844 equal-depth targets")
    for sheet in ("B_within_array_targets", "Fig4b_targets"):
        _frame(tables[sheet], frames["within"], ["screen", "subpair"], ["sst", "sse_owner", "sse_partner", "subsets_owner", "subsets_partner"], label=sheet)
    record_numeric = ["fold", "sst", *[f"sse_{k}" for k in KS], "pair_index", "molecule_index", "common_target"]
    for sheet in ("C_retained_depth_records", "Fig4a_records"):
        _frame(tables[sheet], data["records"], ["screen", "target_pair", "role"], record_numeric, ["side", "molecule"], sheet)
    compatibility = pd.DataFrame([dict(side=side, k=k, records=len(group), additive_profile_explained_pct=100 * (1 - group[f"sse_{k}"].sum() / group.sst.sum()))
                                  for side, group in data["records"].groupby("side") for k in KS])
    _frame(tables["Fig4a"], compatibility, ["side", "k"], ["records", "additive_profile_explained_pct"], label="Full-cohort compatibility curves")
    within = b[b["set"].eq("Analysis arrays")].iloc[0]
    legacy = tables["Fig4b"]
    assert len(legacy) == 1 and legacy.row_type.iloc[0] == "pooled"
    _close(legacy[["targets", "arrays", "owner_fraction", "partner_fraction", "D"]],
           [[908, within.arrays, within.owner, within.partner, within.owner - within.partner]], "Within-array compatibility summary")
    group = frames["k2"][frames["k2"]["set"].eq("Set A")]
    deeper = {"partner": int((group.history_n > group.history_e).sum()), "owner": int((group.history_e > group.history_n).sum())}
    assert deeper == {"partner": 167, "owner": 121}
    expected = []
    def add(panel, value, display, source, field, selection_text, **extra):
        expected.append(dict(panel=panel, value=float(value), display=display, source_file=source, source_field=field,
                             source_selection=selection_text, **extra))
    for row in a.itertuples():
        source = "05_DATA/Results/" + LOSS_FILES[row.set]
        for side, field in (("owner", "sse_e"), ("partner", "sse_n")):
            value = 100 * getattr(row, side)
            add("A", value, f"{value:.1f}" if row.row_type == "pooled" else "", source, "sst;" + field,
                "all rows" if row.row_type == "pooled" else "screen=" + row.screen, cohort=row.set, side=side, row_type=row.row_type)
        if row.row_type == "pooled":
            add("A", row.arrays, str(row.arrays), source, "screen", "all rows", cohort=row.set)
    for row in b.itertuples():
        source = "05_DATA/Results/" + LOSS_FILES["k2" if row.set != "Analysis arrays" else "within"]
        for side, field in (("owner", "sse_e_k2" if row.set != "Analysis arrays" else "sse_owner"),
                            ("partner", "sse_n_k2" if row.set != "Analysis arrays" else "sse_partner")):
            value = 100 * getattr(row, side)
            add("B", value, f"{value:.1f}", source, "sst;" + field, "set=" + row.set if row.set != "Analysis arrays" else "all rows", cohort=row.cohort, side=side)
    for side in ("partner", "owner"):
        add("B", deeper[side], str(deeper[side]), "05_DATA/Results/" + LOSS_FILES["k2"], "history_e;history_n", "set=Set A", side=side)
    add("B", 2, "2", "05_DATA/Results/" + LOSS_FILES["k2"] + ";05_DATA/Results/" + LOSS_FILES["within"], "sse_e_k2;sse_n_k2;subsets_owner;subsets_partner", "endpoint definition")
    for row in data["pooled"].itertuples():
        add("C", row.explained_pct, f"{row.explained_pct:.1f}%" if row.partners in (1, 6) else "", "05_DATA/Results/" + LOSS_FILES["depth"],
            f"sst;sse_{row.partners}", "common_target=True;side=" + row.side, side=row.side, history_partners=row.partners)
    for row in data["thin"].itertuples():
        add("C", row.explained_pct, "", "05_DATA/Results/" + LOSS_FILES["depth"], f"sst;sse_{row.partners}",
            f"common_target=True;screen={row.screen};side={row.side}", side=row.side, history_partners=row.partners)
    add("C", 310, "310", "05_DATA/Results/" + LOSS_FILES["depth"], "screen;pair;role", "both roles observed")
    for screen in data["selected"]:
        count = int(data["counts"][screen])
        add("C", count, str(count), "05_DATA/Results/" + LOSS_FILES["depth"], "screen;pair;role", "common_target=True;screen=" + screen)
    for k in KS:
        add("C", k, str(k), "05_DATA/Results/" + LOSS_FILES["depth"], "sse_" + str(k), "retained history depth")
    for panel, ticks in (("A", [-20, 0, 10, 20, 30]), ("B", [0, 8, 16]), ("C", [-5, 0, 10, 20, 30])):
        for value in ticks:
            add(panel, value, f"{value:g}", "06_CODE/Figures/figure5.py", "draw: axis ticks", "display coordinate")
    ledger_rows = _ledger(tables["Printed_numbers"], expected)
    return dict(set_A_targets=309, set_A_arrays=6, set_B_targets=535, set_B_arrays=13, equal_depth_targets=844,
                within_array_targets=908, retained_depth_records=1247, common_targets=310, pooled_arrays=5,
                selected_array_counts={key: int(data["counts"][key]) for key in data["selected"]},
                curve_depths=list(KS), printed_numbers=ledger_rows)


def audit_tables(number, root: Path, tables: dict) -> dict:
    root = Path(root).resolve()
    assert number in (3, 5), "This module covers Figures 3 and 5"
    return (_figure3 if number == 3 else _figure5)(root, tables)


def _gid_items(readback, key):
    items = [item for axis in readback["axes"] for item in axis.get(key, []) if item.get("gid")]
    identifiers = [item["gid"] for item in items]
    assert len(identifiers) == len(set(identifiers)), "Duplicate quantitative artist identifiers"
    return {item["gid"]: item for item in items}


def _figure3_artists(root, tables, readback):
    values, _ = _anatomy(str(root))
    patches = _gid_items(readback, "patches")
    composition_axes = [axis for axis in readback["axes"] if axis["xlim"] == [0.0, 100.0]]
    share_axes = [axis for axis in readback["axes"] if axis["xlim"] == [-8.0, 50.0]]
    assert len(composition_axes) == len(share_axes) == 1
    assert len(composition_axes[0]["patches"]) == 45 and len(share_axes[0]["patches"]) == 30
    sizes = tables["Structure_sizes"]
    assert sizes.screen.tolist() == ORDER and len(sizes) == 15
    for i, row in enumerate(values.itertuples(), 1):
        size = sizes.iloc[i - 1]
        center = float(size.center_y_mm)
        assert size.row_top_mm < center < size.row_bottom_mm
        _close([center], [(size.row_top_mm + size.row_bottom_mm) / 2], "Figure 3 structure-centered row")
        left = 0.0
        for field in ("share", "combination", "noise"):
            patch = patches[f"A_{field}_{i}"]
            value = 100 * getattr(row, field)
            _close([patch["width"], patch["height"], *patch["xy"]], [value, 2.2, left, center - 1.1], "Figure 3 composition artist")
            left += value
        for side, offset in (("owner", -1.4), ("partner", 1.4)):
            patch = patches[f"A_{side}_share_{i}"]
            _close([patch["width"], patch["height"], *patch["xy"]], [100 * getattr(row, side + "_share"), 2.0, 0, center + offset - 1.0], "Figure 3 signed share artist")
    expected_gids = {f"A_{field}_{i}" for i in range(1, 16) for field in ("share", "combination", "noise", "owner_share", "partner_share")}
    quantitative_gids = {key for key in patches if any(key.startswith("A_" + field + "_") for field in ("share", "combination", "noise", "owner_share", "partner_share"))}
    assert quantitative_gids == expected_gids, "Missing or extra Figure 3 quantitative bars"
    dots = _gid_items(readback, "collections")
    assert set(dots) == {"B_47_measured_combinations"}
    conditions, _, _ = _snar(str(root))
    actual = dots["B_47_measured_combinations"]["offsets"]
    assert len(actual) == len(set(map(tuple, actual))) == 47
    assert set(map(tuple, actual)) == set(map(tuple, conditions[["matrix_row", "matrix_column"]].to_numpy()))
    return dict(composition_bars=45, signed_role_bars=30, measured_condition_points=47)


def _figure5_artists(root, readback):
    data = _transfer(str(root))
    lines = _gid_items(readback, "lines")
    scatter_axes = [axis for axis in readback["axes"] if axis["xlim"] == [-20.0, 28.0] and axis["ylim"] == [-8.0, 31.0]]
    assert len(scatter_axes) == 1
    assert len([line for line in scatter_axes[0]["lines"] if len(line["x"]) == len(line["y"]) == 1]) == 21
    expected_gids = set()
    for row in data["a"].itertuples():
        gid = "A|pooled|" + row.set if row.row_type == "pooled" else "A|" + row.set + "|" + row.screen
        expected_gids.add(gid)
        _close(lines[gid]["x"], [row.partner_display_pct], "Figure 5 array/pooled x")
        _close(lines[gid]["y"], [row.owner_display_pct], "Figure 5 array/pooled y")
    equal_axes = [axis for axis in readback["axes"] if axis["xlim"] == [0.0, 16.0] and axis["ylim"] == [3.5, 0.5]]
    assert len(equal_axes) == 1, "Missing or duplicated equal-depth axes"
    marks = equal_axes[0]["lines"]
    assert len(marks) == 9
    for i, row in enumerate(data["b"].itertuples(), 1):
        connector, owner, partner = marks[3 * (i - 1):3 * i]
        _close(connector["x"], [row.partner_display_pct, row.owner_display_pct], "Figure 5 equal-depth connector")
        _close(connector["y"], [i, i], "Figure 5 equal-depth connector row")
        for mark, value in ((owner, row.owner_display_pct), (partner, row.partner_display_pct)):
            _close(mark["x"], [value], "Figure 5 equal-depth endpoint")
            _close(mark["y"], [i], "Figure 5 equal-depth endpoint row")
    for side, group in data["pooled"].groupby("side"):
        gid = "C|pooled|" + side
        expected_gids.add(gid)
        _close(lines[gid]["x"], list(KS), "Figure 5 pooled curve depths", 0)
        _close(lines[gid]["y"], group.explained_pct, "Figure 5 all-five-array pooled curve")
    for (screen, side), group in data["thin"].groupby(["screen", "side"]):
        gid = "C|thin|" + screen + "|" + side
        expected_gids.add(gid)
        _close(lines[gid]["x"], list(KS), "Figure 5 thin curve depths", 0)
        _close(lines[gid]["y"], group.explained_pct, "Figure 5 frozen Rule 3 thin curve")
    curve_axes = [axis for axis in readback["axes"] if axis["xlim"] == [0.85, 6.45] and axis["ylim"] == [-6.0, 31.0]]
    assert len(curve_axes) == 1
    assert len([line for line in curve_axes[0]["lines"] if len(line["x"]) == len(KS)]) == 6
    assert set(lines) == expected_gids, "Missing or extra Figure 5 quantitative artist"
    return dict(array_points=19, pooled_points=2, separate_equal_depth_connectors=3, equal_depth_endpoints=6,
                thin_curves=4, pooled_curves=2, curve_points=30)


def audit_artists(number, root, tables, readback) -> dict:
    root = Path(root).resolve()
    assert number in (3, 5)
    assert int(readback["figure"]) == number
    return _figure3_artists(root, tables, readback) if number == 3 else _figure5_artists(root, readback)


def fault_tests(number, root, tables, readback=None) -> list[dict]:
    """Require a passing baseline, then reject each isolated in-memory fault."""
    audit_tables(number, root, tables)
    if readback is not None:
        audit_artists(number, root, tables, readback)
    faults = []
    def reject(name, target, expected_issue, mutate, artists=False):
        bad_tables = {key: frame.copy(deep=True) for key, frame in tables.items()}
        bad_readback = deepcopy(readback)
        mutate(bad_tables, bad_readback)
        try:
            if artists:
                audit_artists(number, root, bad_tables, bad_readback)
            else:
                audit_tables(number, root, bad_tables)
        except AssertionError as error:
            assert expected_issue in str(error), f"Fault {name} produced an unrelated issue: {error}"
            faults.append(dict(fault=name, target=target, rejected=True, baseline_passed=True, new_issue=str(error)))
            return
        raise AssertionError("Planted fault accepted: " + name)
    if number == 3:
        def signed_share(bad, artists):
            owner_role = int(bad["Fig3A_arrays"].loc[0, "owner_role"])
            bad["Fig3A_arrays"].loc[0, f"share{owner_role}"] += 0.01
            bad["Fig3A_arrays"].loc[0, ["owner_share", "share"]] += 0.01
            bad["Fig3A_arrays"].loc[0, "combination"] -= 0.01
            bad["Fig3A_arrays"].loc[0, "owner_fraction"] = bad["Fig3A_arrays"].loc[0, "owner_share"] / bad["Fig3A_arrays"].loc[0, "share"]
            label = bad["Fig3A_arrays"].loc[0, "screen"]
            for field, delta in (("owner_share", 1), ("share", 1), ("combination", -1)):
                mask = bad["Printed_numbers"].quantity.eq(label + ": " + field)
                bad["Printed_numbers"].loc[mask, "value"] += delta
            if artists is not None:
                for field, delta in (("owner_share", 1), ("share", 1), ("combination", -1)):
                    next(item for axis in artists["axes"] for item in axis["patches"] if item.get("gid") == f"A_{field}_1")["width"] += delta
                next(item for axis in artists["axes"] for item in axis["patches"] if item.get("gid") == "A_combination_1")["xy"][0] += 1
        reject("coherent owner-share source/ledger/artist mutation", "Figure3_SourceData.xlsx:Fig3A_arrays:first owner share; Printed_numbers; A_owner_share_1", "Figure 3 signed shares and variance composition", signed_share)
        reject("Rule 2 selected owner changed", "Figure3_SourceData.xlsx:Rule2_selected:smiles", "Independent Rule 2 selections: identities differ", lambda bad, _: bad["Rule2_selected"].__setitem__("smiles", "C"))
        reject("forbidden Rule 2 threshold fallback", "Figure3_SourceData.xlsx:Rule2_selected:threshold_used", "Independent Rule 2 selections: max difference", lambda bad, _: bad["Rule2_selected"].__setitem__("threshold_used", 2))
        reject("SNAr measured condition deleted", "Figure3_SourceData.xlsx:Fig3B_conditions:last measured cell", "Figure 3 47 measured SNAr cells: row count differs", lambda bad, _: bad.__setitem__("Fig3B_conditions", bad["Fig3B_conditions"].iloc[:-1].copy()))
        reject("SNAr product connectivity changed", "Figure3_SourceData.xlsx:Fig3B_measurements:product_smiles", "Figure 3 SNAr measured product connectivity", lambda bad, _: bad["Fig3B_measurements"].__setitem__("product_smiles", "CC"))
        if readback is not None:
            def bar_fault(_, bad):
                next(item for axis in bad["axes"] for item in axis["patches"] if item.get("gid") == "A_owner_share_1")["width"] += 1
            reject("signed bar readback changed", "Figure3_readback.json:A_owner_share_1:width", "Figure 3 signed share artist: max difference", bar_fault, True)
    else:
        def pooled(bad, artists):
            mask = bad["A_array_and_pooled"].row_type.eq("pooled") & bad["A_array_and_pooled"]["set"].eq("Set A")
            bad["A_array_and_pooled"].loc[mask, "owner"] += 0.01
            bad["A_array_and_pooled"].loc[mask, "owner_display_pct"] += 1
            bad["A_array_and_pooled"].loc[mask, "owner_minus_partner_display_points"] += 1
            ledger = bad["Printed_numbers"]
            printed = ledger.panel.eq("A") & ledger.row_type.eq("pooled") & ledger.cohort.eq("Set A") & ledger.side.eq("owner")
            ledger.loc[printed, "value"] += 1
            ledger.loc[printed, "display"] = ledger.loc[printed, "value"].map(lambda value: f"{value:.1f}")
            if artists is not None:
                next(item for axis in artists["axes"] for item in axis["lines"] if item.get("gid") == "A|pooled|Set A")["y"][0] += 1
        reject("coherent pooled SSE/SST source/ledger/artist mutation", "Figure5_SourceData.xlsx:A_array_and_pooled:Set A pooled owner; Printed_numbers; A|pooled|Set A", "Figure 5 per-array and pooled SSE/SST: max difference", pooled)
        reject("equal-depth cohorts combined", "Figure5_SourceData.xlsx:B_equal_depth:within-array cohort", "Figure 5 separate equal-depth cohort order/count", lambda bad, _: bad.__setitem__("B_equal_depth", bad["B_equal_depth"].iloc[:2].copy()))
        reject("small array excluded from common pooled curves", "Figure5_SourceData.xlsx:C_pooled_curves:arrays", "Figure 5 all-five-array common-target curves: max difference", lambda bad, _: bad["C_pooled_curves"].__setitem__("arrays", 2))
        reject("Rule 3 low-count array selected", "Figure5_SourceData.xlsx:C_selection_counts:thin_curve", "Rule 3 counts: max difference", lambda bad, _: bad["C_selection_counts"].__setitem__("thin_curve", True))
        reject("retained loss record changed", "Figure5_SourceData.xlsx:A_SetA_target_errors:sse_e", "A_SetA_target_errors: max difference", lambda bad, _: bad["A_SetA_target_errors"].__setitem__("sse_e", bad["A_SetA_target_errors"].sse_e + 0.01))
        if readback is not None:
            def curve_fault(_, bad):
                next(item for axis in bad["axes"] for item in axis["lines"] if item.get("gid") == "C|pooled|owner")["y"][0] += 1
            reject("pooled curve artist changed", "Figure5_readback.json:C|pooled|owner:y[0]", "Figure 5 all-five-array pooled curve: max difference", curve_fault, True)
    return faults
