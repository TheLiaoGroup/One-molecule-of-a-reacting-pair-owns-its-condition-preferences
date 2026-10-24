"""Independent numerical and artist checks for formal Figure 4.

Observed workbook values and artist coordinates are compared with the existing
independent yield calculation. No figure generator or figure helper is imported.
"""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from audit_main_v2_values23 import _close, _directional, _result


TOL = 1e-9
DISPLAY_ORDER = ("M4", "M8", "M1", "M2", "M3", "M5", "M7", "M6")
CONTEXT_COLUMNS = ("T_amide", "T_BH1", "T_BH2")
ARRAYS = ("Acylation|1", "Buchwald-Hartwig|1", "Buchwald-Hartwig|2")
KEY = ["molecule", "array", "role"]


def _reference(root):
    a, b, contexts, cohorts = _directional(str(Path(root).resolve()))
    b = b.sort_values(["T_amide", "plot_label"], kind="stable").reset_index(drop=True)
    assert tuple(b.plot_label) == DISPLAY_ORDER, "Figure 4A authorized display order"
    assert len(a) == 23 and a.stratum.value_counts().to_dict() == {"amine": 19, "acid": 4}
    assert len(contexts) == 55
    return a, b, contexts, cohorts


def _empty(value):
    return "" if pd.isna(value) else str(value)


def _printed_numbers(tables, a, b):
    """The printed-number table is an observation, never the reference."""
    expected = defaultdict(list)

    def add(panel, field, selection, value, display=None, item=""):
        expected[(panel, field, selection, item)].append((float(value), display))

    for row in b.itertuples():
        for column, array in zip(CONTEXT_COLUMNS, ARRAYS):
            add("A", "T", f"molecule={row.smiles};array={array};role=1",
                getattr(row, column), item=row.plot_label)
        for value in (-0.4, 0, 0.4):
            add("A", "draw: axis ticks", "display coordinate", value, value, row.plot_label)
    for column, field in (("T_BH1", "mean_amide_minus_BH1"), ("T_BH2", "mean_amide_minus_BH2")):
        value = float((b.T_amide - b[column]).mean())
        add("A", "C." + field, "existing summary", value, float(f"{value:.2f}"))
    names = tables["B_23_molecules"].set_index("molecule")["name"]
    for row in a.itertuples():
        for field in ("T_O", "T_N"):
            add("B", field, "molecule=" + row.molecule, getattr(row, field), item=names.loc[row.molecule])
    for field in ("T_O", "T_N"):
        value = float(a[field].mean())
        add("B", "descriptives." + field + "_mean", "existing summary", value, float(f"{value:.2f}"))
    for field, value in (("positive", int((a.D > 0).sum())), ("n", len(a))):
        add("B", "A." + field, "existing summary", value, value)
    for value in (-0.4, 0, 0.4, 0.8):
        add("B", "draw: axis ticks", "display coordinate", value, value)

    observed = defaultdict(list)
    for row in tables["Printed_numbers"].itertuples():
        key = (row.panel, row.source_field, row.source_selection, _empty(row.item))
        display = None if pd.isna(row.display) or row.display == "" else float(row.display)
        observed[key].append((float(row.value), display))
    assert set(observed) == set(expected), "Figure 4 printed-number identities missing or unexpected"
    for key, reference in expected.items():
        values = sorted(observed[key], key=lambda pair: pair[0])
        reference = sorted(reference, key=lambda pair: pair[0])
        assert len(values) == len(reference), "Figure 4 printed-number multiplicity: " + repr(key)
        _close([value[0] for value in values], [value[0] for value in reference],
               "Figure 4 independent printed values " + repr(key), TOL)
        for actual, wanted in zip(values, reference):
            assert (actual[1] is None) == (wanted[1] is None), "Figure 4 printed display missing or unexpected"
            if wanted[1] is not None:
                _close([actual[1]], [wanted[1]], "Figure 4 independent rounded display " + repr(key), TOL)
    count = sum(map(len, expected.values()))
    assert count == len(tables["Printed_numbers"]) == 104
    return count


def audit_tables(number, root: Path, tables: dict[str, pd.DataFrame]) -> dict:
    assert number == 4, "Figure 4 module received another figure"
    root = Path(root).resolve()
    a, b, contexts, cohorts = _reference(root)
    observed_b = tables["A_eight_amines"]
    assert tuple(observed_b.plot_label) == DISPLAY_ORDER
    assert observed_b.smiles.tolist() == b.smiles.tolist()
    deviation = _close(observed_b[list(CONTEXT_COLUMNS)], b[list(CONTEXT_COLUMNS)],
                       "Figure 4A independent triple-context T", TOL)
    cn_mean = b[["T_BH1", "T_BH2"]].mean(axis=1)
    differences = b.T_amide - cn_mean
    _close(observed_b.CN_mean, cn_mean, "Figure 4A C-N mean", TOL)
    _close(observed_b.R, differences, "Figure 4A amide contrast", TOL)
    assert set(b.loc[differences < 0, "plot_label"]) == {"M3", "M8"}
    partner_counts = cohorts.set_index(KEY).n_pf_partners
    for row in observed_b.itertuples():
        molecule = Chem.MolFromSmiles(row.smiles)
        assert molecule is not None and row.heavy_atoms == molecule.GetNumHeavyAtoms()
        for field, array in zip(("partners_amide", "partners_BH1", "partners_BH2"), ARRAYS):
            assert getattr(row, field) == partner_counts.loc[(row.smiles, array, 1)]
        for field, wanted in (("amide_above_CN_mean", row.R > 0),
                              ("amide_above_both", row.T_amide > max(row.T_BH1, row.T_BH2))):
            actual = getattr(row, field)
            assert pd.notna(actual) and actual in (False, True) and bool(actual) == bool(wanted)
        assert row.line_colour == ("exception" if row.plot_label in ("M3", "M8") else "standard")

    observed_a = tables["B_23_molecules"]
    assert observed_a.molecule.tolist() == a.molecule.tolist()
    assert observed_a.stratum.tolist() == a.stratum.tolist()
    assert observed_a.plot_row.tolist() == list(range(1, 24))
    deviation = max(deviation, _close(observed_a[["T_O", "T_N", "D"]], a[["T_O", "T_N", "D"]],
                                      "Figure 4B independent molecule values", TOL))
    result_d = pd.read_csv(_result(root, "directional/phase1_D.csv")).set_index("molecule")
    assert result_d.index.is_unique and set(result_d.index) == set(a.molecule)
    result_d = result_d.loc[a.molecule]
    _close(result_d[["T_O", "T_N", "D"]], a[["T_O", "T_N", "D"]],
           "Figure 4 independent D versus analysis", TOL)
    for field in ("owner_arrays", "nonowner_arrays"):
        assert observed_a[field].tolist() == result_d[field].tolist()
    _close(observed_a.dTanimoto, result_d.dTanimoto, "Figure 4 retained directional descriptor", TOL)

    source_contexts = tables["B_retained_contexts"].set_index(KEY)
    reference_contexts = contexts.set_index(KEY)
    assert source_contexts.index.is_unique and len(source_contexts) == 55
    assert set(source_contexts.index) == set(reference_contexts.index)
    source_contexts = source_contexts.loc[reference_contexts.index]
    _close(source_contexts["T"], reference_contexts["T"], "Figure 4 all 55 context values", TOL)
    _close(source_contexts.n_ordered_pair_pairs, reference_contexts.ordered_pairs_of_pairs,
           "Figure 4 context comparison counts", TOL)
    assert source_contexts.context.tolist() == reference_contexts.context.tolist()
    design = cohorts[(cohorts.set == "G1") & cohorts.array_clear_owner].set_index(KEY).loc[reference_contexts.index]
    _close(source_contexts.n_pf_partners, design.n_pf_partners, "Figure 4 context partner counts", TOL)

    retained = tables["Retained_T"].set_index(KEY)
    result_t = pd.read_csv(_result(root, "directional/phase1_T.csv")).set_index(KEY)
    assert retained.index.is_unique and result_t.index.is_unique
    assert len(retained) == len(result_t) and set(retained.index) == set(result_t.index)
    _close(retained.loc[result_t.index, ["T", "n_ordered_pair_pairs"]],
           result_t[["T", "n_ordered_pair_pairs"]], "Figure 4 retained T table", TOL)

    amide_gains = [float((b.T_amide - b[field]).mean()) for field in ("T_BH1", "T_BH2")]
    assert [f"{value:.2f}" for value in amide_gains] == ["0.19", "0.22"]
    assert [f"{a[field].mean():.2f}" for field in ("T_O", "T_N")] == ["0.26", "0.06"]
    assert int((a.D > 0).sum()) == 17 and len(a) == 23
    printed = _printed_numbers(tables, a, b)
    return dict(independent_slope_values=24, independent_molecules=23, independent_per_array_contexts=55,
                numerical_ledger_rows=printed, display_order=list(DISPLAY_ORDER),
                analysis_max_difference=deviation)


def audit_artists(number, root, tables, readback: dict) -> dict:
    assert number == 4 and readback["figure"] == 4
    a, b, _, _ = _reference(root)
    slope_axes = [axis for axis in readback["axes"]
                  if len(axis["xlim"]) == 2 and np.max(np.abs(np.asarray(axis["xlim"]) - [-0.1, 2.1])) <= TOL]
    assert len(slope_axes) == 8, "Figure 4A must have eight small-multiple axes"
    for axis, row in zip(slope_axes, b.itertuples()):
        _close(axis["ylim"], [-0.48, 0.70], "Figure 4A common vertical scale", TOL)
        assert len(axis["lines"]) == 2
        zero, path = axis["lines"]
        _close(zero["x"], [0, 1], "Figure 4A zero-line x", TOL)
        _close(zero["y"], [0, 0], "Figure 4A zero-line y", TOL)
        assert path["gid"] == "A|path|" + row.plot_label, "Figure 4A path order or identity"
        _close(path["x"], [0, 1, 2], "Figure 4A three context positions", TOL)
        _close(path["y"], [getattr(row, field) for field in CONTEXT_COLUMNS],
               "Figure 4A independent path and marker values " + row.plot_label, TOL)

    dumbbells = [axis for axis in readback["axes"]
                 if len(axis["xlim"]) == 2 and np.max(np.abs(np.asarray(axis["xlim"]) - [-0.52, 0.88])) <= TOL]
    assert len(dumbbells) == 1, "Figure 4B must have one molecule-comparison axis"
    dumbbell = dumbbells[0]
    _close(dumbbell["ylim"], [192.5, 32.5], "Figure 4B row range", TOL)
    assert len(dumbbell["lines"]) == 70
    _close(dumbbell["lines"][0]["x"], [0, 0], "Figure 4B zero-line x", TOL)
    _close(dumbbell["lines"][0]["y"], [0, 1], "Figure 4B zero-line y", TOL)
    for i, row in enumerate(a.itertuples()):
        connector, nonowner, owner = dumbbell["lines"][1 + 3 * i:4 + 3 * i]
        y = 37 + 6.8 * i
        _close(connector["x"], [row.T_N, row.T_O], "Figure 4B independent connector values", TOL)
        _close(connector["y"], [y, y], "Figure 4B connector row", TOL)
        for mark, value in ((nonowner, row.T_N), (owner, row.T_O)):
            _close(mark["x"], [value], "Figure 4B independent point value", TOL)
            _close(mark["y"], [y], "Figure 4B point row", TOL)

    text = [item["text"] for item in readback["texts"]]
    captions = [f"Amide \u2212 ligand library +{(b.T_amide - b.T_BH1).mean():.2f};",
                f"amide \u2212 metal library +{(b.T_amide - b.T_BH2).mean():.2f}",
                f"Means: owner {a.T_O.mean():.2f}; non-owner {a.T_N.mean():.2f}",
                f"Owner context higher for {int((a.D > 0).sum())}/{len(a)} molecules"]
    assert all(text.count(caption) == 1 for caption in captions), "Figure 4 independent summary labels"
    return dict(molecule_trajectories=8, trajectory_points=24, trajectory_segments=16,
                context_points=46, context_connectors=23, independent_summary_labels=4,
                line_style_capture="coordinates only; marker styles are checked separately")


def fault_tests(number, root, tables, readback=None) -> list[dict]:
    """Run only after a passing baseline; a coherent table/ledger change must fail."""
    audit_tables(number, root, tables)
    if readback is not None:
        audit_artists(number, root, tables, readback)
    bad = {name: frame.copy(deep=True) for name, frame in tables.items()}
    first = bad["A_eight_amines"].index[0]
    row = bad["A_eight_amines"].loc[first]
    bad["A_eight_amines"].loc[first, ["T_amide", "R"]] += 0.01
    ledger = bad["Printed_numbers"]
    affected = ((ledger.panel == "A") & (ledger.item == row.plot_label)
                & (ledger.source_selection == f"molecule={row.smiles};array=Acylation|1;role=1"))
    assert int(affected.sum()) == 1
    ledger.loc[affected, "value"] += 0.01
    for field, column in (("C.mean_amide_minus_BH1", "T_BH1"), ("C.mean_amide_minus_BH2", "T_BH2")):
        selected = (ledger.panel == "A") & (ledger.source_field == field)
        assert int(selected.sum()) == 1
        value = float((bad["A_eight_amines"].T_amide - bad["A_eight_amines"][column]).mean())
        ledger.loc[selected, "value"] = value
        ledger.loc[selected, "display"] = f"+{value:.2f}"
    try:
        audit_tables(number, root, bad)
    except AssertionError as error:
        results = [dict(fault="coherent_Figure4A_T_and_ledger_change", rejected=True, new_issue=str(error))]
    else:
        raise AssertionError("Figure 4 coherent numerical mutation was accepted")
    if readback is not None:
        bad_artists = deepcopy(readback)
        paths = [line for axis in bad_artists["axes"] for line in axis["lines"]
                 if line.get("gid") == "A|path|M4"]
        assert len(paths) == 1
        paths[0]["y"][0] += 0.01
        try:
            audit_artists(number, root, tables, bad_artists)
        except AssertionError as error:
            results.append(dict(fault="Figure4A_artist_coordinate_change", rejected=True, new_issue=str(error)))
        else:
            raise AssertionError("Figure 4 artist-coordinate mutation was accepted")
    return results
